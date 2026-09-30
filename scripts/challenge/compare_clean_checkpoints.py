#!/usr/bin/env python3
"""Sequential clean-only checkpoint screening; never train or submit a model.

Run with the inference environment's Python after sourcing .challenge.env.
Each invocation requires a new output directory; interrupted runs are retained,
not silently reused or overwritten. Launcher traps manage detached workers.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_stats(text, tasks=50, episodes=1):
    """Require complete task rows as well as aggregate counts (fail closed)."""
    text = ANSI.sub("", text)
    done = re.search(r"Result:\s*(\d+) done,\s*(\d+) skipped", text)
    summary = re.search(r"Summary: total (\d+)s, success (\d+)/(\d+)", text)
    rows = re.findall(r"^([a-z][a-z0-9_]+)\s+(\d+)\s+YES\s+(\d+)/(\d+)\s+[\d.]+%\s*$", text, re.M)
    if not done or not summary:
        raise ValueError("Missing completion/summary markers")
    duration, success, total = map(int, summary.groups())
    if tuple(map(int, done.groups())) != (tasks, 0):
        raise ValueError("Skipped or incomplete simulator tasks")
    if len(rows) != tasks or len({r[0] for r in rows}) != tasks:
        raise ValueError("Missing/duplicate task result rows")
    if any(int(r[3]) != episodes or not 0 <= int(r[2]) <= episodes for r in rows):
        raise ValueError("Incomplete/invalid episode counts")
    if total != tasks * episodes or success != sum(int(r[2]) for r in rows):
        raise ValueError("Inconsistent aggregate counts")
    return {"success": success, "total": total, "rate": success / total,
            "sim_seconds": duration,
            "tasks": {r[0]: {"success": int(r[2]), "total": int(r[3]),
                              "seconds": int(r[1])} for r in rows}}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(command, cwd=ROOT):
    """Propagate stop requests to the launcher's own cleanup traps."""
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    previous = {}

    def stop(signum, frame):
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Worker teardown is the launcher's responsibility; never kill
            # unrelated groups or escalate to a blanket GPU/process kill.
            raise RuntimeError("Launcher did not stop; inspect its PID files")
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, stop)
    try:
        result = process.wait()
        if result:
            raise RuntimeError(f"Command failed with exit code {result}: {command[0]}")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate/print plan without running or writing")
    args = parser.parse_args()
    required = ["CHALLENGE_VLA_MODEL_DIR", "CHALLENGE_ROBOTWIN_ROOT",
                "CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX", "CHALLENGE_CONDA_SH",
                "CHALLENGE_ENV_NAME", "CHALLENGE_QWEN3_DIR"]
    env = {key: os.environ[key] for key in required}
    config = args.training_run / "lingbotvla_cli.yaml"
    base = Path(env["CHALLENGE_VLA_MODEL_DIR"])
    sim = Path(env["CHALLENGE_ROBOTWIN_ROOT"])
    expected_revision = os.environ.get("CHALLENGE_ROBOTWIN_REVISION", "13c3c47ff4312dd62484bcd51be034af55c062d1")
    revision = subprocess.check_output(["git", "-C", str(sim), "rev-parse", "HEAD"], text=True).strip()
    if revision != expected_revision:
        raise ValueError(f"Unexpected simulator revision: {revision}")
    if not config.is_file() or not base.is_dir():
        raise FileNotFoundError("Training YAML or original base checkpoint missing")
    import yaml
    configuration = yaml.safe_load(config.read_text(encoding="utf-8"))
    data = configuration["data"]
    train_list = Path(data["train_path"])
    norm_stats = Path(data["norm_stats_file"])
    if train_list.resolve() != Path(os.environ["CHALLENGE_TRAIN_LIST"]).resolve():
        raise ValueError("Training config does not reference the approved clean training list")
    if norm_stats.resolve() != Path(os.environ["CHALLENGE_CLEAN_NORM_STATS"]).resolve():
        raise ValueError("Training config does not reference the approved clean normalization")
    stages = [{"name": "base_50k", "model": str(base)}]
    for step in (1000, 5000, 10000):
        checkpoint = args.training_run / "checkpoints" / f"global_step_{step}"
        adapter = checkpoint / "lora_adapter"
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing compact adapter: {adapter}")
        stages.append({"name": f"lora_{step}", "model": str(checkpoint / "merged_hf_ckpt"),
                       "adapter": str(adapter), "adapter_sha256": file_hash(adapter / "adapter_model.safetensors")})
    # Set one tokenizer, offline behavior and identical model/data configuration.
    os.environ.update(QWEN3VL_PATH=env["CHALLENGE_QWEN3_DIR"], HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    manifest = {"status": "planned", "sim_revision": revision, "training_config": str(config),
                "training_config_sha256": file_hash(config), "task_config": "demo_clean",
                "train_list_sha256": file_hash(train_list), "norm_stats_sha256": file_hash(norm_stats),
                "tasks": 50, "episodes": 1, "sim_seed": 0, "policy_seed": 42,
                "precision": "BF16", "compile": False, "use_length": 50,
                "purpose": "Weight-only screening under shared clean normalization; not official release reproduction",
                "stages": stages,
                "code_sha256": {p: file_hash(ROOT / p) for p in (
                    "scripts/challenge/compare_clean_checkpoints.py",
                    "experiment/robotwin/start_robotwin_infer_and_eval.sh",
                    "experiment/robotwin/eval_policy_client_lingbotvla.py",
                    "deploy/lingbot_vla_v2_policy.py")}}
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / "comparison.json"

    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    save()
    try:
        manifest["status"] = "running"
        manifest["started_unix"] = time.time()
        for stage in stages:
            stage["status"] = "running"
            save()
            print(f"COMPARISON START {stage['name']}", flush=True)
            model = Path(stage["model"])
            if "adapter" in stage and not model.exists():
                execute([sys.executable, "tools/merge_lora_adapter.py", "--base-model", str(base),
                         "--adapter", stage["adapter"], "--output", str(model)])
            if not list(model.glob("*.safetensors")):
                raise ValueError(f"No deployable safetensors: {model}")
            if "adapter" in stage:
                metadata = model / "lora_merge_manifest.json"
                if not metadata.is_file():
                    raise ValueError(f"Merged checkpoint missing provenance: {metadata}")
                provenance = json.loads(metadata.read_text(encoding="utf-8"))
                if (Path(provenance["base_model_dir"]).resolve() != base.resolve() or
                        Path(provenance["adapter_path"]).resolve() != Path(stage["adapter"]).resolve()):
                    raise ValueError(f"Merged checkpoint source mismatch: {metadata}")
            stage_output = args.output / stage["name"]
            execute(["bash", "experiment/robotwin/start_robotwin_infer_and_eval.sh",
                     "--model_path", str(model), "--training_config", str(config),
                     "--eval_workdir", str(sim), "--inference_workdir", str(ROOT),
                     "--conda_sh", env["CHALLENGE_CONDA_SH"], "--inference_env", env["CHALLENGE_ENV_NAME"],
                     "--sim_env", env["CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX"],
                     "--output_base", str(stage_output), "--num_tasks", "50", "--episodes", "1",
                     "--num_gpus", "1", "--num_per_gpu", "1", "--use_bf16", "True",
                     "--use_fp32", "False", "--use_compile", "False", "--use_length", "50",
                     "--task_config", "demo_clean", "--no_video"])
            stats = list(stage_output.glob("*/stats.txt"))
            if len(stats) != 1:
                raise ValueError("Expected exactly one stats file")
            stage["result"] = parse_stats(stats[0].read_text(encoding="utf-8"))
            stage["stats_file"] = str(stats[0])
            stage["status"] = "complete"
            save()
            print(f"COMPARISON DONE {stage['name']}: {stage['result']['success']}/50", flush=True)
        manifest["status"] = "complete"
        manifest["finished_unix"] = time.time()
        lines = ["# Clean checkpoint screening", "", "50 tasks × 1 episode; exploratory, not an official score.", "",
                 "| Model | Success | Rate | Sim seconds |", "|---|---:|---:|---:|"]
        for stage in stages:
            result = stage["result"]
            lines.append(f"| {stage['name']} | {result['success']}/50 | {result['rate']:.1%} | {result['sim_seconds']} |")
        lines += ["", "Use multi-seed validation before selecting a winner. Shared clean normalization is held fixed."]
        (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        save()
    except BaseException as error:
        manifest["status"] = "failed_or_interrupted"
        manifest["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
