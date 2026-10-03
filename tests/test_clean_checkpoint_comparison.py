"""Lightweight parser tests without model/GPU dependencies."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
import shutil
import subprocess

PATH = Path(__file__).resolve().parents[1] / "scripts/challenge/compare_clean_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("comparison", PATH)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class StatsTests(unittest.TestCase):
    TEXT = "Result: 2 done, 0 skipped\nlift_pot 70 YES 1/1 100.0%\nopen_laptop 80 YES 0/1 0.0%\nSummary: total 150s, success 1/2, overall rate 50.0%\n"

    def test_complete(self):
        result = comparison.parse_stats(self.TEXT, tasks=2)
        self.assertEqual(result["rate"], 0.5)
        self.assertEqual(result["tasks"]["lift_pot"]["seconds"], 70)

    def test_skipped(self):
        with self.assertRaises(ValueError):
            comparison.parse_stats(self.TEXT.replace("2 done, 0 skipped", "1 done, 1 skipped"), tasks=2)

    def test_missing_episode(self):
        with self.assertRaises(ValueError):
            comparison.parse_stats(self.TEXT.replace("YES 0/1", "NO 0/0"), tasks=2)

    def test_duplicate_task(self):
        with self.assertRaises(ValueError):
            comparison.parse_stats(self.TEXT.replace("open_laptop", "lift_pot"), tasks=2)

    def test_inconsistent_summary(self):
        with self.assertRaises(ValueError):
            comparison.parse_stats(self.TEXT.replace("success 1/2", "success 2/2"), tasks=2)

    def test_multiple_episodes(self):
        text = self.TEXT.replace("1/1", "7/10").replace("0/1", "3/10").replace("success 1/2", "success 10/20")
        self.assertEqual(comparison.parse_stats(text, tasks=2, episodes=10)["total"], 20)

    def test_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.txt"
            path.write_text("# diagnostic\nlift_pot\nopen_laptop\n", encoding="utf-8")
            self.assertEqual(comparison.read_tasks(path), ["lift_pot", "open_laptop"])
            for invalid in ("lift_pot\nlift_pot\n", "unknown_task\n", "# empty\n"):
                path.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValueError):
                    comparison.read_tasks(path)

    def test_episode_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "eval_results/lift_pot/episodes.jsonl"
            path.parent.mkdir(parents=True)
            row = {"task": "lift_pot", "episode": 0, "seed": 200000,
                   "instruction": "lift the pot", "success": True, "task_config": "demo_clean"}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertTrue(comparison.read_episode_audit(root, ["lift_pot"], 1)["lift_pot"][0]["success"])
            path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaises(ValueError):
                comparison.read_episode_audit(root, ["lift_pot"], 2)


@unittest.skipUnless(shutil.which("bash"), "Bash is required for launcher subset validation")
class LauncherSubsetTests(unittest.TestCase):
    def run_subset(self, contents):
        launcher = (PATH.parents[2] / "experiment/robotwin/start_robotwin_infer_and_eval.sh").read_text(encoding="utf-8")
        section = launcher.split("# ===== Full task list (50) =====", 1)[1].split("# ===== Compute inference slot count =====", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.txt"
            path.write_text(contents, encoding="utf-8")
            return subprocess.run(["bash", "-c", 'tasks_file=$1; num_tasks=50; ' + section + '\nprintf "%s\\n" "${task_queue[@]}"',
                                   "subset-test", str(path)], text=True, capture_output=True)

    def test_subset_and_crlf(self):
        result = self.run_subset("# clean\r\nlift_pot\r\nopen_laptop\r\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["lift_pot", "open_laptop"])

    def test_reject_unknown_duplicate_empty(self):
        for contents in ("unknown_task\n", "lift_pot\nlift_pot\n", "# empty\n"):
            result = self.run_subset(contents)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
