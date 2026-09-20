# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LoRA helpers shared by VLA training entry points.

These helpers use PEFT's low-level ``inject_adapter_in_model`` API because the
LingBot-VLA checkpoint code expects the original model class rather than a
``PeftModel`` wrapper. Exports are loaded with :func:`add_lora_to_model`.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file


DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

LORA_TRAINING_CHECKPOINT_FORMAT = "lingbotvla-lora-training-checkpoint-v1"
LORA_TRAINING_MANIFEST = "lora_training_checkpoint.json"
LORA_OPTIMIZER_STATE = "optimizer.pt"
LORA_EXTRA_STATE = "extra_state.pt"


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP-style containers without importing distributed modules."""

    while hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module
    return model


def freeze_parameters(model: nn.Module) -> None:
    """Freeze the base model while leaving it in training mode."""

    model.requires_grad_(False)
    model.train()


def resolve_lora_target_modules(
    model: nn.Module,
    target_modules: Sequence[str],
    target_scope: str = "action_expert",
) -> List[str]:
    """Resolve suffixes to exact ``nn.Linear`` module names.

    Exact names prevent PEFT suffix matching from adapting the Qwen VLM when a
    single-GPU run only intends to tune the action expert. Fused MoE tensors are
    parameters rather than linears and are intentionally skipped.
    """

    if target_scope not in {"action_expert", "all"}:
        raise ValueError("lora_target_scope must be 'action_expert' or 'all'.")
    suffixes = tuple(item.strip() for item in target_modules if item.strip())
    if not suffixes:
        raise ValueError("At least one LoRA target module must be configured.")

    matches = []
    for name, module in _unwrap_model(model).named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_scope == "action_expert" and not (
            name.startswith("qwen_expert.") or ".qwen_expert." in name
        ):
            continue
        if any(name == suffix or name.endswith(f".{suffix}") for suffix in suffixes):
            matches.append(name)
    if not matches:
        raise ValueError(
            f"No nn.Linear modules matched LoRA targets {list(suffixes)!r} "
            f"in scope {target_scope!r}."
        )
    return matches


def add_lora_to_model(
    model: nn.Module,
    lora_rank: int = 4,
    lora_alpha: int = 4,
    lora_target_modules: Union[str, Sequence[str]] = DEFAULT_LORA_TARGET_MODULES,
    init_lora_weights: Union[str, bool] = "kaiming",
    pretrained_lora_path: Optional[str] = None,
    state_dict_converter=None,
    lora_target_modules_support: Optional[Iterable[str]] = None,
    lora_dropout: float = 0.0,
    lora_target_scope: str = "action_expert",
    modules_to_save: Optional[Sequence[str]] = None,
) -> nn.Module:
    """Freeze ``model`` and inject trainable LoRA adapters in-place."""

    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:  # pragma: no cover - depends on runtime extras
        raise ImportError("LoRA training requires peft; install project requirements first.") from exc

    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive.")
    if lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive.")
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in [0, 1).")
    if isinstance(lora_target_modules, str):
        requested = [item.strip() for item in lora_target_modules.split(",") if item.strip()]
    else:
        requested = [item.strip() for item in lora_target_modules if item.strip()]
    if lora_target_modules_support is not None:
        supported = set(lora_target_modules_support)
        unsupported = [item for item in requested if item not in supported]
        if unsupported:
            raise ValueError(f"LoRA target modules are not supported: {unsupported}")

    exact_targets = resolve_lora_target_modules(model, requested, lora_target_scope)
    if init_lora_weights == "kaiming":
        init_lora_weights = True
    freeze_parameters(model)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_lora_weights=init_lora_weights,
        target_modules=exact_targets,
        bias="none",
    )
    model = inject_adapter_in_model(lora_config, model)
    model.lora_alpha = lora_alpha
    model._lingbot_lora_export_config = {
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": exact_targets,
        "target_scope": lora_target_scope,
        "modules_to_save": list(modules_to_save or []),
        "use_dora": False,
        "use_rslora": False,
    }
    if modules_to_save:
        requested_modules = set(modules_to_save)
        matched_modules = set()
        for name, module in model.named_modules():
            for requested_name in requested_modules:
                if name == requested_name or name.endswith(f".{requested_name}"):
                    module.requires_grad_(True)
                    matched_modules.add(requested_name)
        missing_modules = requested_modules - matched_modules
        if missing_modules:
            raise ValueError(f"LoRA modules_to_save did not match model modules: {sorted(missing_modules)}")
    for name, param in model.named_parameters():
        if param.requires_grad or "lora_" in name:
            param.data = param.data.float()

    if pretrained_lora_path is not None:
        state_dict = load_lora_state_dict(pretrained_lora_path)
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        loaded_keys = len(state_dict) - len(unexpected_keys)
        if loaded_keys == 0:
            raise ValueError(f"No LoRA parameters were loaded from {pretrained_lora_path}.")
        print(
            f"Loaded {loaded_keys} LoRA tensors from {pretrained_lora_path}; "
            f"{len(unexpected_keys)} unexpected, {len(missing_keys)} keys unchanged."
        )
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return trainable adapter/modules-to-save tensors on CPU."""

    state = {}
    for name, param in _unwrap_model(model).named_parameters():
        if not param.requires_grad:
            continue
        tensor = param.detach()
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()
        state[name] = tensor.cpu().contiguous()
    if not state:
        raise ValueError("The model has no trainable LoRA parameters to export.")
    return state


def save_lora_adapter(model: nn.Module, output_dir: str, metadata: Optional[dict] = None) -> str:
    """Save a compact adapter checkpoint and return its weights path."""

    model = _unwrap_model(model)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    weights_path = output_path / "adapter_model.safetensors"
    temp_weights_path = output_path / "adapter_model.safetensors.tmp"
    state_dict = get_lora_state_dict(model)
    save_file(state_dict, str(temp_weights_path))
    os.replace(temp_weights_path, weights_path)

    config = {}
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict) and peft_config:
        active_config = peft_config.get("default", next(iter(peft_config.values())))
        if hasattr(active_config, "to_dict"):
            config.update(active_config.to_dict())
    config.update(getattr(model, "_lingbot_lora_export_config", {}))
    config.update(
        {
            "format": "lingbotvla-low-level-peft-adapter-v1",
            "weights": weights_path.name,
            "trainable_parameter_names": list(state_dict),
        }
    )
    if metadata:
        config["training_metadata"] = metadata
    config_path = output_path / "adapter_config.json"
    temp_config_path = output_path / "adapter_config.json.tmp"
    with open(temp_config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2, default=str)
    os.replace(temp_config_path, config_path)
    return str(weights_path)


def _atomic_torch_save(value, output_path: Path) -> None:
    """Write a torch payload atomically within one filesystem."""

    temp_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def is_lora_training_checkpoint(checkpoint_dir: Union[str, Path]) -> bool:
    """Return whether a checkpoint has every file required for exact resume."""

    checkpoint_path = Path(checkpoint_dir)
    manifest_path = checkpoint_path / LORA_TRAINING_MANIFEST
    if not manifest_path.is_file():
        return False
    try:
        with open(manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("format") != LORA_TRAINING_CHECKPOINT_FORMAT:
        return False
    return all(
        (
            (checkpoint_path / "lora_adapter" / "adapter_model.safetensors").is_file(),
            (checkpoint_path / "lora_adapter" / "adapter_config.json").is_file(),
            (checkpoint_path / LORA_OPTIMIZER_STATE).is_file(),
            (checkpoint_path / LORA_EXTRA_STATE).is_file(),
        )
    )


def save_lora_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Union[str, Path],
    extra_state: Dict[str, Any],
    metadata: Optional[dict] = None,
) -> str:
    """Save only trainable LoRA/head tensors plus optimizer and resume state.

    The completion manifest is written last. Auto-resume therefore ignores a
    checkpoint interrupted while any payload is still being written. The
    frozen base checkpoint is deliberately not serialized.
    """

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_path / LORA_TRAINING_MANIFEST
    if manifest_path.exists():
        manifest_path.unlink()

    adapter_weights = Path(
        save_lora_adapter(model, str(checkpoint_path / "lora_adapter"), metadata=metadata)
    )
    optimizer_path = checkpoint_path / LORA_OPTIMIZER_STATE
    extra_state_path = checkpoint_path / LORA_EXTRA_STATE
    _atomic_torch_save(optimizer.state_dict(), optimizer_path)
    _atomic_torch_save(extra_state, extra_state_path)

    payload_paths = {
        "adapter": adapter_weights,
        "optimizer": optimizer_path,
        "extra_state": extra_state_path,
    }
    manifest = {
        "format": LORA_TRAINING_CHECKPOINT_FORMAT,
        "files": {name: str(path.relative_to(checkpoint_path)) for name, path in payload_paths.items()},
        "payload_bytes": {name: path.stat().st_size for name, path in payload_paths.items()},
        "frozen_base_included": False,
    }
    if metadata:
        manifest["training_metadata"] = metadata
    temp_manifest_path = checkpoint_path / f".{LORA_TRAINING_MANIFEST}.tmp-{os.getpid()}"
    try:
        with open(temp_manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2, default=str)
        os.replace(temp_manifest_path, manifest_path)
    finally:
        if temp_manifest_path.exists():
            temp_manifest_path.unlink()
    return str(checkpoint_path)


def load_lora_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Restore adapter/head weights, optimizer state, and loop state."""

    checkpoint_path = Path(checkpoint_dir)
    if not is_lora_training_checkpoint(checkpoint_path):
        raise ValueError(f"Incomplete or unsupported LoRA training checkpoint: {checkpoint_path}")

    adapter_state = load_lora_state_dict(str(checkpoint_path / "lora_adapter"))
    expected_trainable = {
        name for name, parameter in _unwrap_model(model).named_parameters() if parameter.requires_grad
    }
    if set(adapter_state) != expected_trainable:
        missing = sorted(expected_trainable - set(adapter_state))
        extra = sorted(set(adapter_state) - expected_trainable)
        raise KeyError(
            "LoRA checkpoint trainable tensors do not match the current model: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    _, unexpected_keys = _unwrap_model(model).load_state_dict(adapter_state, strict=False)
    if unexpected_keys:
        raise KeyError(f"LoRA checkpoint has unexpected model tensors: {unexpected_keys[:10]}")

    optimizer_state = torch.load(
        checkpoint_path / LORA_OPTIMIZER_STATE,
        map_location="cpu",
        weights_only=True,
    )
    optimizer.load_state_dict(optimizer_state)
    return torch.load(
        checkpoint_path / LORA_EXTRA_STATE,
        map_location="cpu",
        weights_only=False,
    )


def _resolve_adapter_weights_path(file_path: str) -> str:
    path = Path(file_path)
    if path.is_dir():
        for filename in ("adapter_model.safetensors", "adapter_model.bin"):
            candidate = path / filename
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {file_path}.")
    return str(path)


def load_lora_state_dict(file_path: str, torch_dtype=None):
    return load_state_dict(_resolve_adapter_weights_path(file_path), torch_dtype=torch_dtype)


_LORA_A_PATTERN = re.compile(r"^(?P<prefix>.+)\.lora_A\.(?P<adapter>[^.]+)\.weight$")


def _build_merge_updates(
    adapter_path: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Build additive LoRA deltas and direct task-head replacements."""

    adapter_dir = Path(adapter_path)
    config_path = (
        adapter_dir / "adapter_config.json"
        if adapter_dir.is_dir()
        else adapter_dir.parent / "adapter_config.json"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter_config.json next to {adapter_path}.")
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    if config.get("use_dora") or config.get("use_rslora"):
        raise ValueError("Offline merge currently supports standard LoRA only (no DoRA or RS-LoRA).")

    state = load_lora_state_dict(adapter_path, torch_dtype=torch.float32)
    alpha = float(config.get("lora_alpha", 0))
    configured_rank = int(config.get("r", 0))
    if alpha <= 0 or configured_rank <= 0:
        raise ValueError("adapter_config.json must contain positive lora_alpha and r values.")

    additive_updates = {}
    consumed = set()
    for key, lora_a in state.items():
        match = _LORA_A_PATTERN.fullmatch(key)
        if match is None:
            continue
        prefix = match.group("prefix")
        adapter_name = match.group("adapter")
        lora_b_key = f"{prefix}.lora_B.{adapter_name}.weight"
        if lora_b_key not in state:
            raise ValueError(f"Adapter is missing matching tensor {lora_b_key!r}.")
        lora_b = state[lora_b_key]
        rank = lora_a.shape[0]
        if rank != configured_rank or lora_b.shape[1] != rank:
            raise ValueError(
                f"LoRA rank mismatch for {prefix}: config={configured_rank}, "
                f"A={rank}, B={lora_b.shape[1]}."
            )
        base_weight_key = f"{prefix}.weight"
        additive_updates[base_weight_key] = torch.matmul(lora_b, lora_a).mul_(alpha / rank)
        consumed.update({key, lora_b_key})

    unsupported_lora_keys = [key for key in state if "lora_" in key and key not in consumed]
    if unsupported_lora_keys:
        raise ValueError(f"Unsupported LoRA tensors in adapter export: {unsupported_lora_keys[:5]}")
    if not additive_updates:
        raise ValueError("No LoRA A/B tensor pairs were found in the adapter export.")
    direct_updates = {key: value for key, value in state.items() if key not in consumed}
    return additive_updates, direct_updates


def merge_lora_adapter_into_hf_checkpoint(
    base_model_dir: str,
    adapter_path: str,
    output_dir: str,
    overwrite: bool = False,
) -> str:
    """Merge a LingBot-VLA LoRA export into a deployable HF checkpoint.

    The operation is offline and streaming by existing HF shard: it never
    downloads weights and does not instantiate the 6B model. Base assets and
    shard names are preserved, so the output remains compatible with the
    strict deployment loader.
    """

    base_path = Path(base_model_dir).resolve()
    output_path = Path(output_dir).resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"Base HF checkpoint directory does not exist: {base_path}")
    filesystem_root = Path(output_path.anchor).resolve()
    if output_path in {filesystem_root, Path.home().resolve()} or (output_path / ".git").exists():
        raise ValueError(f"Refusing to use a broad or repository directory as merge output: {output_path}")
    if output_path == base_path or base_path in output_path.parents:
        raise ValueError("output_dir must not be the base checkpoint or a child of it.")
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_path}")

    index_path = base_path / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path, encoding="utf-8") as file:
            index = json.load(file)
        weight_files = list(dict.fromkeys(index.get("weight_map", {}).values()))
    elif (base_path / "model.safetensors").is_file():
        index = None
        weight_files = ["model.safetensors"]
    else:
        raise FileNotFoundError(
            f"Base checkpoint must contain model.safetensors or model.safetensors.index.json: {base_path}"
        )
    missing_shards = [name for name in weight_files if not (base_path / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(f"Base checkpoint is missing weight shards: {missing_shards}")

    additive_updates, direct_updates = _build_merge_updates(adapter_path)
    pending_additive = set(additive_updates)
    pending_direct = set(direct_updates)
    temp_path = output_path.parent / f".{output_path.name}.merge-tmp-{os.getpid()}"
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True)
    try:
        weight_file_set = set(weight_files)
        for item in base_path.iterdir():
            if item.name in weight_file_set or item.name == "model.safetensors.index.json":
                continue
            destination = temp_path / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)

        for filename in weight_files:
            shard = load_file(str(base_path / filename), device="cpu")
            for key, delta in additive_updates.items():
                if key not in shard:
                    continue
                if shard[key].shape != delta.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: base={tuple(shard[key].shape)}, "
                        f"delta={tuple(delta.shape)}"
                    )
                original_dtype = shard[key].dtype
                shard[key] = (shard[key].float() + delta).to(original_dtype).contiguous()
                pending_additive.discard(key)
            for key, replacement in direct_updates.items():
                if key not in shard:
                    continue
                if shard[key].shape != replacement.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: base={tuple(shard[key].shape)}, adapter={tuple(replacement.shape)}"
                    )
                shard[key] = replacement.to(shard[key].dtype).contiguous()
                pending_direct.discard(key)
            save_file(shard, str(temp_path / filename), metadata={"format": "pt"})
            del shard

        if pending_additive or pending_direct:
            missing = sorted(pending_additive | pending_direct)
            raise KeyError(f"Adapter tensors do not match base checkpoint parameters: {missing[:10]}")
        if index is not None:
            shutil.copy2(index_path, temp_path / index_path.name)
        manifest = {
            "format": "lingbotvla-merged-lora-hf-v1",
            "base_model_dir": str(base_path),
            "adapter_path": str(Path(adapter_path).resolve()),
            "merged_lora_weights": sorted(additive_updates),
            "replaced_task_parameters": sorted(direct_updates),
        }
        with open(temp_path / "lora_merge_manifest.json", "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

        if output_path.exists():
            if any(output_path.iterdir()):
                shutil.rmtree(output_path)
            else:
                output_path.rmdir()
        os.replace(temp_path, output_path)
    except Exception:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise
    return str(output_path)


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as file:
        for key in file.keys():
            state_dict[key] = file.get_tensor(key)
            if torch_dtype is not None:
                state_dict[key] = state_dict[key].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict[key] = value.to(torch_dtype)
    return state_dict
