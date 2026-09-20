import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from lingbotvla.utils.lora_utils import (
    add_lora_to_model,
    get_lora_state_dict,
    is_lora_training_checkpoint,
    load_lora_training_checkpoint,
    load_lora_state_dict,
    merge_lora_adapter_into_hf_checkpoint,
    resolve_lora_target_modules,
    save_lora_adapter,
    save_lora_training_checkpoint,
)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)


class _ToyVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwenvl = nn.Module()
        self.qwenvl.self_attn = _Attention()
        self.qwen_expert = nn.Module()
        self.qwen_expert.self_attn = _Attention()
        self.qwen_expert.mlp = nn.Module()
        self.qwen_expert.mlp.gate_proj = nn.Linear(4, 8, bias=False)
        self.state_proj = nn.Linear(2, 4)


class LoraUtilsTest(unittest.TestCase):
    def test_add_lora_freezes_base_and_keeps_requested_heads_trainable(self):
        fake_peft = ModuleType("peft")

        class FakeLoraConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def to_dict(self):
                return self.kwargs

        def fake_inject(config, model):
            for target in config.kwargs["target_modules"]:
                module = model.get_submodule(target)
                module.lora_A = nn.Linear(module.in_features, 2, bias=False)
                module.lora_A.requires_grad_(True)
            model.peft_config = {"default": config}
            return model

        fake_peft.LoraConfig = FakeLoraConfig
        fake_peft.inject_adapter_in_model = fake_inject
        model = _ToyVLA()
        with mock.patch.dict("sys.modules", {"peft": fake_peft}):
            result = add_lora_to_model(
                model,
                lora_rank=2,
                lora_alpha=4,
                lora_target_modules=["q_proj"],
                modules_to_save=["state_proj"],
            )

        self.assertIs(result, model)
        self.assertFalse(model.qwenvl.self_attn.q_proj.weight.requires_grad)
        self.assertFalse(model.qwen_expert.self_attn.q_proj.weight.requires_grad)
        self.assertTrue(model.qwen_expert.self_attn.q_proj.lora_A.weight.requires_grad)
        self.assertTrue(model.state_proj.weight.requires_grad)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_lora_adapter(model, temp_dir)
            with open(Path(temp_dir) / "adapter_config.json", encoding="utf-8") as file:
                exported_config = json.load(file)
            self.assertEqual(exported_config["r"], 2)
            self.assertEqual(exported_config["lora_alpha"], 4)

    def test_action_expert_scope_resolves_exact_linear_names(self):
        model = _ToyVLA()
        targets = resolve_lora_target_modules(model, ["q_proj", "gate_proj"], "action_expert")
        self.assertEqual(
            targets,
            ["qwen_expert.self_attn.q_proj", "qwen_expert.mlp.gate_proj"],
        )

    def test_all_scope_includes_vlm(self):
        model = _ToyVLA()
        targets = resolve_lora_target_modules(model, ["q_proj"], "all")
        self.assertEqual(targets, ["qwenvl.self_attn.q_proj", "qwen_expert.self_attn.q_proj"])

    def test_adapter_export_contains_only_trainable_parameters(self):
        model = _ToyVLA()
        model.requires_grad_(False)
        model.qwen_expert.self_attn.q_proj.weight.requires_grad_(True)
        model.state_proj.weight.requires_grad_(True)
        model.state_proj.bias.requires_grad_(True)

        expected = {
            "qwen_expert.self_attn.q_proj.weight",
            "state_proj.weight",
            "state_proj.bias",
        }
        self.assertEqual(set(get_lora_state_dict(model)), expected)
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = save_lora_adapter(model, temp_dir, metadata={"global_step": 7})
            self.assertTrue(Path(weights_path).is_file())
            self.assertEqual(set(load_lora_state_dict(temp_dir)), expected)
            with open(Path(temp_dir) / "adapter_config.json", encoding="utf-8") as file:
                config = json.load(file)
            self.assertEqual(config["training_metadata"]["global_step"], 7)
            self.assertEqual(set(config["trainable_parameter_names"]), expected)

    def test_invalid_scope_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, "lora_target_scope"):
            resolve_lora_target_modules(_ToyVLA(), ["q_proj"], "vision_only")

    def test_offline_merge_produces_base_keyspace_and_expected_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_dir = root / "base"
            adapter_dir = root / "adapter"
            output_dir = root / "merged"
            base_dir.mkdir()
            adapter_dir.mkdir()
            base_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            base_state = {
                "model.qwen_expert.self_attn.q_proj.weight": base_weight,
                "model.state_proj.weight": torch.zeros(4, 2),
                "model.state_proj.bias": torch.zeros(4),
            }
            save_file(base_state, str(base_dir / "model.safetensors"))
            (base_dir / "config.json").write_text('{"model_type": "toy"}', encoding="utf-8")

            lora_a = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            lora_b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
            adapter_state = {
                "model.qwen_expert.self_attn.q_proj.lora_A.default.weight": lora_a,
                "model.qwen_expert.self_attn.q_proj.lora_B.default.weight": lora_b,
                "model.state_proj.weight": torch.ones(4, 2),
                "model.state_proj.bias": torch.full((4,), 2.0),
            }
            save_file(adapter_state, str(adapter_dir / "adapter_model.safetensors"))
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"r": 2, "lora_alpha": 4, "use_dora": False, "use_rslora": False}),
                encoding="utf-8",
            )

            result = merge_lora_adapter_into_hf_checkpoint(
                str(base_dir), str(adapter_dir), str(output_dir)
            )
            self.assertEqual(Path(result), output_dir)
            merged = load_file(str(output_dir / "model.safetensors"))
            expected_weight = base_weight + (lora_b @ lora_a) * 2
            torch.testing.assert_close(merged["model.qwen_expert.self_attn.q_proj.weight"], expected_weight)
            torch.testing.assert_close(merged["model.state_proj.weight"], torch.ones(4, 2))
            torch.testing.assert_close(merged["model.state_proj.bias"], torch.full((4,), 2.0))
            self.assertFalse(any("lora_" in key for key in merged))
            self.assertTrue((output_dir / "config.json").is_file())
            self.assertTrue((output_dir / "lora_merge_manifest.json").is_file())

    def test_lightweight_training_checkpoint_round_trip_excludes_frozen_base(self):
        model = _ToyVLA()
        model.frozen_blob = nn.Parameter(torch.zeros(1024, 1024), requires_grad=False)
        model.requires_grad_(False)
        model.qwen_expert.self_attn.q_proj.weight.requires_grad_(True)
        model.state_proj.weight.requires_grad_(True)
        model.state_proj.bias.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        loss = sum(parameter.sum() for parameter in model.parameters() if parameter.requires_grad)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        expected_weights = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        extra_state = {
            "global_step": 17,
            "lr_scheduler": {"last_epoch": 17},
            "train_dataloader": {"index": 4},
            "environ_meter": {"seen": 68},
            "torch_rng_state": torch.get_rng_state(),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "global_step_17"
            save_lora_training_checkpoint(
                model,
                optimizer,
                checkpoint_dir,
                extra_state,
                metadata={"global_step": 17},
            )
            self.assertTrue(is_lora_training_checkpoint(checkpoint_dir))
            self.assertFalse((checkpoint_dir / "model").exists())
            self.assertFalse((checkpoint_dir / "ema").exists())
            checkpoint_bytes = sum(
                path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file()
            )
            self.assertLess(checkpoint_bytes, model.frozen_blob.numel() * model.frozen_blob.element_size())

            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data.zero_()
            optimizer.state.clear()
            restored = load_lora_training_checkpoint(model, optimizer, checkpoint_dir)
            self.assertEqual(restored["global_step"], 17)
            self.assertTrue(optimizer.state)
            for name, expected in expected_weights.items():
                torch.testing.assert_close(dict(model.named_parameters())[name], expected)

            (checkpoint_dir / "lora_training_checkpoint.json").unlink()
            self.assertFalse(is_lora_training_checkpoint(checkpoint_dir))
            with self.assertRaisesRegex(ValueError, "Incomplete or unsupported"):
                load_lora_training_checkpoint(model, optimizer, checkpoint_dir)


if __name__ == "__main__":
    unittest.main()
