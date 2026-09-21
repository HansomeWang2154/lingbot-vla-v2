import unittest
from pathlib import Path

import yaml


class Robotwin4090LoraConfigTest(unittest.TestCase):
    def test_single_gpu_lora_config_parses_and_is_memory_safe(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "vla"
            / "robotwin"
            / "robotwin_4090_lora.yaml"
        )
        with open(config_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.assertEqual(set(config), {"model", "data", "train"})
        self.assertTrue(
            config["model"]["model_path"].endswith(
                "/checkpoints/global_step_50000/hf_ckpt"
            )
        )
        self.assertIn("clean", config["data"]["norm_stats_file"].lower())
        train = config["train"]
        self.assertTrue(train["use_lora"])
        self.assertEqual(train["data_parallel_mode"], "ddp")
        self.assertEqual(train["optimizer"], "adamw")
        self.assertEqual(train["micro_batch_size"], 2)
        self.assertEqual(train["gradient_accumulation_steps"], 2)
        self.assertEqual(train["global_batch_size"], 4)
        self.assertFalse(train["enable_mixed_precision"])
        self.assertFalse(train["enable_fp32"])
        self.assertTrue(train["enable_gradient_checkpointing"])
        self.assertEqual(train["align_params"], {})
        self.assertFalse(train["save_hf_weights"])
        self.assertIn("clean", config["data"]["train_path"].lower())


if __name__ == "__main__":
    unittest.main()
