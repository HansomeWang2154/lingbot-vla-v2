"""Lightweight parser tests without model/GPU dependencies."""
import importlib.util
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
