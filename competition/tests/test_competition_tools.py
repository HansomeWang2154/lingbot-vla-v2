from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from competition.package_submission import build_entries, write_zip
from competition.validate_results import DEFAULT_TASKS, load_tasks, validate_document


ROOT = Path(__file__).resolve().parents[1]


def valid_document() -> dict:
    tasks = load_tasks(DEFAULT_TASKS)
    return {
        "schema_version": 1,
        "submission": {
            "team_name": "test-team",
            "model_name": "one-cotrain-model",
            "checkpoint_id": "sha256:deadbeef",
        },
        "training": {
            "strategy": "cotrain_50_tasks",
            "single_checkpoint": True,
            "train_splits": ["clean"],
            "randomized_training_data_used": False,
        },
        "evaluation": {
            split: {
                "tasks": {
                    task: {"attempts": 100, "successes": 0}
                    for task in tasks
                }
            }
            for split in ("clean", "randomized")
        },
    }


class ResultsValidationTests(unittest.TestCase):
    def test_template_schema_and_task_manifest(self) -> None:
        template = json.loads((ROOT / "results.template.json").read_text(encoding="utf-8"))
        errors, summary = validate_document(template, allow_placeholders=True)
        self.assertEqual(errors, [])
        self.assertEqual(summary["splits"]["clean"]["task_count"], 50)

    def test_valid_complete_document(self) -> None:
        errors, summary = validate_document(valid_document())
        self.assertEqual(errors, [])
        self.assertEqual(summary["splits"]["randomized"]["total_attempts"], 5000)
        self.assertEqual(summary["macro_split_success_rate"], 0.0)

    def test_rejects_randomized_training_and_missing_task(self) -> None:
        document = valid_document()
        document["training"]["randomized_training_data_used"] = True
        document["training"]["train_splits"].append("randomized")
        document["evaluation"]["clean"]["tasks"].pop("lift_pot")
        errors, _ = validate_document(document)
        joined = "\n".join(errors)
        self.assertIn("evaluation-only", joined)
        self.assertIn("missing 1 task", joined)

    def test_rejects_bool_and_out_of_range_successes(self) -> None:
        document = valid_document()
        document["evaluation"]["clean"]["tasks"]["lift_pot"]["successes"] = True
        document["evaluation"]["randomized"]["tasks"]["lift_pot"]["successes"] = 101
        errors, _ = validate_document(document)
        self.assertGreaterEqual(len(errors), 2)


class PackagingTests(unittest.TestCase):
    def test_package_has_configured_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            checkpoint = temp / "one-checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            results = temp / "results.json"
            results.write_text(json.dumps(valid_document()), encoding="utf-8")
            report = temp / "report.md"
            report.write_text("# Complete report\nNo placeholders remain.\n", encoding="utf-8")
            entries = build_entries(checkpoint, results, report, ROOT / "submission_layout.json")
            archive = temp / "submission.zip"
            write_zip(entries, archive)
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(
                    sorted(zipped.namelist()),
                    [
                        "submission/checkpoint/config.json",
                        "submission/reproduction_report.md",
                        "submission/results.json",
                    ],
                )

    def test_report_placeholders_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            checkpoint = temp / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "weights.bin").write_bytes(b"weights")
            results = temp / "results.json"
            results.write_text(json.dumps(valid_document()), encoding="utf-8")
            report = temp / "report.md"
            report.write_text("TODO", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "placeholder"):
                build_entries(checkpoint, results, report, ROOT / "submission_layout.json")


if __name__ == "__main__":
    unittest.main()
