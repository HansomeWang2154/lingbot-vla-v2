import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "challenge"
    / "convert_robotwin_v21_to_v30.py"
)
SPEC = importlib.util.spec_from_file_location("convert_robotwin_v21_to_v30", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class ConvertRobotwinV21ToV30Tests(unittest.TestCase):
    def _source(self, output_root: Path) -> tuple[Path, module.prepare.DatasetSummary]:
        source = output_root / module.V21_ROOT
        (source / "meta").mkdir(parents=True)
        (source / "data" / "chunk-000").mkdir(parents=True)
        (source / "videos" / "chunk-000" / "camera").mkdir(parents=True)
        (source / "meta" / "info.json").write_text(
            json.dumps({"codebase_version": "v2.1", "total_episodes": 2500, "total_tasks": 2413}),
            encoding="utf-8",
        )
        (source / "meta" / "tasks.jsonl").write_text("{}\n", encoding="utf-8")
        (source / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
        (source / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"data")
        (source / "videos" / "chunk-000" / "camera" / "episode_000000.mp4").write_bytes(b"video")
        summary = module.prepare.DatasetSummary(
            competition_task_count=50,
            instruction_count=2413,
            episode_count=2500,
            episodes_per_competition_task=50,
            instruction_reference_counts={},
            instructions_by_index={},
            group_instruction_counts={},
        )
        provenance = {
            "archive": module.prepare.ARCHIVE_NAME,
            "archive_sha256": module.prepare.ARCHIVE_SHA256,
            "competition_task_count": 50,
            "dataset_format": "LeRobot v2.1",
            "dataset_root": str(source.resolve()),
            "episode_count": 2500,
            "instruction_catalog_count": 2413,
            "source": "TianxingChen/RoboTwin2.0::lerobot_dataset/RoboTwin_lerobot_v21.zip",
            "training_scope": "competition-authorized clean trajectories only",
        }
        (output_root / module.SOURCE_PROVENANCE).write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        return source, summary

    @staticmethod
    def _fake_converter(staging_parent: Path) -> None:
        staged = staging_parent / module.V21_ROOT
        info = staged / "meta" / "info.json"
        replacement = info.with_suffix(".new")
        replacement.write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0",
                    "total_episodes": 2500,
                    "total_tasks": 2413,
                }
            ),
            encoding="utf-8",
        )
        os.replace(replacement, info)
        (staged / "meta" / "tasks.jsonl").unlink()
        (staged / "meta" / "episodes.jsonl").unlink()
        (staged / "meta" / "tasks.parquet").write_bytes(b"tasks")
        episodes = staged / "meta" / "episodes" / "chunk-000"
        episodes.mkdir(parents=True)
        (episodes / "file-000.parquet").write_bytes(b"episodes")

    def test_conversion_is_atomic_preserves_source_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            source, summary = self._source(output_root)
            original_info = (source / "meta" / "info.json").read_bytes()
            runner_calls = []

            def runner(staging_parent: Path) -> None:
                staged_info = staging_parent / module.V21_ROOT / "meta" / "info.json"
                self.assertEqual(
                    os.stat(source / "meta" / "info.json").st_ino,
                    os.stat(staged_info).st_ino,
                )
                runner_calls.append(staging_parent)
                self._fake_converter(staging_parent)

            with mock.patch.object(module.prepare, "validate_dataset", return_value=summary):
                destination, converted = module.convert_and_publish(
                    source, output_root, runner=runner
                )
                self.assertTrue(converted)
                self.assertEqual(destination.name, module.V30_ROOT)
                self.assertEqual((source / "meta" / "info.json").read_bytes(), original_info)
                self.assertEqual(
                    (output_root / module.TRAINING_LIST).read_text(encoding="utf-8"),
                    f"robotwin {destination.resolve()}\n",
                )
                provenance = json.loads(
                    (output_root / module.SOURCE_PROVENANCE).read_text(encoding="utf-8")
                )
                self.assertEqual(provenance["dataset_format"], "LeRobot v3.0")
                self.assertEqual(provenance["source_dataset_format"], "LeRobot v2.1")
                self.assertFalse(provenance["conversion"]["push_to_hub"])

                destination_again, converted_again = module.convert_and_publish(
                    source,
                    output_root,
                    runner=lambda _: self.fail("idempotent run invoked converter"),
                )
                self.assertEqual(destination_again, destination)
                self.assertFalse(converted_again)
                module.validate_prepared(source, output_root)

            self.assertEqual(len(runner_calls), 1)
            self.assertEqual(list(output_root.glob(f".{module.V30_ROOT}.converting-*")), [])

    def test_failed_conversion_removes_only_its_staging_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            source, summary = self._source(output_root)
            unrelated = output_root / ".keep-me"
            unrelated.mkdir()
            old_list = output_root / module.TRAINING_LIST
            old_list.write_text(f"robotwin {source.resolve()}\n", encoding="utf-8")

            def fail(_: Path) -> None:
                raise module.ConversionError("synthetic failure")

            with mock.patch.object(module.prepare, "validate_dataset", return_value=summary):
                with self.assertRaisesRegex(module.ConversionError, "synthetic failure"):
                    module.convert_and_publish(source, output_root, runner=fail)

            self.assertTrue(unrelated.is_dir())
            self.assertFalse((output_root / module.V30_ROOT).exists())
            self.assertEqual(old_list.read_text(encoding="utf-8"), f"robotwin {source.resolve()}\n")
            self.assertEqual(list(output_root.glob(f".{module.V30_ROOT}.converting-*")), [])

    def test_rejects_invalid_existing_destination_without_replacing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            source, summary = self._source(output_root)
            destination = output_root / module.V30_ROOT
            (destination / "meta").mkdir(parents=True)
            (destination / "meta" / "info.json").write_text(
                json.dumps({"codebase_version": "v2.1"}), encoding="utf-8"
            )
            with mock.patch.object(module.prepare, "validate_dataset", return_value=summary):
                with self.assertRaisesRegex(module.ConversionError, "codebase_version"):
                    module.convert_and_publish(source, output_root)
            self.assertTrue(destination.is_dir())

    def test_rejects_provenance_with_unreviewed_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            source, summary = self._source(output_root)
            path = output_root / module.SOURCE_PROVENANCE
            provenance = json.loads(path.read_text(encoding="utf-8"))
            provenance["archive_sha256"] = "0" * 64
            path.write_text(json.dumps(provenance), encoding="utf-8")
            with mock.patch.object(module.prepare, "validate_dataset", return_value=summary):
                with self.assertRaisesRegex(module.ConversionError, "SHA-256"):
                    module.convert_and_publish(source, output_root)


if __name__ == "__main__":
    unittest.main()
