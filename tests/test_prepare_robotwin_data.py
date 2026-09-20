import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "challenge" / "prepare_robotwin_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_robotwin_data", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class PrepareRobotwinDataTests(unittest.TestCase):
    def _write_dataset_zip(self, base: Path, *, bad_member: str | None = None) -> Path:
        archive = base / module.ARCHIVE_NAME
        root = module.ARCHIVE_ROOT
        tasks = [{"task_index": index, "task": f"instruction {index}"} for index in range(2)]
        episodes = [
            {"episode_index": index, "tasks": [f"instruction {index // 2}"]}
            for index in range(4)
        ]
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(f"{root}/meta/info.json", json.dumps({"total_tasks": 2, "total_episodes": 4}))
            bundle.writestr(f"{root}/meta/tasks.jsonl", "\n".join(map(json.dumps, tasks)) + "\n")
            bundle.writestr(f"{root}/meta/episodes.jsonl", "\n".join(map(json.dumps, episodes)) + "\n")
            for episode_index in range(4):
                bundle.writestr(
                    f"{root}/data/chunk-000/episode_{episode_index:06d}.parquet", "fixture"
                )
                bundle.writestr(
                    f"{root}/videos/chunk-000/camera/episode_{episode_index:06d}.mp4", "fixture"
                )
            if bad_member:
                bundle.writestr(bad_member, "forbidden")
        return archive

    def test_inspect_and_validate_small_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = self._write_dataset_zip(base)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            members, actual = module.inspect_archive(archive, expected_sha256=digest)
            staging = base / "staging"
            staging.mkdir()
            module._extract_members(archive, members, staging)
            summary = module.validate_dataset(
                staging / module.ARCHIVE_ROOT,
                expected_tasks=2,
                expected_episodes=4,
                expected_per_task=2,
            )
            self.assertEqual(actual, digest)
            self.assertEqual(summary.task_count, 2)
            self.assertEqual(summary.episode_count, 4)

    def test_rejects_randomized_member(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = self._write_dataset_zip(
                base, bad_member=f"{module.ARCHIVE_ROOT}/data/randomized/episode.parquet"
            )
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with self.assertRaisesRegex(module.PreparationError, "randomized"):
                module.inspect_archive(archive, expected_sha256=digest)

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = self._write_dataset_zip(
                base, bad_member=f"{module.ARCHIVE_ROOT}/data/../../outside"
            )
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with self.assertRaisesRegex(module.PreparationError, "escapes"):
                module.inspect_archive(archive, expected_sha256=digest)

    def test_rejects_unbalanced_task_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = self._write_dataset_zip(base)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            members, _ = module.inspect_archive(archive, expected_sha256=digest)
            staging = base / "staging"
            staging.mkdir()
            module._extract_members(archive, members, staging)
            episodes = staging / module.ARCHIVE_ROOT / "meta" / "episodes.jsonl"
            rows = [json.loads(line) for line in episodes.read_text().splitlines()]
            rows[-1]["tasks"] = ["instruction 0"]
            episodes.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(module.PreparationError, "episodes per task"):
                module.validate_dataset(
                    staging / module.ARCHIVE_ROOT,
                    expected_tasks=2,
                    expected_episodes=4,
                    expected_per_task=2,
                )


if __name__ == "__main__":
    unittest.main()
