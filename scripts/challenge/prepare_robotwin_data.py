#!/usr/bin/env python3
"""Safely prepare the competition-authorized RoboTwin LeRobot v2.1 archive.

The official archive is one merged LeRobot dataset (50 tasks / 2500 episodes),
not 50 independent dataset directories.  Consequently this program writes:

* ``clean_training_data.txt``: one ``robotwin <dataset-root>`` row for the
  LingBot ``MultiVLADataset`` loader; and
* ``clean_training_tasks.jsonl``: 50 audit rows proving expected task
  coverage without loading the same merged dataset 50 times.

Extraction is deliberately strict: the official archive filename and SHA-256
are pinned, members must remain below the expected archive root, links and
special files are rejected, and any path containing ``randomized`` is fatal.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


ARCHIVE_NAME = "RoboTwin_lerobot_v21.zip"
ARCHIVE_ROOT = "RoboTwin_lerobot_v21"
# Current official object from TianxingChen/RoboTwin2.0 commit 981c92a.
ARCHIVE_SHA256 = "98a1d4bfb51d6e90469c9ef8d55e004a2cba01d20374adb68b554e48e9fe2cd8"
EXPECTED_TASKS = 50
EXPECTED_EPISODES = 2500
EXPECTED_EPISODES_PER_TASK = 50
MAX_MEMBERS = 20_000
MAX_UNCOMPRESSED_BYTES = 64 * 1024**3
ALLOWED_SECTIONS = {"meta", "data", "videos"}
FORBIDDEN_COMPONENT_RE = re.compile(r"randomized", re.IGNORECASE)


class PreparationError(RuntimeError):
    """Raised when an input violates the competition data boundary."""


@dataclass(frozen=True)
class DatasetSummary:
    task_count: int
    episode_count: int
    task_episode_counts: dict[str, int]
    tasks_by_index: dict[int, str]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"Expected a JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PreparationError(
                        f"Expected an object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Invalid JSONL file {path}: {exc}") from exc
    return rows


def load_canonical_tasks(path: Path, expected_count: int = EXPECTED_TASKS) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PreparationError(f"Cannot read canonical task list {path}: {exc}") from exc
    tasks = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if len(tasks) != expected_count:
        raise PreparationError(
            f"Canonical task list has {len(tasks)} entries; expected {expected_count}"
        )
    if len(set(tasks)) != len(tasks):
        raise PreparationError("Canonical task list contains duplicate entries")
    invalid = [task for task in tasks if not re.fullmatch(r"[a-z0-9_]+", task)]
    if invalid:
        raise PreparationError(f"Invalid canonical task identifiers: {invalid}")
    return tasks


def _member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    if "\\" in name or "\x00" in name:
        raise PreparationError(f"Unsafe archive member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise PreparationError(f"Archive member escapes its root: {name!r}")
    if path.parts[0] != ARCHIVE_ROOT:
        raise PreparationError(
            f"Unexpected archive root {path.parts[0]!r}; expected {ARCHIVE_ROOT!r}"
        )
    if len(path.parts) > 1 and path.parts[1] not in ALLOWED_SECTIONS:
        raise PreparationError(f"Unexpected top-level dataset section: {name!r}")
    if any(FORBIDDEN_COMPONENT_RE.search(part) for part in path.parts):
        raise PreparationError(f"Forbidden randomized data in archive: {name!r}")
    return path


def inspect_archive(
    archive: Path,
    *,
    expected_sha256: str = ARCHIVE_SHA256,
    max_members: int = MAX_MEMBERS,
    max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
) -> tuple[list[zipfile.ZipInfo], str]:
    if archive.name != ARCHIVE_NAME:
        raise PreparationError(
            f"Archive must be named {ARCHIVE_NAME!r}; got {archive.name!r}"
        )
    if not archive.is_file():
        raise PreparationError(f"Archive does not exist: {archive}")
    actual_sha256 = sha256_file(archive)
    if actual_sha256.lower() != expected_sha256.lower():
        raise PreparationError(
            "Archive SHA-256 mismatch. Refuse to use an unreviewed dataset object: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise PreparationError(f"Invalid zip archive {archive}: {exc}") from exc
    if not members or len(members) > max_members:
        raise PreparationError(
            f"Archive member count {len(members)} is outside the allowed range 1..{max_members}"
        )

    seen: set[PurePosixPath] = set()
    total_size = 0
    for info in members:
        path = _member_path(info)
        if path in seen:
            raise PreparationError(f"Duplicate archive member: {info.filename!r}")
        seen.add(path)
        total_size += info.file_size
        if total_size > max_uncompressed_bytes:
            raise PreparationError(
                f"Archive expands beyond the {max_uncompressed_bytes} byte safety limit"
            )
        unix_mode = info.external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise PreparationError(f"Links/special files are forbidden: {info.filename!r}")

    required = {
        PurePosixPath(ARCHIVE_ROOT, "meta", "info.json"),
        PurePosixPath(ARCHIVE_ROOT, "meta", "tasks.jsonl"),
        PurePosixPath(ARCHIVE_ROOT, "meta", "episodes.jsonl"),
    }
    missing = sorted(str(path) for path in required - seen)
    if missing:
        raise PreparationError(f"Archive is missing required metadata: {missing}")
    return members, actual_sha256


def _extract_members(archive: Path, members: Sequence[zipfile.ZipInfo], staging: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for info in members:
            relative = _member_path(info)
            destination = staging.joinpath(*relative.parts)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info, "r") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def _first_int(mapping: dict, keys: Iterable[str]) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def validate_dataset(
    dataset_root: Path,
    *,
    expected_tasks: int = EXPECTED_TASKS,
    expected_episodes: int = EXPECTED_EPISODES,
    expected_per_task: int = EXPECTED_EPISODES_PER_TASK,
) -> DatasetSummary:
    if dataset_root.name != ARCHIVE_ROOT or not dataset_root.is_dir():
        raise PreparationError(f"Expected extracted dataset directory {ARCHIVE_ROOT}: {dataset_root}")
    for path in dataset_root.rglob("*"):
        if path.is_symlink():
            raise PreparationError(f"Symlink is forbidden in training data: {path}")
        if FORBIDDEN_COMPONENT_RE.search(path.name):
            raise PreparationError(f"Randomized data is forbidden in training data: {path}")

    meta = dataset_root / "meta"
    info = _load_json(meta / "info.json")
    tasks = _load_jsonl(meta / "tasks.jsonl")
    episodes = _load_jsonl(meta / "episodes.jsonl")

    task_by_index: dict[int, str] = {}
    for row in tasks:
        index, task = row.get("task_index"), row.get("task")
        if not isinstance(index, int) or not isinstance(task, str) or not task.strip():
            raise PreparationError("Each tasks.jsonl row must contain task_index:int and task:str")
        if index in task_by_index or task in task_by_index.values():
            raise PreparationError("tasks.jsonl contains duplicate task indices or task strings")
        task_by_index[index] = task
    if set(task_by_index) != set(range(expected_tasks)):
        raise PreparationError(
            f"Dataset task indices are not contiguous 0..{expected_tasks - 1}"
        )

    episode_indices: set[int] = set()
    counts: collections.Counter[str] = collections.Counter()
    for row in episodes:
        episode_index = row.get("episode_index")
        episode_tasks = row.get("tasks")
        if not isinstance(episode_index, int) or episode_index in episode_indices:
            raise PreparationError("episodes.jsonl has an invalid or duplicate episode_index")
        episode_indices.add(episode_index)
        if not isinstance(episode_tasks, list) or len(episode_tasks) != 1 or not isinstance(episode_tasks[0], str):
            raise PreparationError("Each episode must contain exactly one string in its tasks field")
        if episode_tasks[0] not in task_by_index.values():
            raise PreparationError(f"Episode references unknown task: {episode_tasks[0]!r}")
        counts[episode_tasks[0]] += 1

    if set(episode_indices) != set(range(expected_episodes)):
        raise PreparationError(
            f"Dataset episode indices are not contiguous 0..{expected_episodes - 1}"
        )
    bad_counts = {task: counts[task] for task in task_by_index.values() if counts[task] != expected_per_task}
    if bad_counts:
        raise PreparationError(
            f"Expected {expected_per_task} clean episodes per task; mismatches: {bad_counts}"
        )

    info_tasks = _first_int(info, ("total_tasks", "num_tasks"))
    info_episodes = _first_int(info, ("total_episodes", "num_episodes"))
    if info_tasks is not None and info_tasks != expected_tasks:
        raise PreparationError(f"info.json reports {info_tasks} tasks; expected {expected_tasks}")
    if info_episodes is not None and info_episodes != expected_episodes:
        raise PreparationError(
            f"info.json reports {info_episodes} episodes; expected {expected_episodes}"
        )
    if not (dataset_root / "data").is_dir() or not (dataset_root / "videos").is_dir():
        raise PreparationError("LeRobot data/ or videos/ directory is missing")

    def payload_episode_indices(section: str, suffix: str) -> list[int]:
        indices: list[int] = []
        pattern = re.compile(r"episode_(\d+)" + re.escape(suffix) + r"$")
        for path in (dataset_root / section).rglob(f"episode_*{suffix}"):
            match = pattern.fullmatch(path.name)
            if match:
                indices.append(int(match.group(1)))
        return indices

    expected_indices = set(range(expected_episodes))
    parquet_indices = payload_episode_indices("data", ".parquet")
    video_indices = payload_episode_indices("videos", ".mp4")
    if len(parquet_indices) != expected_episodes or set(parquet_indices) != expected_indices:
        raise PreparationError(
            "Parquet payload does not cover every metadata episode exactly by index"
        )
    if set(video_indices) != expected_indices:
        raise PreparationError(
            "Video payload does not cover every metadata episode by index"
        )
    return DatasetSummary(expected_tasks, expected_episodes, dict(counts), task_by_index)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifests(
    output_root: Path,
    dataset_root: Path,
    canonical_tasks: Sequence[str],
    summary: DatasetSummary,
    archive_sha256: str,
) -> tuple[Path, Path, Path]:
    training_list = output_root / "clean_training_data.txt"
    task_manifest = output_root / "clean_training_tasks.jsonl"
    provenance = output_root / "clean_training_provenance.json"
    _atomic_write(training_list, f"robotwin {dataset_root.resolve()}\n")

    task_rows = []
    for task_id in canonical_tasks:
        task_rows.append(
            json.dumps(
                {
                    "task_id": task_id,
                    "expected_clean_episodes": EXPECTED_EPISODES_PER_TASK,
                    "dataset_root": str(dataset_root.resolve()),
                    "coverage_evidence": f"pinned_archive_sha256:{archive_sha256}",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    _atomic_write(task_manifest, "\n".join(task_rows) + "\n")
    _atomic_write(
        provenance,
        json.dumps(
            {
                "archive": ARCHIVE_NAME,
                "archive_sha256": archive_sha256,
                "dataset_format": "LeRobot v2.1",
                "dataset_root": str(dataset_root.resolve()),
                "episode_count": summary.episode_count,
                "episodes_per_task": EXPECTED_EPISODES_PER_TASK,
                "source": "TianxingChen/RoboTwin2.0::lerobot_dataset/RoboTwin_lerobot_v21.zip",
                "task_count": summary.task_count,
                "dataset_task_catalog": [
                    {
                        "task_index": index,
                        "task": task,
                        "observed_clean_episodes": summary.task_episode_counts[task],
                    }
                    for index, task in sorted(summary.tasks_by_index.items())
                ],
                "training_scope": "competition-authorized clean trajectories only",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return training_list, task_manifest, provenance


def extract_and_prepare(
    archive: Path,
    output_root: Path,
    task_list: Path,
    expected_sha256: str = ARCHIVE_SHA256,
) -> tuple[Path, DatasetSummary]:
    canonical_tasks = load_canonical_tasks(task_list)
    members, actual_sha256 = inspect_archive(archive, expected_sha256=expected_sha256)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve()
    destination = output_root / ARCHIVE_ROOT
    if destination.exists():
        raise PreparationError(
            f"Destination already exists: {destination}. Validate it or move it aside explicitly."
        )
    staging = Path(tempfile.mkdtemp(prefix=f".{ARCHIVE_ROOT}.extracting-", dir=output_root))
    try:
        _extract_members(archive, members, staging)
        staged_dataset = staging / ARCHIVE_ROOT
        summary = validate_dataset(staged_dataset)
        staged_dataset.rename(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    write_manifests(output_root, destination, canonical_tasks, summary, actual_sha256)
    return destination, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="verify, safely extract, validate, and write manifests")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument("--task-list", type=Path, required=True)
    extract.add_argument(
        "--expected-sha256",
        default=ARCHIVE_SHA256,
        help="reviewed official object digest (default: pinned current archive)",
    )

    validate = subparsers.add_parser("validate", help="revalidate an already prepared dataset")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--task-list", type=Path, required=True)
    validate.add_argument(
        "--archive-sha256",
        default=ARCHIVE_SHA256,
        help="digest recorded in regenerated provenance",
    )
    validate.add_argument(
        "--rewrite-manifests",
        action="store_true",
        help="rewrite loader/audit manifests after successful validation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "extract":
            dataset, summary = extract_and_prepare(
                args.archive, args.output_root, args.task_list, args.expected_sha256
            )
            output_root = args.output_root.resolve()
        else:
            canonical_tasks = load_canonical_tasks(args.task_list)
            dataset = args.dataset.resolve()
            summary = validate_dataset(dataset)
            output_root = dataset.parent
            if args.rewrite_manifests:
                write_manifests(output_root, dataset, canonical_tasks, summary, args.archive_sha256)
    except PreparationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[OK] clean dataset: {dataset}")
    print(f"[OK] coverage: {summary.task_count} tasks, {summary.episode_count} episodes")
    print(f"[OK] loader list: {output_root / 'clean_training_data.txt'} (1 merged dataset row)")
    print(f"[OK] task audit: {output_root / 'clean_training_tasks.jsonl'} (50 rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
