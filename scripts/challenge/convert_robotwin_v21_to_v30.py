#!/usr/bin/env python3
"""Convert the audited clean RoboTwin LeRobot v2.1 dataset to LeRobot v3.0.

The source directory is never converted in place.  A hard-linked copy is made
under the same parent filesystem, the upstream LeRobot converter operates on
that private copy, and the verified result is atomically renamed into place.
Only after publication are the loader list and provenance atomically updated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_robotwin_data as prepare


V21_ROOT = prepare.ARCHIVE_ROOT
V30_ROOT = "RoboTwin_lerobot_v30"
V30_VERSION = "v3.0"
CONVERTER_MODULE = "lerobot.datasets.v30.convert_dataset_v21_to_v30"
SOURCE_PROVENANCE = "clean_training_provenance.json"
TRAINING_LIST = "clean_training_data.txt"


class ConversionError(RuntimeError):
    """Raised when conversion or publication cannot be proven safe."""


def _load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConversionError(f"Expected a JSON object in {path}")
    return value


def _resolved_text(path: Path) -> str:
    return str(path.resolve())


def _metadata_digest(dataset_root: Path) -> str:
    """Fingerprint the v2 metadata most at risk from an in-place converter."""
    digest = hashlib.sha256()
    for relative in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"):
        path = dataset_root / relative
        digest.update(relative.encode("utf-8"))
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise ConversionError(f"Cannot fingerprint v2 metadata {path}: {exc}") from exc
    return digest.hexdigest()


def validate_v21_provenance(
    provenance_path: Path,
    source_root: Path,
    summary: prepare.DatasetSummary,
) -> dict:
    provenance = _load_object(provenance_path)
    if provenance.get("archive") != prepare.ARCHIVE_NAME:
        raise ConversionError("Provenance does not name the pinned RoboTwin v2.1 archive")
    if provenance.get("archive_sha256") != prepare.ARCHIVE_SHA256:
        raise ConversionError("Provenance archive SHA-256 is not the reviewed clean-data digest")
    if provenance.get("source") != (
        "TianxingChen/RoboTwin2.0::lerobot_dataset/RoboTwin_lerobot_v21.zip"
    ):
        raise ConversionError("Provenance source is not the approved RoboTwin v2.1 object")
    if provenance.get("training_scope") != "competition-authorized clean trajectories only":
        raise ConversionError("Provenance does not enforce the clean-only training scope")
    if provenance.get("episode_count") != summary.episode_count:
        raise ConversionError("Provenance episode count disagrees with the validated v2.1 source")
    if provenance.get("instruction_catalog_count") != summary.instruction_count:
        raise ConversionError("Provenance task count disagrees with the validated v2.1 source")

    # On the first run dataset_root describes v2.1.  On repeat runs the main
    # provenance describes v3.0 and retains the source root explicitly.
    recorded_source = provenance.get("source_dataset_root", provenance.get("dataset_root"))
    recorded_format = provenance.get("source_dataset_format", provenance.get("dataset_format"))
    if recorded_source != _resolved_text(source_root):
        raise ConversionError("Provenance v2.1 source path does not match the validated source")
    if recorded_format != "LeRobot v2.1":
        raise ConversionError("Provenance does not identify a LeRobot v2.1 source")
    return provenance


def validate_v30_dataset(
    dataset_root: Path,
    *,
    expected_episodes: int = prepare.EXPECTED_EPISODES,
    expected_tasks: int = prepare.EXPECTED_INSTRUCTIONS,
    require_published_name: bool = True,
) -> dict:
    if not dataset_root.is_dir():
        raise ConversionError(f"LeRobot v3.0 dataset directory is missing: {dataset_root}")
    if require_published_name and dataset_root.name != V30_ROOT:
        raise ConversionError(f"Expected published dataset directory {V30_ROOT}: {dataset_root}")
    for path in dataset_root.rglob("*"):
        if path.is_symlink():
            raise ConversionError(f"Symlink is forbidden in converted training data: {path}")
        if prepare.FORBIDDEN_COMPONENT_RE.search(path.name):
            raise ConversionError(f"Randomized data is forbidden in converted training data: {path}")

    info = _load_object(dataset_root / "meta" / "info.json")
    if info.get("codebase_version") != V30_VERSION:
        raise ConversionError(
            f"Converted dataset codebase_version is {info.get('codebase_version')!r}; "
            f"expected {V30_VERSION!r}"
        )
    if info.get("total_episodes") != expected_episodes:
        raise ConversionError(
            f"Converted dataset reports {info.get('total_episodes')!r} episodes; "
            f"expected {expected_episodes}"
        )
    if info.get("total_tasks") != expected_tasks:
        raise ConversionError(
            f"Converted dataset reports {info.get('total_tasks')!r} tasks; "
            f"expected {expected_tasks}"
        )
    if not (dataset_root / "meta" / "tasks.parquet").is_file():
        raise ConversionError("Converted dataset is missing meta/tasks.parquet")
    if not any((dataset_root / "meta" / "episodes").glob("**/*.parquet")):
        raise ConversionError("Converted dataset is missing v3 episode metadata parquet files")
    if not any((dataset_root / "data").glob("**/*.parquet")):
        raise ConversionError("Converted dataset is missing v3 data parquet files")
    if not any((dataset_root / "videos").glob("**/*.mp4")):
        raise ConversionError("Converted dataset is missing v3 video files")
    return info


def _hardlink_copy(source: Path, destination: Path) -> None:
    try:
        shutil.copytree(source, destination, copy_function=os.link)
    except OSError as exc:
        raise ConversionError(
            "Could not create the same-filesystem hard-link staging copy; "
            f"source={source}, staging={destination}: {exc}"
        ) from exc


def _run_converter(staging_parent: Path) -> None:
    command = [
        sys.executable,
        "-m",
        CONVERTER_MODULE,
        f"--repo-id={V21_ROOT}",
        f"--root={staging_parent}",
        "--push-to-hub=false",
        "--force-conversion",
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConversionError(f"LeRobot v2.1 to v3.0 converter failed: {exc}") from exc


def _atomic_write(path: Path, text: str) -> None:
    prepare._atomic_write(path, text)


def _updated_provenance(
    original: dict,
    source_root: Path,
    destination: Path,
    source_metadata_sha256: str,
) -> dict:
    result = dict(original)
    result.update(
        {
            "dataset_format": "LeRobot v3.0",
            "dataset_root": _resolved_text(destination),
            "source_dataset_format": "LeRobot v2.1",
            "source_dataset_root": _resolved_text(source_root),
            "source_metadata_sha256": source_metadata_sha256,
            "conversion": {
                "converter_module": CONVERTER_MODULE,
                "copy_mode": "same-filesystem hard links",
                "force_conversion": True,
                "push_to_hub": False,
                "target_codebase_version": V30_VERSION,
                "lerobot_version": _lerobot_version(),
            },
        }
    )
    return result


def _lerobot_version() -> str:
    try:
        return importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _safe_remove_own_staging(staging: Path, output_root: Path) -> None:
    try:
        is_owned = (
            staging.parent.resolve() == output_root.resolve()
            and staging.name.startswith(f".{V30_ROOT}.converting-")
        )
    except OSError:
        is_owned = False
    if not is_owned:
        raise ConversionError(f"Refuse to clean an unowned staging path: {staging}")
    shutil.rmtree(staging, ignore_errors=True)


def convert_and_publish(
    source_root: Path,
    output_root: Path,
    *,
    runner: Callable[[Path], None] = _run_converter,
) -> tuple[Path, bool]:
    source_root = source_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve()
    if source_root.parent != output_root or source_root.name != V21_ROOT:
        raise ConversionError(
            f"Expected {V21_ROOT} directly under conversion output root {output_root}"
        )

    try:
        source_summary = prepare.validate_dataset(source_root)
    except prepare.PreparationError as exc:
        raise ConversionError(f"LeRobot v2.1 source validation failed: {exc}") from exc
    provenance_path = output_root / SOURCE_PROVENANCE
    source_provenance = validate_v21_provenance(
        provenance_path, source_root, source_summary
    )
    source_metadata_sha256 = _metadata_digest(source_root)

    destination = output_root / V30_ROOT
    converted = False
    if destination.exists():
        validate_v30_dataset(destination)
    else:
        staging = Path(tempfile.mkdtemp(prefix=f".{V30_ROOT}.converting-", dir=output_root))
        try:
            staged_dataset = staging / V21_ROOT
            _hardlink_copy(source_root, staged_dataset)
            runner(staging)
            validate_v30_dataset(staged_dataset, require_published_name=False)

            # Prove the converter did not modify hard-linked source metadata.
            try:
                prepare.validate_dataset(source_root)
            except prepare.PreparationError as exc:
                raise ConversionError(
                    f"Converter changed or invalidated the preserved v2.1 source: {exc}"
                ) from exc
            if _metadata_digest(source_root) != source_metadata_sha256:
                raise ConversionError("Converter modified the preserved v2.1 source metadata")

            try:
                staged_dataset.rename(destination)
            except OSError as exc:
                # A concurrent successful invocation may have won publication.
                if destination.is_dir():
                    validate_v30_dataset(destination)
                else:
                    raise ConversionError(f"Could not atomically publish v3.0 dataset: {exc}") from exc
            converted = True
        finally:
            _safe_remove_own_staging(staging, output_root)

    validate_v30_dataset(destination)
    updated = _updated_provenance(
        source_provenance, source_root, destination, source_metadata_sha256
    )
    # Provenance is committed before the list.  The list is the final switch
    # consumed by training, so a crash cannot point training at an unproven tree.
    _atomic_write(
        provenance_path,
        json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(output_root / TRAINING_LIST, f"robotwin {_resolved_text(destination)}\n")
    return destination, converted


def validate_prepared(source_root: Path, output_root: Path) -> Path:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    try:
        source_summary = prepare.validate_dataset(source_root)
    except prepare.PreparationError as exc:
        raise ConversionError(f"LeRobot v2.1 source validation failed: {exc}") from exc
    provenance = validate_v21_provenance(
        output_root / SOURCE_PROVENANCE, source_root, source_summary
    )
    destination = output_root / V30_ROOT
    validate_v30_dataset(destination)
    if provenance.get("dataset_format") != "LeRobot v3.0":
        raise ConversionError("Current provenance does not identify a LeRobot v3.0 training dataset")
    if provenance.get("dataset_root") != _resolved_text(destination):
        raise ConversionError("Current provenance does not point to the published v3.0 dataset")
    conversion = provenance.get("conversion")
    if not isinstance(conversion, dict) or conversion.get("push_to_hub") is not False:
        raise ConversionError("Current provenance does not prove an offline local conversion")
    if provenance.get("source_metadata_sha256") != _metadata_digest(source_root):
        raise ConversionError("Preserved v2.1 source metadata no longer matches conversion provenance")
    expected = f"robotwin {_resolved_text(destination)}"
    try:
        rows = [line.strip() for line in (output_root / TRAINING_LIST).read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise ConversionError(f"Cannot read clean training list: {exc}") from exc
    if rows != [expected]:
        raise ConversionError(
            f"Clean training list must contain exactly one v3.0 dataset row: {expected}"
        )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("convert", "convert if needed, validate, publish, and update manifests"),
        ("validate", "validate the preserved v2.1 source and published v3.0 dataset"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--source", type=Path, required=True)
        subparser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "convert":
            destination, converted = convert_and_publish(args.source, args.output_root)
            action = "converted and published" if converted else "already valid; manifests refreshed"
        else:
            destination = validate_prepared(args.source, args.output_root)
            action = "validated"
    except (ConversionError, prepare.PreparationError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[OK] LeRobot v3.0 dataset {action}: {destination}")
    print(f"[OK] loader list: {args.output_root.resolve() / TRAINING_LIST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
