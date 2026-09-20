#!/usr/bin/env python3
"""Build a deterministic submission ZIP from one co-trained checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

try:
    from .validate_results import validate_path
except ImportError:  # Direct invocation: python competition/package_submission.py
    from validate_results import validate_path

HERE = Path(__file__).resolve().parent
DEFAULT_LAYOUT = HERE / "submission_layout.json"
PLACEHOLDER_TOKENS = ("REPLACE_ME", "TODO", "TBD")


def _load_layout(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"archive_root", "results", "checkpoint", "reproduction_report"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"layout missing keys: {', '.join(sorted(missing))}")
    layout = {key: str(data[key]) for key in required}
    for key, value in layout.items():
        posix = PurePosixPath(value)
        if not value or posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"unsafe layout path for {key}: {value!r}")
    return layout


def _checkpoint_files(checkpoint: Path) -> list[Path]:
    files: list[Path] = []
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"checkpoint contains a symlink, materialize it first: {path}")
        if path.is_file():
            files.append(path)
    if not files:
        raise ValueError("checkpoint directory is empty")
    return sorted(files, key=lambda item: item.relative_to(checkpoint).as_posix())


def build_entries(
    checkpoint: Path,
    results: Path,
    report: Path,
    layout_path: Path,
) -> list[tuple[Path, str]]:
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint must be one directory: {checkpoint}")
    if not results.is_file():
        raise ValueError(f"results file not found: {results}")
    if not report.is_file():
        raise ValueError(f"reproduction report not found: {report}")

    report_text = report.read_text(encoding="utf-8")
    found = [token for token in PLACEHOLDER_TOKENS if token in report_text]
    if found:
        raise ValueError(f"reproduction report still contains placeholder token(s): {', '.join(found)}")

    layout = _load_layout(layout_path)
    root = PurePosixPath(layout["archive_root"])
    entries = [
        (results, str(root / PurePosixPath(layout["results"]))),
        (report, str(root / PurePosixPath(layout["reproduction_report"]))),
    ]
    checkpoint_root = root / PurePosixPath(layout["checkpoint"])
    entries.extend(
        (path, str(checkpoint_root / PurePosixPath(path.relative_to(checkpoint).as_posix())))
        for path in _checkpoint_files(checkpoint)
    )
    archive_names = [name for _, name in entries]
    if len(archive_names) != len(set(archive_names)):
        raise ValueError("layout creates duplicate archive paths")
    return sorted(entries, key=lambda item: item[1])


def write_zip(entries: list[tuple[Path, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for source, arcname in entries:
                info = zipfile.ZipInfo(arcname, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 & 0xFFFF) << 16
                with source.open("rb") as source_stream, archive.open(info, "w") as zip_stream:
                    shutil.copyfileobj(source_stream, zip_stream, length=1024 * 1024)
        os.replace(temp, output)
    finally:
        if temp.exists():
            temp.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="the one 50-task co-trained checkpoint directory")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--dry-run", action="store_true", help="validate and print contents without writing a ZIP")
    args = parser.parse_args(argv)

    errors, summary = validate_path(args.results)
    if errors:
        print("Refusing to package invalid results:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    try:
        entries = build_entries(args.checkpoint, args.results, args.report, args.layout)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Cannot package submission: {exc}", file=sys.stderr)
        return 2

    print(f"Results valid: {summary['splits']['clean']['total_attempts']} clean + "
          f"{summary['splits']['randomized']['total_attempts']} randomized attempts")
    print(f"Archive entries: {len(entries)}")
    for _, arcname in entries[:20]:
        print(f"  {arcname}")
    if len(entries) > 20:
        print(f"  ... and {len(entries) - 20} more")
    if args.dry_run:
        print("Dry run complete; no archive written.")
        return 0

    if args.output.exists():
        print(f"Refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    write_zip(entries, args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
