#!/usr/bin/env python3
"""Validate the local competition results contract before packaging.

This is intentionally dependency-free.  The contest's authenticated download may
use a different JSON shape; adapt ``validate_document`` once that template is
available, while retaining the invariants checked here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_TASKS = HERE / "tasks.txt"
EXPECTED_SPLITS = ("clean", "randomized")


def load_tasks(path: Path = DEFAULT_TASKS) -> list[str]:
    tasks = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(tasks) != 50:
        raise ValueError(f"task manifest must contain exactly 50 tasks, got {len(tasks)}")
    if len(tasks) != len(set(tasks)):
        raise ValueError("task manifest contains duplicate names")
    return tasks


def _is_placeholder(value: Any) -> bool:
    return value is None or (isinstance(value, str) and (not value or "REPLACE" in value))


def validate_document(
    document: Any,
    *,
    tasks: list[str] | None = None,
    allow_placeholders: bool = False,
    expected_attempts: int = 100,
) -> tuple[list[str], dict[str, Any]]:
    """Return ``(errors, summary)`` for a parsed result document."""
    errors: list[str] = []
    summary: dict[str, Any] = {"splits": {}}
    tasks = tasks or load_tasks()
    expected = set(tasks)

    if not isinstance(document, dict):
        return ["root must be a JSON object"], summary

    if document.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    submission = document.get("submission")
    if not isinstance(submission, dict):
        errors.append("submission must be an object")
    else:
        for key in ("team_name", "model_name", "checkpoint_id"):
            if _is_placeholder(submission.get(key)) and not allow_placeholders:
                errors.append(f"submission.{key} is still a placeholder")

    training = document.get("training")
    if not isinstance(training, dict):
        errors.append("training must be an object")
    else:
        if training.get("strategy") != "cotrain_50_tasks":
            errors.append("training.strategy must be 'cotrain_50_tasks'")
        if training.get("single_checkpoint") is not True:
            errors.append("training.single_checkpoint must be true")
        if training.get("train_splits") != ["clean"]:
            errors.append("training.train_splits must be exactly ['clean']")
        if training.get("randomized_training_data_used") is not False:
            errors.append("randomized data is evaluation-only; randomized_training_data_used must be false")

    evaluation = document.get("evaluation")
    if not isinstance(evaluation, dict):
        errors.append("evaluation must be an object")
        return errors, summary
    unexpected_splits = sorted(set(evaluation) - set(EXPECTED_SPLITS))
    missing_splits = sorted(set(EXPECTED_SPLITS) - set(evaluation))
    if missing_splits:
        errors.append(f"missing evaluation splits: {', '.join(missing_splits)}")
    if unexpected_splits:
        errors.append(f"unexpected evaluation splits: {', '.join(unexpected_splits)}")

    for split in EXPECTED_SPLITS:
        split_data = evaluation.get(split)
        if not isinstance(split_data, dict):
            errors.append(f"evaluation.{split} must be an object")
            continue
        task_results = split_data.get("tasks")
        if not isinstance(task_results, dict):
            errors.append(f"evaluation.{split}.tasks must be an object")
            continue
        actual = set(task_results)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing:
            errors.append(f"evaluation.{split}.tasks missing {len(missing)} task(s): {', '.join(missing)}")
        if unexpected:
            errors.append(
                f"evaluation.{split}.tasks has {len(unexpected)} unknown task(s): {', '.join(unexpected)}"
            )

        total_successes = 0
        completed = 0
        for task in tasks:
            result = task_results.get(task)
            if not isinstance(result, dict):
                errors.append(f"evaluation.{split}.tasks.{task} must be an object")
                continue
            attempts = result.get("attempts")
            if type(attempts) is not int or attempts != expected_attempts:
                errors.append(
                    f"evaluation.{split}.tasks.{task}.attempts must be integer {expected_attempts}"
                )
            value = result.get("successes")
            if _is_placeholder(value):
                if not allow_placeholders:
                    errors.append(f"evaluation.{split}.tasks.{task}.successes is not filled")
                continue
            if type(value) is not int:
                errors.append(f"evaluation.{split}.tasks.{task}.successes must be an integer")
            elif type(attempts) is int and not 0 <= value <= attempts:
                errors.append(
                    f"evaluation.{split}.tasks.{task}.successes must be between 0 and {attempts}"
                )
            else:
                total_successes += value
                completed += 1
        total_attempts = expected_attempts * len(tasks)
        summary["splits"][split] = {
            "completed_tasks": completed,
            "task_count": len(tasks),
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "success_rate": total_successes / total_attempts if completed == len(tasks) else None,
        }

    rates = [summary["splits"].get(name, {}).get("success_rate") for name in EXPECTED_SPLITS]
    if all(isinstance(rate, float) and math.isfinite(rate) for rate in rates):
        summary["macro_split_success_rate"] = sum(rates) / len(rates)
    else:
        summary["macro_split_success_rate"] = None
    return errors, summary


def validate_path(
    path: Path, *, allow_placeholders: bool = False, expected_attempts: int = 100
) -> tuple[list[str], dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"file not found: {path}"], {"splits": {}}
    except json.JSONDecodeError as exc:
        return [f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"], {"splits": {}}
    return validate_document(
        document,
        allow_placeholders=allow_placeholders,
        expected_attempts=expected_attempts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results.json to validate")
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="schema-check the distributed template (never use for a final package)",
    )
    parser.add_argument("--expected-attempts", type=int, default=100)
    args = parser.parse_args(argv)
    if args.expected_attempts <= 0:
        parser.error("--expected-attempts must be positive")

    errors, summary = validate_path(
        args.results,
        allow_placeholders=args.allow_placeholders,
        expected_attempts=args.expected_attempts,
    )
    if errors:
        print("INVALID", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("VALID")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
