# Competition delivery scaffold

This directory is a pre-submission quality gate for the 50-task RoboTwin track. It does not
change training code or VLA configs.

## Non-negotiable experiment contract

1. Co-train **one model across all 50 tasks** and submit one checkpoint. Per-task checkpoints,
   split-specific checkpoints, routing to separate weights, or checkpoint ensembling violate
   this local contract.
2. Train on **clean data only**. `randomized` is evaluation-only and must not influence
   optimization, hyperparameter tuning, early stopping, or checkpoint selection.
3. Evaluate every task with exactly **100 attempts** on each split: 50 × 100 clean plus
   50 × 100 randomized = 10,000 total attempts.
4. Use the same immutable checkpoint for both splits and preserve raw logs.

The authenticated organizer download is the source of truth. The final official
`results.json` template and exact archive layout were not available in this repository when
this scaffold was created. **Before submitting, download the official template after
registration and adapt `results.template.json`, `validate_results.py`, and
`submission_layout.json` to it.** Do not assume this local schema alone guarantees acceptance.

## Files

- `tasks.txt`: canonical 50-task order copied from the repository's evaluation launcher.
- `results.template.json`: local result template; every task has explicit `attempts` and
  `successes`, and `null` marks an unfilled success count.
- `validate_results.py`: checks all tasks, both splits, exact attempts, integer ranges, and
  the single-checkpoint / clean-only training declarations.
- `submission_layout.json`: the only place where archive paths are defined. Update it to
  match the organizer's official directory tree.
- `package_submission.py`: validates results, rejects report placeholders/symlinks, then
  packages exactly one checkpoint directory.
- `reproduction_report.template.md`: provenance and reproducibility checklist.

## Workflow

Create working copies and fill them from machine-generated evaluator output:

```bash
cp competition/results.template.json work/results.json
cp competition/reproduction_report.template.md work/reproduction_report.md
python competition/validate_results.py work/results.json
```

The strict validator intentionally fails until every `null` and identity placeholder is
replaced. To check only the distributed template structure:

```bash
python competition/validate_results.py competition/results.template.json --allow-placeholders
```

Dry-run the final package first:

```bash
python competition/package_submission.py \
  --checkpoint /absolute/path/to/the_one_50_task_hf_ckpt \
  --results work/results.json \
  --report work/reproduction_report.md \
  --output dist/submission.zip \
  --dry-run
```

Remove `--dry-run` to write the ZIP. Existing output is never overwritten. The default tree is:

```text
submission/
├── checkpoint/                 # exactly one 50-task co-trained weight directory
├── reproduction_report.md
└── results.json
```

Once the official package is available, edit only `submission_layout.json` if names/paths
change. If its JSON field structure differs, update the template and validator together and
add a regression test before packaging.

## Self-check

No third-party packages are required:

```bash
python -m unittest discover -s competition/tests -v
```

Never commit credentials, private dataset URLs, raw secrets, or cloud-container passwords.
