# Reproduction report

> Delete this note and replace every `REPLACE_ME` / `TODO` / `TBD` before packaging.

## Submission identity

- Team: REPLACE_ME
- Model: REPLACE_ME
- Git commit: REPLACE_ME
- Single submitted checkpoint: REPLACE_ME
- Checkpoint SHA-256 / immutable artifact ID: REPLACE_ME

## Compliance statement

- [ ] One checkpoint is used for all 50 tasks and both evaluation splits.
- [ ] The checkpoint was co-trained across all 50 tasks; it is not a collection of task-specific weights.
- [ ] Only the clean split was used for training.
- [ ] The randomized split was used only for final evaluation, never for training, tuning, early stopping, or checkpoint selection.
- [ ] Each of the 50 tasks was evaluated for exactly 100 attempts on clean and 100 attempts on randomized.

## Hardware and software

- GPU model/count: REPLACE_ME
- CPU/RAM: REPLACE_ME
- OS/CUDA/driver: REPLACE_ME
- Python/PyTorch: REPLACE_ME
- LingBot-VLA repository commit: REPLACE_ME
- RoboTwin repository commit: REPLACE_ME
- Container image digest or environment lock file: REPLACE_ME

## Data

Document the official dataset source, download date, checksums, clean-only training paths,
preprocessing command, normalization statistics, exclusions, and any augmentations. State
explicitly how randomized data was isolated from the training and model-selection pipeline.

REPLACE_ME

## Training

Provide the exact single 50-task co-training command, config snapshot, initialization
checkpoint, seed(s), precision, GPU count, global batch size, optimizer, learning-rate
schedule, total steps, checkpoint-selection rule, and wall-clock time.

```bash
REPLACE_ME
```

## Evaluation

Provide the exact commands for `demo_clean` and `demo_randomized`, the inference precision,
the seed policy, and the path to raw logs. Both runs must use the identical submitted
checkpoint and exactly 100 attempts per task.

```bash
REPLACE_ME
```

## Results provenance

Explain how raw evaluator logs were converted into `results.json`. Retain the raw logs and
record their hashes in the experiment archive. Do not manually transcribe final counts.

REPLACE_ME

## Known deviations

List every deviation from the organizer's pinned environment or write `None`.

REPLACE_ME

