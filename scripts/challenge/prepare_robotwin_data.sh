#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

challenge_load_env
REPO_ROOT="$(challenge_repo_root)"
challenge_require_shared_path "${CHALLENGE_DATASET_ARCHIVE}" 'RoboTwin v2.1 archive'
challenge_require_shared_path "${CHALLENGE_TRAIN_DATA}" 'training data root'

python "${SCRIPT_DIR}/prepare_robotwin_data.py" extract \
  --archive "${CHALLENGE_DATASET_ARCHIVE}" \
  --output-root "${CHALLENGE_TRAIN_DATA}" \
  --task-list "${REPO_ROOT}/competition/tasks.txt"

challenge_ok 'Prepared only the pinned clean RoboTwin v2.1 dataset.'
