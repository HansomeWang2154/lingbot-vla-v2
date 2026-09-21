#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

challenge_load_env
REPO_ROOT="$(challenge_repo_root)"
challenge_require_shared_path "${CHALLENGE_DATASET_ARCHIVE}" 'RoboTwin v2.1 archive'
challenge_require_shared_path "${CHALLENGE_TRAIN_DATA}" 'training data root'

if [[ -d "${CHALLENGE_TRAIN_DATA}/RoboTwin_lerobot_v21" ]]; then
  python "${SCRIPT_DIR}/prepare_robotwin_data.py" validate \
    --dataset "${CHALLENGE_TRAIN_DATA}/RoboTwin_lerobot_v21" \
    --task-list "${REPO_ROOT}/competition/tasks.txt" \
    --rewrite-manifests
else
  python "${SCRIPT_DIR}/prepare_robotwin_data.py" extract \
    --archive "${CHALLENGE_DATASET_ARCHIVE}" \
    --output-root "${CHALLENGE_TRAIN_DATA}" \
    --task-list "${REPO_ROOT}/competition/tasks.txt"
fi

python "${SCRIPT_DIR}/convert_robotwin_v21_to_v30.py" convert \
  --source "${CHALLENGE_TRAIN_DATA}/RoboTwin_lerobot_v21" \
  --output-root "${CHALLENGE_TRAIN_DATA}"

challenge_ok 'Preserved the pinned clean RoboTwin v2.1 source and prepared its local LeRobot v3.0 training copy.'
