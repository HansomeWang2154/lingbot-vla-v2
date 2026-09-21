#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

OVERWRITE=0
if [[ "${1:-}" == "--overwrite" ]]; then
  OVERWRITE=1
  shift
fi
[[ $# -eq 0 ]] || challenge_die 'Usage: bash scripts/challenge/compute_clean_norm_stats.sh [--overwrite]'

challenge_load_env
REPO_ROOT="$(challenge_repo_root)"
challenge_require_shared_path "${CHALLENGE_TRAIN_DATA}" 'training data root'
challenge_require_shared_path "${CHALLENGE_TRAIN_LIST}" 'clean training list'
challenge_require_shared_path "$(dirname "${CHALLENGE_CLEAN_NORM_STATS}")" 'clean normalization output directory'
challenge_path_is_within "${CHALLENGE_CLEAN_NORM_STATS}" "${CHALLENGE_SHARED_ROOT}" || \
  challenge_die "Clean normalization output escapes the shared disk: ${CHALLENGE_CLEAN_NORM_STATS}"
[[ -f "${CHALLENGE_TRAIN_LIST}" ]] || challenge_die "Missing clean training list: ${CHALLENGE_TRAIN_LIST}"

python "${SCRIPT_DIR}/prepare_robotwin_data.py" validate \
  --dataset "${CHALLENGE_TRAIN_DATA}/RoboTwin_lerobot_v21" \
  --task-list "${REPO_ROOT}/competition/tasks.txt"

if [[ -e "${CHALLENGE_CLEAN_NORM_STATS}" && "${OVERWRITE}" != 1 ]]; then
  challenge_die "Normalization file already exists: ${CHALLENGE_CLEAN_NORM_STATS}; pass --overwrite to recompute it."
fi

cd "${REPO_ROOT}"
NPROC_PER_NODE=1 bash train.sh \
  scripts/compute_norm_stats.py \
  configs/vla/norm_compute/post_data.yaml \
  --data.robot_name robotwin \
  --data.train_path "${CHALLENGE_TRAIN_LIST}" \
  --data.norm_path "${CHALLENGE_CLEAN_NORM_STATS}" \
  --data.num_workers 8 \
  --train.micro_batch_size 48

[[ -s "${CHALLENGE_CLEAN_NORM_STATS}" ]] || challenge_die 'Normalization computation did not create a non-empty output file.'
challenge_ok "Clean-only normalization statistics: ${CHALLENGE_CLEAN_NORM_STATS}"
