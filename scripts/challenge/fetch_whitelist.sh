#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

TARGET="${1:-}"
if [[ "${TARGET}" != dataset && "${TARGET}" != models && "${TARGET}" != all ]]; then
  printf 'Usage: bash scripts/challenge/fetch_whitelist.sh dataset|models|all\n' >&2
  exit 2
fi

challenge_load_env
command -v hf >/dev/null 2>&1 || challenge_die "The 'hf' CLI is unavailable; activate the challenge environment."

# Keep this allowlist literal. In particular, never replace the single dataset
# filename with a repository snapshot: that repository is approximately 1.53 TB.
readonly DATASET_REPO='TianxingChen/RoboTwin2.0'
readonly DATASET_FILE='lerobot_dataset/RoboTwin_lerobot_v21.zip'
# Pin the reviewed RGB-fixed clean export. prepare_robotwin_data.py additionally
# verifies the downloaded object's SHA-256 before extracting a single byte.
readonly DATASET_REVISION='981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042'
readonly VLA_REPO='robbyant/lingbot-vla-v2-6b-robotwin'
readonly QWEN_REPO='Qwen/Qwen3-VL-4B-Instruct'
readonly MOGE_REPO='Ruicheng/moge-2-vitb-normal'

if [[ "${TARGET}" == dataset || "${TARGET}" == all ]]; then
  download_dir="${CHALLENGE_SHARED_ROOT}/downloads/robotwin_v21"
  mkdir -p "${download_dir}"
  hf download "${DATASET_REPO}" "${DATASET_FILE}" \
    --repo-type dataset --revision "${DATASET_REVISION}" --local-dir "${download_dir}"
  archive="${download_dir}/${DATASET_FILE}"
  [[ -f "${archive}" ]] || challenge_die "Dataset archive was not downloaded: ${archive}"
  [[ "$(challenge_realpath "${archive}")" == "$(challenge_realpath "${CHALLENGE_DATASET_ARCHIVE}")" ]] || \
    challenge_die 'Downloaded archive path does not match the bootstrap allowlist.'
  challenge_ok "Downloaded the one allowed dataset file at reviewed revision ${DATASET_REVISION} to ${archive}"
fi

if [[ "${TARGET}" == models || "${TARGET}" == all ]]; then
  hf download "${VLA_REPO}" --local-dir "${CHALLENGE_VLA_REPO_DIR}"
  hf download "${QWEN_REPO}" --local-dir "${CHALLENGE_QWEN3_DIR}"
  # This repository only needs the 419 MB checkpoint; README metadata is not
  # required by training, so use a positional filename instead of a snapshot.
  hf download "${MOGE_REPO}" model.pt --local-dir "${CHALLENGE_MOGE_DIR}"
  challenge_ok 'Downloaded only allowlisted model repositories/files.'
fi
