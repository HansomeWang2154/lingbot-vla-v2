#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

PHASE="host"
MIN_FREE_GB="${CHALLENGE_MIN_FREE_GB:-50}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/challenge/preflight.sh [--phase host|train|infer] [--min-free-gb N]

host  checks the shared disk and exactly one 24 GB RTX 4090-class GPU.
train additionally checks Python packages, private training data, and base weights.
infer checks packages and an exported Hugging Face checkpoint in CHALLENGE_INFER_MODEL_DIR.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="${2:?--phase requires a value}"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="${2:?--min-free-gb requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) challenge_die "Unknown argument: $1" ;;
  esac
done
[[ "${PHASE}" =~ ^(host|train|infer)$ ]] || challenge_die '--phase must be host, train, or infer.'
[[ "${MIN_FREE_GB}" =~ ^[0-9]+$ ]] || challenge_die '--min-free-gb must be an integer.'

challenge_load_env
REPO_ROOT="$(challenge_repo_root)"
challenge_require_shared_path "${CHALLENGE_SHARED_ROOT}" 'shared root'
[[ -w "${CHALLENGE_SHARED_ROOT}" ]] || challenge_die 'Shared root is not writable.'

cache_vars=(HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE HF_ASSETS_CACHE HF_XET_CACHE TRANSFORMERS_CACHE TORCH_HOME PIP_CACHE_DIR XDG_CACHE_HOME PYTHONPYCACHEPREFIX CONDA_PKGS_DIRS CONDA_ENVS_PATH TMPDIR WANDB_DIR CHALLENGE_MODEL_ROOT CHALLENGE_OUTPUT_ROOT)
for var_name in "${cache_vars[@]}"; do
  value="${!var_name:-}"
  [[ -n "${value}" ]] || challenge_die "${var_name} is unset."
  [[ -e "${value}" ]] || challenge_die "${var_name} path does not exist: ${value}"
  challenge_path_is_within "${value}" "${CHALLENGE_SHARED_ROOT}" || challenge_die "${var_name} is outside the shared disk."
done
challenge_ok 'All caches, models, outputs, and temporary files resolve under the shared disk.'

free_kb="$(df -Pk "${CHALLENGE_SHARED_ROOT}" | awk 'NR==2 {print $4}')"
(( free_kb >= MIN_FREE_GB * 1024 * 1024 )) || challenge_die "Less than ${MIN_FREE_GB} GiB remains on the shared disk."
challenge_ok "Shared disk has at least ${MIN_FREE_GB} GiB free."

command -v nvidia-smi >/dev/null 2>&1 || challenge_die 'nvidia-smi is unavailable.'
mapfile -t gpu_rows < <(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
[[ "${#gpu_rows[@]}" -eq 1 ]] || challenge_die "Expected exactly one visible GPU, found ${#gpu_rows[@]}."
IFS=',' read -r gpu_name gpu_mem gpu_driver <<< "${gpu_rows[0]}"
gpu_name="${gpu_name# }"; gpu_mem="${gpu_mem// /}"; gpu_driver="${gpu_driver# }"
[[ "${gpu_name}" == *4090* ]] || challenge_die "Expected an RTX 4090/4090D, found: ${gpu_name}"
[[ "${gpu_mem}" =~ ^[0-9]+$ ]] || challenge_die "Could not parse GPU memory: ${gpu_mem}"
(( gpu_mem >= 23000 )) || challenge_die "GPU memory is ${gpu_mem} MiB; at least 23000 MiB is required."
challenge_ok "GPU: ${gpu_name}, ${gpu_mem} MiB, driver ${gpu_driver}."

mode="$(stat -c '%a' "${REPO_ROOT}/.challenge.env")"
[[ "${mode}" == 600 ]] || challenge_warn ".challenge.env permissions are ${mode}; run chmod 600 .challenge.env."
if git -C "${REPO_ROOT}" ls-files --error-unmatch .challenge.env >/dev/null 2>&1; then
  challenge_die '.challenge.env is tracked by Git.'
fi
if git -C "${REPO_ROOT}" grep -IlE '(-----BEGIN (OPENSSH|RSA|EC|DSA) PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})' -- . >/dev/null 2>&1; then
  challenge_die 'A tracked file appears to contain a private key or access token.'
fi
challenge_ok 'Basic repository secret scan passed.'

[[ "${PHASE}" != host ]] || exit 0

command -v python >/dev/null 2>&1 || challenge_die 'python is unavailable; activate the challenge conda environment.'
python - <<'PY'
import importlib
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, got {platform.python_version()}")

import torch
if torch.__version__.split("+", 1)[0] != "2.8.0":
    raise SystemExit(f"expected torch 2.8.0, got {torch.__version__}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one CUDA device, got {torch.cuda.device_count()}")
major, minor = torch.cuda.get_device_capability(0)
if (major, minor) < (8, 9):
    raise SystemExit(f"expected compute capability >= 8.9, got {major}.{minor}")
for module in ("transformers", "flash_attn", "lingbotvla"):
    importlib.import_module(module)
print(f"Python {platform.python_version()}, torch {torch.__version__}, CUDA {torch.version.cuda}, capability {major}.{minor}")
PY
challenge_ok 'Pinned Python/CUDA imports passed.'

if [[ "${PHASE}" == train ]]; then
  challenge_require_shared_path "${CHALLENGE_TRAIN_DATA}" 'training data'
  challenge_require_shared_path "${CHALLENGE_LOCAL_VAL_DATA}" 'local validation data'
  challenge_require_shared_path "${CHALLENGE_VLA_MODEL_DIR}" 'LingBot-VLA base model'
  challenge_require_shared_path "${CHALLENGE_QWEN3_DIR}" 'Qwen3-VL model'
  [[ -f "${CHALLENGE_VLA_MODEL_DIR}/config.json" ]] || challenge_die 'LingBot-VLA config.json is missing.'
  [[ -f "${CHALLENGE_QWEN3_DIR}/config.json" ]] || challenge_die 'Qwen3-VL config.json is missing.'
  [[ -f "${CHALLENGE_MOGE_CHECKPOINT}" ]] || challenge_die "MoGe checkpoint is missing: ${CHALLENGE_MOGE_CHECKPOINT}"
  [[ ! -e "${CHALLENGE_TRAIN_DATA}/.competition_eval" ]] || challenge_die 'Training data contains the forbidden .competition_eval marker.'
  [[ "$(challenge_realpath "${CHALLENGE_TRAIN_DATA}")" != "$(challenge_realpath "${CHALLENGE_LOCAL_VAL_DATA}")" ]] || challenge_die 'Training and local-validation data resolve to the same directory.'
  if find "${CHALLENGE_TRAIN_DATA}" -iname '*randomized*' -print -quit | grep -q .; then
    challenge_die 'Randomized data found under the training root; the competition permits only the 50 clean tasks for training.'
  fi
  if [[ -n "${COMPETITION_EVAL_DATA:-}" ]]; then
    challenge_die 'COMPETITION_EVAL_DATA must not be mounted or exposed to a training process.'
  fi
  challenge_ok 'Training inputs are present and the competition-evaluation boundary is intact.'
else
  [[ -n "${CHALLENGE_INFER_MODEL_DIR:-}" ]] || challenge_die 'Set CHALLENGE_INFER_MODEL_DIR to an exported hf_ckpt directory.'
  challenge_require_shared_path "${CHALLENGE_INFER_MODEL_DIR}" 'inference checkpoint'
  [[ -f "${CHALLENGE_INFER_MODEL_DIR}/config.json" ]] || challenge_die 'Inference checkpoint config.json is missing.'
  challenge_ok 'Inference checkpoint layout passed.'
fi
