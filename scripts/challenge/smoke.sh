#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

PHASE="host"
SKIP_PREFLIGHT=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/challenge/smoke.sh [--phase host|train|infer] [--skip-preflight]

Runs preflight and then a small BF16 CUDA computation plus repository imports.
It never downloads weights and never reads competition evaluation data.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="${2:?--phase requires a value}"; shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) challenge_die "Unknown argument: $1" ;;
  esac
done
[[ "${PHASE}" =~ ^(host|train|infer)$ ]] || challenge_die '--phase must be host, train, or infer.'

challenge_load_env
if (( SKIP_PREFLIGHT == 0 )); then
  bash "${SCRIPT_DIR}/preflight.sh" --phase "${PHASE}"
fi

python - <<'PY'
import torch
import transformers
import lingbotvla

torch.manual_seed(7)
device = torch.device("cuda:0")
a = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
b = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
if c.shape != (1024, 1024) or not torch.isfinite(c.float()).all():
    raise SystemExit("BF16 CUDA smoke computation failed")
allocated = torch.cuda.max_memory_allocated() / 1024**2
print(f"CUDA smoke passed on {torch.cuda.get_device_name(0)}; peak allocated {allocated:.1f} MiB")
print(f"transformers {transformers.__version__}; lingbotvla import OK")
PY

challenge_ok "${PHASE} smoke test completed without loading or downloading model weights."

