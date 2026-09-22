#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

readonly ROBOTWIN_REPO_URL='https://github.com/RoboTwin-Platform/RoboTwin.git'
readonly ROBOTWIN_REVISION='13c3c47ff4312dd62484bcd51be034af55c062d1'
readonly ROBOTWIN_SHORT_REVISION="${ROBOTWIN_REVISION:0:7}"

DO_CLONE=0
DO_INSTALL_ENV=0
DO_DOWNLOAD_ASSETS=0
DO_VERIFY=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/challenge/setup_robotwin.sh [actions]

Prepare the pinned RoboTwin simulator in a dedicated checkout and conda-prefix
environment. With no action flag, the script performs read-only verification.

Actions:
  --clone             Clone/fetch and detach at the pinned RoboTwin revision
  --install-env       Create and install the dedicated Python 3.10 sim env
  --download-assets   Run the pinned RoboTwin asset downloader (explicit opt-in)
  --verify            Verify checkout, environment, assets, Vulkan, and CUDA
  -h, --help          Show this help

The paths come from .challenge.env. This script never removes or changes an
existing conda environment outside CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clone) DO_CLONE=1; shift ;;
    --install-env) DO_INSTALL_ENV=1; shift ;;
    --download-assets) DO_DOWNLOAD_ASSETS=1; shift ;;
    --verify) DO_VERIFY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) challenge_die "Unknown argument: $1" ;;
  esac
done

if (( DO_CLONE == 0 && DO_INSTALL_ENV == 0 && DO_DOWNLOAD_ASSETS == 0 && DO_VERIFY == 0 )); then
  DO_VERIFY=1
fi

challenge_load_env
: "${CHALLENGE_ROBOTWIN_ROOT:=${CHALLENGE_SHARED_ROOT}/sim/RoboTwin}"
: "${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX:=${CHALLENGE_SHARED_ROOT}/conda/envs/robotwin-sim-${ROBOTWIN_SHORT_REVISION}}"
if [[ -z "${CHALLENGE_CONDA_SH:-}" ]]; then
  command -v conda >/dev/null 2>&1 || \
    challenge_die 'CHALLENGE_CONDA_SH is unset and conda cannot be auto-detected; rerun bootstrap.sh.'
  CHALLENGE_CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi

challenge_path_is_within "$(dirname "${CHALLENGE_ROBOTWIN_ROOT}")" "${CHALLENGE_SHARED_ROOT}" || \
  challenge_die "RoboTwin parent escapes the shared disk: ${CHALLENGE_ROBOTWIN_ROOT}"
challenge_path_is_within "$(dirname "${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}")" "${CHALLENGE_SHARED_ROOT}" || \
  challenge_die "Simulator env parent escapes the shared disk: ${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}"

origin_is_official() {
  local origin="$1"
  [[ "${origin}" =~ ^https://github\.com/RoboTwin-Platform/RoboTwin(\.git)?$ || \
     "${origin}" =~ ^git@github\.com:RoboTwin-Platform/RoboTwin(\.git)?$ || \
     "${origin}" =~ ^ssh://git@github\.com/RoboTwin-Platform/RoboTwin(\.git)?$ ]]
}

verify_checkout() {
  [[ -d "${CHALLENGE_ROBOTWIN_ROOT}/.git" ]] || \
    challenge_die "RoboTwin checkout is absent: ${CHALLENGE_ROBOTWIN_ROOT}. Run with --clone."
  local origin head tracked_changes
  origin="$(git -C "${CHALLENGE_ROBOTWIN_ROOT}" remote get-url origin)"
  origin_is_official "${origin}" || challenge_die "RoboTwin origin is not the official repository: ${origin}"
  head="$(git -C "${CHALLENGE_ROBOTWIN_ROOT}" rev-parse HEAD)"
  [[ "${head}" == "${ROBOTWIN_REVISION}" ]] || \
    challenge_die "RoboTwin HEAD is ${head}; expected ${ROBOTWIN_REVISION}."
  tracked_changes="$(git -C "${CHALLENGE_ROBOTWIN_ROOT}" status --porcelain --untracked-files=no)"
  [[ -z "${tracked_changes}" ]] || challenge_die 'Pinned RoboTwin tracked files have local modifications.'
  for required in envs assets task_config script/eval_policy.py script/_install.sh script/_download_assets.sh; do
    [[ -e "${CHALLENGE_ROBOTWIN_ROOT}/${required}" ]] || challenge_die "Pinned checkout is missing ${required}."
  done
  challenge_ok "RoboTwin checkout is pinned at ${ROBOTWIN_REVISION}."
}

clone_checkout() {
  command -v git >/dev/null 2>&1 || challenge_die 'git is required.'
  mkdir -p "$(dirname "${CHALLENGE_ROBOTWIN_ROOT}")"
  if [[ ! -e "${CHALLENGE_ROBOTWIN_ROOT}" ]]; then
    # Fetch only the required historical snapshot; the cloud link need not pull
    # the moving default branch and the full RoboTwin history first.
    git init --quiet "${CHALLENGE_ROBOTWIN_ROOT}"
    git -C "${CHALLENGE_ROBOTWIN_ROOT}" remote add origin "${ROBOTWIN_REPO_URL}"
  elif [[ ! -d "${CHALLENGE_ROBOTWIN_ROOT}/.git" ]]; then
    challenge_die "Refusing to reuse a non-Git path: ${CHALLENGE_ROBOTWIN_ROOT}"
  fi

  local origin tracked_changes
  origin="$(git -C "${CHALLENGE_ROBOTWIN_ROOT}" remote get-url origin)"
  origin_is_official "${origin}" || challenge_die "Refusing to modify non-official checkout: ${origin}"
  tracked_changes="$(git -C "${CHALLENGE_ROBOTWIN_ROOT}" status --porcelain --untracked-files=no)"
  [[ -z "${tracked_changes}" ]] || challenge_die 'Refusing checkout because tracked RoboTwin files are modified.'
  if ! git -C "${CHALLENGE_ROBOTWIN_ROOT}" cat-file -e "${ROBOTWIN_REVISION}^{commit}" 2>/dev/null; then
    git -C "${CHALLENGE_ROBOTWIN_ROOT}" fetch --depth 1 origin "${ROBOTWIN_REVISION}"
  fi
  git -C "${CHALLENGE_ROBOTWIN_ROOT}" checkout --detach "${ROBOTWIN_REVISION}"
  verify_checkout
}

require_conda() {
  [[ -f "${CHALLENGE_CONDA_SH}" ]] || challenge_die "conda.sh not found: ${CHALLENGE_CONDA_SH}"
  # shellcheck disable=SC1090
  source "${CHALLENGE_CONDA_SH}"
  command -v conda >/dev/null 2>&1 || challenge_die 'conda is unavailable after sourcing conda.sh.'
}

sim_python() {
  printf '%s/bin/python' "${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}"
}

install_sim_env() {
  verify_checkout
  require_conda
  local marker python_bin pytorch3d_source curobo_dir curobo_origin sapien_dir mplib_dir urdf_loader planner
  marker="${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}/.robotwin-source-revision"
  python_bin="$(sim_python)"

  if [[ ! -x "${python_bin}" ]]; then
    [[ ! -e "${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}" ]] || \
      challenge_die "Refusing to overwrite incomplete/non-conda path: ${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}"
    conda create --prefix "${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}" python=3.10 pip -y
    printf '%s\n' "${ROBOTWIN_REVISION}" > "${marker}"
  else
    [[ -f "${marker}" ]] || challenge_die "Existing env lacks ownership marker: ${marker}"
    [[ "$(<"${marker}")" == "${ROBOTWIN_REVISION}" ]] || challenge_die 'Existing simulator env belongs to another RoboTwin revision.'
  fi

  PYTHONNOUSERSITE=1 "${python_bin}" -m pip install --upgrade pip
  # SAPIEN still imports pkg_resources, which setuptools 81+ no longer ships.
  PYTHONNOUSERSITE=1 "${python_bin}" -m pip install setuptools==69.5.1
  PYTHONNOUSERSITE=1 "${python_bin}" -m pip install -r "${CHALLENGE_ROBOTWIN_ROOT}/script/requirements.txt"
  PYTHONNOUSERSITE=1 "${python_bin}" -m pip install numpy==1.26.4
  # The default matches upstream.  A verified local archive/directory can be
  # supplied on restricted cloud links without weakening the remaining checks.
  pytorch3d_source="${CHALLENGE_PYTORCH3D_SOURCE:-git+https://github.com/facebookresearch/pytorch3d.git@stable}"
  if PYTHONNOUSERSITE=1 "${python_bin}" -c \
      'import pytorch3d; assert pytorch3d.__version__ == "0.7.8"' 2>/dev/null; then
    challenge_ok 'PyTorch3D 0.7.8 is already installed.'
  else
    PYTHONNOUSERSITE=1 "${python_bin}" -m pip install \
      "${pytorch3d_source}" --no-build-isolation
  fi

  # Apply the two source fixes performed by the pinned upstream _install.sh,
  # but validate each target and make the operation safe to repeat.
  sapien_dir="$(PYTHONNOUSERSITE=1 "${python_bin}" -c 'import pathlib, sapien; print(pathlib.Path(sapien.__file__).resolve().parent)')"
  urdf_loader="${sapien_dir}/wrapper/urdf_loader.py"
  [[ -f "${urdf_loader}" ]] || challenge_die "SAPIEN loader not found: ${urdf_loader}"
  sed -i -E 's/open\((urdf_file|srdf_file), "r"\)/open(\1, "r", encoding="utf-8")/g' "${urdf_loader}"
  sed -i -E 's/urdf_file\[:-4\] \+ "srdf"/urdf_file[:-4] + ".srdf"/g' "${urdf_loader}"
  grep -Fq 'open(urdf_file, "r", encoding="utf-8")' "${urdf_loader}" || \
    challenge_die 'The pinned SAPIEN UTF-8 patch was not applied.'
  grep -Fq 'urdf_file[:-4] + ".srdf"' "${urdf_loader}" || \
    challenge_die 'The pinned SAPIEN .srdf patch was not applied.'

  mplib_dir="$(PYTHONNOUSERSITE=1 "${python_bin}" -c 'import pathlib, mplib; print(pathlib.Path(mplib.__file__).resolve().parent)')"
  planner="${mplib_dir}/planner.py"
  [[ -f "${planner}" ]] || challenge_die "mplib planner not found: ${planner}"
  sed -i -E 's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${planner}"
  if grep -Fq 'if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:' "${planner}"; then
    challenge_die 'The pinned mplib collision patch was not applied.'
  fi

  curobo_dir="${CHALLENGE_ROBOTWIN_ROOT}/envs/curobo"
  if [[ ! -e "${curobo_dir}" ]]; then
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "${curobo_dir}"
  elif [[ ! -d "${curobo_dir}/.git" ]]; then
    challenge_die "Refusing to reuse non-Git cuRobo path: ${curobo_dir}"
  fi
  curobo_origin="$(git -C "${curobo_dir}" remote get-url origin)"
  [[ "${curobo_origin}" =~ ^https://github\.com/NVlabs/curobo(\.git)?$ ]] || \
    challenge_die "Unexpected cuRobo origin: ${curobo_origin}"
  git -C "${curobo_dir}" describe --tags --exact-match 2>/dev/null | grep -qx 'v0.7.8' || \
    challenge_die 'Existing cuRobo checkout is not the upstream v0.7.8 tag.'
  PYTHONNOUSERSITE=1 "${python_bin}" -m pip install -e "${curobo_dir}" --no-build-isolation
  PYTHONNOUSERSITE=1 "${python_bin}" -m pip install \
    warp-lang==1.12.0 setuptools==69.5.1 websockets==15.0.1 msgpack==1.1.1
  PYTHONNOUSERSITE=1 "${python_bin}" -m pip check

  PYTHONNOUSERSITE=1 "${python_bin}" -m pip freeze > \
    "${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}/robotwin-requirements.freeze.txt"
  printf '%s\n' "${ROBOTWIN_REVISION}" > "${marker}"
  challenge_ok "Dedicated simulator environment installed: ${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}"
}

download_assets() {
  verify_checkout
  [[ -x "$(sim_python)" ]] || challenge_die 'Install the dedicated simulator env before downloading assets.'
  command -v unzip >/dev/null 2>&1 || challenge_die 'unzip is required by the pinned RoboTwin asset script.'
  (
    cd "${CHALLENGE_ROBOTWIN_ROOT}"
    PATH="${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}/bin:${PATH}" \
      PYTHONNOUSERSITE=1 bash script/_download_assets.sh
  )
  challenge_ok 'RoboTwin simulator assets downloaded and configured.'
}

verify_sim_env() {
  verify_checkout
  local marker python_bin
  marker="${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}/.robotwin-source-revision"
  python_bin="$(sim_python)"
  [[ -x "${python_bin}" ]] || challenge_die "Simulator env is absent: ${CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX}"
  [[ -f "${marker}" && "$(<"${marker}")" == "${ROBOTWIN_REVISION}" ]] || \
    challenge_die 'Simulator env ownership/revision marker is missing or incorrect.'

  PYTHONNOUSERSITE=1 "${python_bin}" - <<'PY'
import importlib
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10, got {sys.version.split()[0]}")
for name in ("torch", "sapien", "mplib", "open3d", "websockets", "msgpack"):
    importlib.import_module(name)
import numpy
import torch
if torch.__version__.split("+", 1)[0] != "2.4.1":
    raise SystemExit(f"expected torch 2.4.1, got {torch.__version__}")
if torch.version.cuda != "12.1":
    raise SystemExit(f"expected the pinned CUDA 12.1 torch build, got {torch.version.cuda}")
if numpy.__version__ != "1.26.4":
    raise SystemExit(f"expected numpy 1.26.4, got {numpy.__version__}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one CUDA GPU, got {torch.cuda.device_count()}")
if "4090" not in torch.cuda.get_device_name(0):
    raise SystemExit(f"expected RTX 4090-class GPU, got {torch.cuda.get_device_name(0)}")
print(f"Python {sys.version.split()[0]}, torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}")
PY

  for asset_dir in assets/background_texture assets/embodiments assets/objects; do
    [[ -d "${CHALLENGE_ROBOTWIN_ROOT}/${asset_dir}" ]] || challenge_die "RoboTwin asset directory is missing: ${asset_dir}"
    find "${CHALLENGE_ROBOTWIN_ROOT}/${asset_dir}" -mindepth 1 -print -quit | grep -q . || \
      challenge_die "RoboTwin asset directory is empty: ${asset_dir}"
  done
  command -v vulkaninfo >/dev/null 2>&1 || challenge_die 'vulkaninfo is missing; install vulkan-tools on the host.'
  vulkaninfo --summary >/dev/null 2>&1 || \
    challenge_die 'Vulkan is unavailable. The container must expose NVIDIA graphics capability.'
  challenge_ok 'Simulator imports, assets, CUDA, and Vulkan passed.'

  cat <<EOF

One-episode BF16 pipeline smoke (not a competition score):
  source .challenge.env
  QWEN3VL_PATH="\$CHALLENGE_QWEN3_DIR" \\
  bash experiment/robotwin/start_robotwin_infer_and_eval.sh \\
    --model_path "\$CHALLENGE_INFER_MODEL_DIR" \\
    --eval_workdir "\$CHALLENGE_ROBOTWIN_ROOT" \\
    --conda_sh "\$CHALLENGE_CONDA_SH" \\
    --inference_env "\$CHALLENGE_ENV_NAME" \\
    --sim_env "\$CHALLENGE_ROBOTWIN_SIM_ENV_PREFIX" \\
    --output_base "\$CHALLENGE_OUTPUT_ROOT/eval" \\
    --task_config demo_clean --num_tasks 1 --episodes 1 \\
    --num_gpus 1 --num_per_gpu 1 --use_bf16 True --use_fp32 False \\
    --use_compile False --no_video

Formal local evaluation must omit --episodes (default: 100).
EOF
}

(( DO_CLONE == 0 )) || clone_checkout
(( DO_INSTALL_ENV == 0 )) || install_sim_env
(( DO_DOWNLOAD_ASSETS == 0 )) || download_assets
(( DO_VERIFY == 0 )) || verify_sim_env
