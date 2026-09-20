#!/usr/bin/env bash

# Shared helpers for the challenge host scripts. This file is meant to be sourced.

challenge_die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

challenge_warn() {
  printf '[WARN] %s\n' "$*" >&2
}

challenge_ok() {
  printf '[OK] %s\n' "$*"
}

challenge_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/../.." && pwd
}

challenge_load_env() {
  local repo_root env_file
  repo_root="$(challenge_repo_root)"
  env_file="${CHALLENGE_ENV_FILE:-${repo_root}/.challenge.env}"
  [[ -f "${env_file}" ]] || challenge_die "Runtime file not found: ${env_file}. Run scripts/challenge/bootstrap.sh first."
  # The file is generated locally by bootstrap.sh and is gitignored.
  # shellcheck disable=SC1090
  source "${env_file}"
}

challenge_realpath() {
  readlink -f -- "$1"
}

challenge_path_is_within() {
  local child parent
  child="$(challenge_realpath "$1")" || return 1
  parent="$(challenge_realpath "$2")" || return 1
  [[ "${child}" == "${parent}" || "${child}" == "${parent}/"* ]]
}

challenge_require_shared_path() {
  local path="$1" label="$2"
  [[ -n "${CHALLENGE_SHARED_ROOT:-}" ]] || challenge_die 'CHALLENGE_SHARED_ROOT is unset.'
  [[ -e "${path}" ]] || challenge_die "${label} does not exist: ${path}"
  challenge_path_is_within "${path}" "${CHALLENGE_SHARED_ROOT}" || \
    challenge_die "${label} escapes the shared disk: ${path}"
}

