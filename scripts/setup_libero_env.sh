#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[setup-libero] %s\n' "$*" >&2
}

fail() {
  printf '[setup-libero] ERROR: %s\n' "$*" >&2
  exit 1
}

find_conda() {
  if [ -n "${CONDA_BIN:-}" ]; then
    [ -x "$CONDA_BIN" ] || fail "CONDA_BIN is not executable: $CONDA_BIN"
    printf '%s\n' "$CONDA_BIN"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi

  for candidate in "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  fail "conda was not found. Set CONDA_BIN to your conda executable."
}

default_data_root() {
  if [ -n "${HIMEM_DATA_ROOT:-}" ]; then
    printf '%s\n' "$HIMEM_DATA_ROOT"
  else
    printf '%s\n' "run_outputs/libero_data"
  fi
}

require_project_relative() {
  local name=$1
  local value=$2
  case "$value" in
    /*) fail "$name must be project-relative: $value" ;;
  esac
}

write_libero_config() {
  local python_bin=$1
  local assets_dir=$2
  local datasets_dir=$3

  "$python_bin" - "$assets_dir" "$datasets_dir" <<'PY'
import importlib.util
import pathlib
import sys

assets_dir = pathlib.Path(sys.argv[1]).expanduser().resolve()
datasets_dir = pathlib.Path(sys.argv[2]).expanduser().resolve()

spec = importlib.util.find_spec("libero.libero")
if spec is None or spec.origin is None:
    raise SystemExit("Could not locate installed libero.libero package")

libero_root = pathlib.Path(spec.origin).resolve().parent
config_dir = pathlib.Path.home() / ".libero"
config_dir.mkdir(parents=True, exist_ok=True)
datasets_dir.mkdir(parents=True, exist_ok=True)

config = "\n".join(
    [
        f"assets: {assets_dir}",
        f"bddl_files: {libero_root / 'bddl_files'}",
        f"benchmark_root: {libero_root}",
        f"datasets: {datasets_dir}",
        f"init_states: {libero_root / 'init_files'}",
        "",
    ]
)
(config_dir / "config.yaml").write_text(config)
print(libero_root)
PY
}

link_assets_dir() {
  local libero_root=$1
  local assets_dir=$2
  local package_assets="$libero_root/assets"

  mkdir -p "$assets_dir"

  if [ -L "$package_assets" ]; then
    ln -sfn "$assets_dir" "$package_assets"
    return
  fi

  if [ -e "$package_assets" ]; then
    if [ -d "$package_assets" ] && ! find "$package_assets" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      rmdir "$package_assets"
    else
      local backup="${package_assets}.package-$(date +%Y%m%d%H%M%S)"
      log "Preserving existing package assets at $backup"
      mv "$package_assets" "$backup"
    fi
  fi

  ln -s "$assets_dir" "$package_assets"
}

install_system_packages() {
  case "${HIMEM_INSTALL_SYSTEM_PACKAGES:-auto}" in
    0|false|False|no|No)
      log "Skipping system package installation"
      return
      ;;
    auto)
      if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
        log "Skipping system package installation; run as root with apt-get or set HIMEM_INSTALL_SYSTEM_PACKAGES=1"
        return
      fi
      ;;
    1|true|True|yes|Yes)
      command -v apt-get >/dev/null 2>&1 || fail "apt-get is required when HIMEM_INSTALL_SYSTEM_PACKAGES=1"
      ;;
    *)
      fail "Invalid HIMEM_INSTALL_SYSTEM_PACKAGES=${HIMEM_INSTALL_SYSTEM_PACKAGES}"
      ;;
  esac

  log "Installing headless MuJoCo system libraries"
  DEBIAN_FRONTEND=noninteractive apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y libegl1 libosmesa6 libglu1-mesa
}

download_assets() {
  local python_bin=$1
  local assets_dir=$2

  if [ "${HIMEM_DOWNLOAD_LIBERO_ASSETS:-1}" = "0" ]; then
    log "Skipping LIBERO asset download"
    return
  fi

  mkdir -p "$assets_dir"
  export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

  log "Downloading LIBERO assets to $assets_dir"
  "$python_bin" - "$assets_dir" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

assets_dir = sys.argv[1]
snapshot_download(
    repo_id=os.environ.get("HIMEM_LIBERO_ASSETS_REPO", "jadechoghari/libero-assets"),
    repo_type="model",
    local_dir=assets_dir,
    local_dir_use_symlinks=False,
    max_workers=int(os.environ.get("HF_MAX_WORKERS", "1")),
)
print(assets_dir)
PY
}

install_python_dependencies() {
  local python_bin=$1
  local requirements_file=$2

  if [ -n "${LIBERO_VERSION:-}" ]; then
    log "Installing LIBERO Python dependencies with LIBERO_VERSION override"
    "$python_bin" -m pip install "libero==$LIBERO_VERSION" websockets imageio huggingface_hub
    return
  fi

  log "Installing LIBERO Python dependencies from $requirements_file"
  "$python_bin" -m pip install -r "$requirements_file"
}

main() {
  local script_dir repo_root data_root conda_bin env_prefix python_bin assets_dir datasets_dir libero_root requirements_file
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "$script_dir/.." && pwd)"
  cd "$repo_root"
  data_root="$(default_data_root)"
  require_project_relative "HIMEM_DATA_ROOT" "$data_root"
  env_prefix="${LIBERO_ENV_PREFIX:-$data_root/envs/libero}"
  assets_dir="${LIBERO_ASSETS_DIR:-$data_root/libero/assets}"
  datasets_dir="${LIBERO_DATASETS_DIR:-$data_root/libero/datasets}"
  requirements_file="${HIMEM_LIBERO_REQUIREMENTS:-requirements-libero.txt}"

  require_project_relative "LIBERO_ENV_PREFIX" "$env_prefix"
  require_project_relative "LIBERO_ASSETS_DIR" "$assets_dir"
  require_project_relative "LIBERO_DATASETS_DIR" "$datasets_dir"
  require_project_relative "HIMEM_LIBERO_REQUIREMENTS" "$requirements_file"
  [ -f "$requirements_file" ] || fail "LIBERO requirements file not found: $requirements_file"

  if [ "${HIMEM_SETUP_LIBERO_DRY_RUN:-0}" = "1" ]; then
    printf 'HIMEM_DATA_ROOT=%s\n' "$data_root"
    printf 'LIBERO_ENV_PREFIX=%s\n' "$env_prefix"
    printf 'LIBERO_ASSETS_DIR=%s\n' "$assets_dir"
    printf 'LIBERO_DATASETS_DIR=%s\n' "$datasets_dir"
    printf 'HIMEM_LIBERO_REQUIREMENTS=%s\n' "$requirements_file"
    printf 'HIMEM_DOWNLOAD_LIBERO_ASSETS=%s\n' "${HIMEM_DOWNLOAD_LIBERO_ASSETS:-1}"
    printf 'HIMEM_INSTALL_SYSTEM_PACKAGES=%s\n' "${HIMEM_INSTALL_SYSTEM_PACKAGES:-auto}"
    printf 'CONDA_BIN=%s\n' "${CONDA_BIN:-auto}"
    exit 0
  fi

  conda_bin="$(find_conda)"

  mkdir -p "$data_root" "$data_root/envs" "$data_root/pip-cache" "$data_root/hf-home" "$data_root/hf-cache"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$data_root/pip-cache}"
  export HF_HOME="${HF_HOME:-$data_root/hf-home}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$data_root/hf-cache}"

  install_system_packages

  if [ ! -x "$env_prefix/bin/python" ]; then
    log "Creating LIBERO conda env at $env_prefix"
    "$conda_bin" create -y -p "$env_prefix" "python=${LIBERO_PYTHON_VERSION:-3.8.13}"
  else
    log "Using existing LIBERO env at $env_prefix"
  fi

  python_bin="$env_prefix/bin/python"
  [ -x "$python_bin" ] || fail "Python not found after conda env creation: $python_bin"

  install_python_dependencies "$python_bin" "$requirements_file"

  download_assets "$python_bin" "$assets_dir"
  libero_root="$(write_libero_config "$python_bin" "$assets_dir" "$datasets_dir" | tail -n 1)"
  link_assets_dir "$libero_root" "$assets_dir"

  log "LIBERO env is ready"
  log "LIBERO_PYTHON=$python_bin"
  log "LIBERO assets=$assets_dir"
  log "LIBERO config=$HOME/.libero/config.yaml"
  log "Repository=$repo_root"
}

main "$@"
