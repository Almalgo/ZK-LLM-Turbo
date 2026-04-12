#!/usr/bin/env bash
set -euo pipefail

# Build TenSEAL against a local SEAL + Intel HEXL toolchain without replacing the
# system Python environment. The resulting wheel can be installed into a venv.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/.build/tenseal-hexl}"
PREFIX_DIR="${PREFIX_DIR:-$ROOT_DIR/.local/tenseal-hexl}"
WHEEL_DIR="${WHEEL_DIR:-$ROOT_DIR/dist/hexl}"

HEXL_REPO="${HEXL_REPO:-https://github.com/intel/hexl.git}"
HEXL_REF="${HEXL_REF:-v1.2.5}"
SEAL_REPO="${SEAL_REPO:-https://github.com/microsoft/SEAL.git}"
SEAL_REF="${SEAL_REF:-v4.1.1}"
TENSEAL_REPO="${TENSEAL_REPO:-https://github.com/OpenMined/TenSEAL.git}"
TENSEAL_REF="${TENSEAL_REF:-v0.3.15}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
JOBS="${JOBS:-$(nproc)}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

check_python_headers() {
  local include_dir
  include_dir="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths().get("include", ""))')"
  if [[ -z "$include_dir" || ! -f "$include_dir/Python.h" ]]; then
    echo "Python development headers not found for interpreter: $PYTHON_BIN" >&2
    echo "Expected header: $include_dir/Python.h" >&2
    echo "Install the matching python-dev package (for example: python3.12-dev) and retry." >&2
    exit 1
  fi
}

require_cmd git
require_cmd cmake
require_cmd "$PYTHON_BIN"
check_python_headers

mkdir -p "$WORK_DIR" "$PREFIX_DIR" "$WHEEL_DIR"

clone_or_update() {
  local repo_url="$1"
  local ref="$2"
  local target_dir="$3"

  if [[ ! -d "$target_dir/.git" ]]; then
    git clone --depth 1 --branch "$ref" "$repo_url" "$target_dir"
  else
    git -C "$target_dir" fetch --depth 1 origin "$ref"
    git -C "$target_dir" checkout "$ref"
    git -C "$target_dir" pull --ff-only origin "$ref"
  fi
}

clone_or_update "$HEXL_REPO" "$HEXL_REF" "$WORK_DIR/hexl"
clone_or_update "$SEAL_REPO" "$SEAL_REF" "$WORK_DIR/SEAL"
clone_or_update "$TENSEAL_REPO" "$TENSEAL_REF" "$WORK_DIR/TenSEAL"

cmake -S "$WORK_DIR/hexl" -B "$WORK_DIR/hexl/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHEXL_BENCHMARK=OFF \
  -DHEXL_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$PREFIX_DIR"
cmake --build "$WORK_DIR/hexl/build" -j"$JOBS"
cmake --install "$WORK_DIR/hexl/build"

cmake -S "$WORK_DIR/SEAL" -B "$WORK_DIR/SEAL/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSEAL_USE_INTEL_HEXL=ON \
  -DSEAL_BUILD_DEPS=ON \
  -DSEAL_BUILD_SEAL_C=OFF \
  -DSEAL_BUILD_EXAMPLES=OFF \
  -DSEAL_BUILD_TESTS=OFF \
  -DCMAKE_PREFIX_PATH="$PREFIX_DIR" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX_DIR"
cmake --build "$WORK_DIR/SEAL/build" -j"$JOBS"
cmake --install "$WORK_DIR/SEAL/build"

pushd "$WORK_DIR/TenSEAL" >/dev/null
  export CMAKE_PREFIX_PATH="$PREFIX_DIR"
  export CMAKE_ARGS="-DSEAL_USE_INTEL_HEXL=ON -DCMAKE_PREFIX_PATH=$PREFIX_DIR"
  "$PYTHON_BIN" -m pip wheel . --no-deps -w "$WHEEL_DIR"
popd >/dev/null

echo "Built HEXL-enabled TenSEAL wheel(s) in: $WHEEL_DIR"
echo "Install into your environment with:"
echo "  $PYTHON_BIN -m pip install --force-reinstall $WHEEL_DIR/tenseal-"'*.whl'
