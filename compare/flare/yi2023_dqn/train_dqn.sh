#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
python_bin="${SOLARCHIP_PYTHON_BIN:-python}"

cd "${repo_root}"
exec "${python_bin}" -m solarchip.main.train \
  -b configs/compare_flare/yi2023_dqn.yaml \
  --seed 42 \
  "$@"
