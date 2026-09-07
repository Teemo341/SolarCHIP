#!/usr/bin/env bash
set -euo pipefail

flare_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
flare_repo_root="$(cd -- "${flare_script_dir}/../.." && pwd)"
cd "${flare_repo_root}"

for flare_ratio in 10 20 30 40 50 60 70 80 90; do
  echo "Training ViT flare split ratio ${flare_ratio}%"
  python -m solarchip.main.train \
    -b "configs/flare_split/solar_predictor_vit_ratio${flare_ratio}.yaml"
done
