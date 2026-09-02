#!/usr/bin/env bash
set -euo pipefail

# Evaluate the deterministic 20% validation subset recorded by each unidata run.
# Both last.ckpt and the best validation checkpoint are tested, following
# shells/test/test_flare.sh.

python -m downstream.flare.test \
  -r logs/flare/solar_predictor_cnn_full_unidata/2026-09-02T11-03-42/checkpoints/last.ckpt \
  --split validation \
  --metrics all

python -m downstream.flare.test \
  -r logs/flare/solar_predictor_cnn_full_unidata/2026-09-02T11-03-42/checkpoints/epoch=000084-val_macro_f1=0.5431.ckpt \
  --split validation \
  --metrics all

python -m downstream.flare.test \
  -r logs/flare/solar_predictor_vit_full_unidata/2026-09-02T10-42-57/checkpoints/last.ckpt \
  --split validation \
  --metrics all

python -m downstream.flare.test \
  -r logs/flare/solar_predictor_vit_full_unidata/2026-09-02T10-42-57/checkpoints/epoch=000027-val_macro_f1=0.4774.ckpt \
  --split validation \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/pcnn_unidata/2026-09-02T10-22-56/checkpoints/last.ckpt \
  --split validation \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/pcnn_unidata/2026-09-02T10-22-56/checkpoints/epoch=000013.ckpt \
  --split validation \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/yi2023_dqn_unidata/2026-09-02T10-23-14/checkpoints/last.ckpt \
  --split validation \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/yi2023_dqn_unidata/2026-09-02T10-23-14/checkpoints/epoch=000050.ckpt \
  --split validation \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_unidata/2026-09-02T10-23-23/checkpoints/last.ckpt \
  --split validation \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_unidata/2026-09-02T10-23-23/checkpoints/macro-f1-000015-0.1799.ckpt \
  --split validation \
  --metrics all
