#!/usr/bin/env bash
python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_k4/2026-08-31T16-05-55/checkpoints/last.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_k4/2026-08-31T16-05-55/checkpoints/last.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_k7/2026-08-31T16-05-59/checkpoints/last.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_k7/2026-08-31T16-05-59/checkpoints/last.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_k14/2026-08-31T16-06-06/checkpoints/last.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_k14/2026-08-31T16-06-06/checkpoints/last.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_k28/2026-08-31T16-06-13/checkpoints/last.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_k28/2026-08-31T16-06-13/checkpoints/last.ckpt \
  --metrics all
