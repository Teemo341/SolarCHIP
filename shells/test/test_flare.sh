#!/usr/bin/env bash
python -m downstream.flare.test \
  -r logs/compare_flare/deepswm/2026-08-30T22-48-11/checkpoints/epoch=000029.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/pcnn/2026-08-30T22-50-28/checkpoints/epoch=000013.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/yi2023_dqn/2026-08-30T22-52-11/checkpoints/epoch=000044.ckpt \
  --metrics all