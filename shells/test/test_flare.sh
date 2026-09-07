#!/usr/bin/env bash
python -m downstream.flare.test \
  -r logs/flare/solar_predictor_cnn_scratch/2026-09-05T23-38-14/checkpoints/epoch=000036-val_macro_f1=0.5079.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/flare/solar_predictor_cnn_scratch/2026-09-05T23-38-14/checkpoints/last.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/flare/solar_predictor_vit_scratch/2026-09-05T23-40-56/checkpoints/epoch=000032-val_macro_f1=0.4329.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/flare/solar_predictor_vit_scratch/2026-09-05T23-40-56/checkpoints/last.ckpt \
  --metrics all

# python -m downstream.flare.test \
#   -r logs/compare_flare/deepswm_k14/2026-08-31T16-06-06/checkpoints/last.ckpt \
#   --metrics all

# python -m downstream.flare.test \
#   -r logs/compare_flare/deepswm_k14/2026-08-31T16-06-06/checkpoints/epoch=000008.ckpt \
#   --metrics all

# python -m downstream.flare.test \
#   -r logs/compare_flare/deepswm_k28/2026-08-31T16-06-13/checkpoints/last.ckpt \
#   --metrics all

# python -m downstream.flare.test \
#   -r logs/compare_flare/deepswm_k28/2026-08-31T16-06-13/checkpoints/epoch=000000.ckpt \
#   --metrics all
