#!/usr/bin/env bash
python -m downstream.flare.test \
  -r logs/flare/solar_predictor_cnn_full/2026-09-01T16-13-31/checkpoints/last.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/flare/solar_predictor_cnn_full/2026-09-01T16-13-31/checkpoints/epoch=000128-val_macro_f1=0.4157.ckpt \
  --metrics all

# python -m downstream.flare.test \
#   -r logs/compare_flare/deepswm_k7/2026-08-31T16-05-59/checkpoints/last.ckpt \
#   --metrics all

# python -m downstream.flare.test \
#   -r logs/compare_flare/deepswm_k7/2026-08-31T16-05-59/checkpoints/epoch=000001.ckpt \
#   --metrics all

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
