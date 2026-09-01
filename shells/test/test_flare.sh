#!/usr/bin/env bash
python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_k1_weighted_ce_only/2026-09-01T10-15-26/checkpoints/last.ckpt \
  --metrics all

python -m downstream.flare.test \
  -r logs/compare_flare/deepswm_k1_weighted_ce_only/2026-09-01T10-15-26/checkpoints/macro-f1-000004-0.1604.ckpt \
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
