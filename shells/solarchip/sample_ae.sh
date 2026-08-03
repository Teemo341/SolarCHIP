#!/usr/bin/env bash
# Sample images from a trained SolarCHIP AE model using the validation set.
#
# Usage:  bash shells/solarchip/sample_ae.sh
#
# Customize the variables below to match your checkpoint and sampling needs.

# ---- Configuration (edit these as needed) ----
CONFIG="configs/solarchip/CNN_AE_base_zscore.yaml"
CKPT="logs/solarchip/CNN_AE_base_zscore/checkpoints/epoch=000187_val_loss=1.0262.ckpt"
OUTDIR="logs/sample_ae_output"
NUM_BATCHES=1        # How many validation batches to sample
MAX_IMAGES=4          # Max images per batch per modal

python -m solarchip.main.sample \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --outdir "${OUTDIR}" \
    --num_batches "${NUM_BATCHES}" \
    --max_images "${MAX_IMAGES}"
