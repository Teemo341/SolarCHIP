#!/usr/bin/env bash

python utils/train/init_ctrl_from_sd.py \
    --sd_ckpt logs/sd_hmi_uncond_xxx/checkpoints/last.ckpt \
    --ctrl_config configs/solarldm/ctrl_0094-hmi.yaml \
    --output logs/xxx/ctrl_init.ckpt