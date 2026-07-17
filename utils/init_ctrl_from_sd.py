#!/usr/bin/env python3
"""
从训练好的无条件 SolarLDM checkpoint 中提取 UNet 权重,
注入到 SolarControl 模型作为冻结 backbone。

用法:
  Step 1: 训练无条件 SolarLDM
    python -m solarchip.main.train -b configs/solarldm/sd_hmi_uncond.yaml

  Step 2: 初始化 ControlNet
    python scripts/train/init_ctrl_from_sd.py \
        --sd_ckpt logs/sd_hmi_uncond_xxx/checkpoints/last.ckpt \
        --ctrl_config configs/solarldm/ctrl_0094-hmi.yaml \
        --output logs/sd_hmi_uncond_xxx/checkpoints/ctrl_init.ckpt

  Step 3: 训练 ControlNet (sd_locked=True, ckpt_path=ctrl_init.ckpt)
    python -m solarchip.main.train -b configs/solarldm/ctrl_0094-hmi.yaml

输出 checkpoint 可以直接作为 ControlNet 训练的起点,
或用于推理测试 ControlNet 的初始状态(此时 ControlNet 输出为零,
采样结果应和 Step1 的无条件模型完全一致)。
"""

import argparse
import torch
import os

from omegaconf import OmegaConf
from solarchip.utils.util import instantiate_from_config


def transfer_weights(sd_state_dict, ctrl_model):
    """
    将 SolarLDM 的 UNet 权重复制到 SolarControl 的主 UNet。
    SolarLDM:  model.diffusion_model.*    (DiffusionWrapper → UNetModel)
    SolarControl: model.diffusion_model.*  (DiffusionWrapper → SolarControlledUnetModel)
    两者参数名完全一致(SolarControlledUnetModel 只覆写 forward, 不新增参数)。
    """
    # 只取 UNet 相关的 key
    unet_prefix = "model.diffusion_model."
    unet_keys = {k: v for k, v in sd_state_dict.items() if k.startswith(unet_prefix)}

    ctrl_sd = ctrl_model.state_dict()
    transferred = 0
    skipped = 0

    for k in unet_keys:
        if k in ctrl_sd:
            if ctrl_sd[k].shape == unet_keys[k].shape:
                ctrl_sd[k] = unet_keys[k].clone()
                transferred += 1
            else:
                print(f"  Shape mismatch: {k}: SD={unet_keys[k].shape}, Ctrl={ctrl_sd[k].shape}")
                skipped += 1
        else:
            # ControlNet 特有的 key (如 control_model.*), 不在 SD 里, 正常跳过
            pass

    ctrl_model.load_state_dict(ctrl_sd, strict=False)
    print(f"Transferred {transferred} UNet params, skipped {skipped} (shape mismatch)")
    return ctrl_model


def main():
    parser = argparse.ArgumentParser(
        description="Initialize SolarControl from trained SolarLDM UNet weights"
    )
    parser.add_argument("--sd_ckpt", required=True, help="Path to trained SolarLDM checkpoint")
    parser.add_argument("--ctrl_config", required=True, help="Path to SolarControl config yaml")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()

    # ---- Load SD checkpoint ----
    print(f"Loading SD checkpoint: {args.sd_ckpt}")
    sd_ckpt = torch.load(args.sd_ckpt, map_location="cpu")
    if "state_dict" in sd_ckpt:
        sd_state_dict = sd_ckpt["state_dict"]
    else:
        sd_state_dict = sd_ckpt
    print(f"SD state_dict has {len(sd_state_dict)} keys")

    # ---- Build SolarControl from config ----
    print(f"Building SolarControl from: {args.ctrl_config}")
    cfg = OmegaConf.load(args.ctrl_config)
    ctrl_model = instantiate_from_config(cfg.model)
    ctrl_model.eval()

    # ---- Transfer UNet weights ----
    print("Transferring UNet weights...")
    ctrl_model = transfer_weights(sd_state_dict, ctrl_model)

    # ---- Save ----
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ctrl_sd = ctrl_model.state_dict()

    # 包装成 Lightning checkpoint 格式, 方便 resume
    out_ckpt = {
        "state_dict": ctrl_sd,
        "epoch": 0,
        "global_step": 0,
    }

    torch.save(out_ckpt, args.output)
    print(f"Saved initialized ControlNet checkpoint to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. In {args.ctrl_config}, set sd_locked: True")
    print(f"  2. Train with: python -m solarchip.main.train -b {args.ctrl_config}")
    print(f"     (Lightning will auto-load ctrl_init.ckpt if set as ckpt_path)")


if __name__ == "__main__":
    main()
