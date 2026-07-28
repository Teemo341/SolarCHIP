"""
Load a trained SolarCHIP AE checkpoint, sample images from the validation dataloader,
and save them using the same SolarImageLogger format as during training.

Usage:
    python -m solarchip.main.sample \
        --config configs/solarchip/CNN_AE_base_zscore.yaml \
        --ckpt logs/solarchip_CNN_AE_base_zscore_2026-07-10T18-03-45/checkpoints/epoch=000187_val_loss=1.0262.ckpt \
        --outdir logs/sample_ae_output \
        --num_batches 5 \
        --max_images 4
"""

import argparse
import os
import torch
from omegaconf import OmegaConf

from solarchip.utils.util import instantiate_from_config
from solarchip.utils.callback import SolarImageLogger


def get_parser():
    parser = argparse.ArgumentParser(description="SolarCHIP AE sampling")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config YAML (same as used for training).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the checkpoint .ckpt file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="logs/sample_ae",
        help="Output directory for sampled images (default: logs/sample_ae).",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of validation batches to sample (default: 10).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=4,
        help="Max images to save per batch per modal (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available, else cpu).",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # ---- 1. Load config ----
    config = OmegaConf.load(args.config)

    # ---- 2. Instantiate model ----
    model = instantiate_from_config(config.model)

    # ---- 3. Load checkpoint weights ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # LightningModule.global_step / .current_epoch are read-only properties,
    # so we wrap the model in a thin proxy that exposes them as plain attributes
    # for SolarImageLogger filename generation.
    gs = ckpt.get("global_step", 0)
    epoch = ckpt.get("epoch", 0)

    class _ModelProxy:
        """Delegate all attribute access to the model except global_step & current_epoch."""
        def __init__(self, model, global_step, current_epoch):
            self._model = model
            self.global_step = global_step
            self.current_epoch = current_epoch

        def __getattr__(self, name):
            return getattr(self._model, name)

    model_proxy = _ModelProxy(model, gs, epoch)

    # ---- 4. Move model to device and set eval mode ----
    model.to(args.device)
    model.eval()
    torch.set_grad_enabled(False)

    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"  global_step = {gs}, epoch = {epoch}")
    print(f"  device = {args.device}")

    # ---- 5. Set up data module (validation only) ----
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    val_loader = data.val_dataloader()
    print(f"Validation dataloader: {len(val_loader)} batches")

    # ---- 6. Set up SolarImageLogger (same as training callback) ----
    image_logger = SolarImageLogger(max_images=args.max_images, batch_frequency=1)

    # ---- 7. Sample and save ----
    os.makedirs(args.outdir, exist_ok=True)
    total_batches = min(args.num_batches, len(val_loader))

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= total_batches:
            break

        # Move batch to device
        batch = {k: v.to(args.device) for k, v in batch.items()}

        images = model.log_images(batch)
        image_logger.log_local(args.outdir, "val", images, batch_idx, model_proxy)
        print(f"Saved batch {batch_idx + 1}/{total_batches}")

    print(f"Done. Images saved to {os.path.join(args.outdir, 'images', 'val')}")


if __name__ == "__main__":
    main()
