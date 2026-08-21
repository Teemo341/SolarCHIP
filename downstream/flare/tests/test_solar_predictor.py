"""Runtime tests for the HMI-only SolarPredictor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from downstream.flare.SolarPredictor import (
    PretrainedCheckpointError,
    SolarPredictor,
)
from downstream.flare.data.class_groups import (
    DEFAULT_CLASS_GROUPS,
    normalize_class_groups,
)
from solarchip.utils.util import instantiate_from_config


def cnn_config(attn_resolutions: list[int] | None = None) -> dict:
    return {
        "target": "solarchip.modules.CNN.AE_CNN",
        "params": {
            "contrastive_dim": 4,
            "ckpt_path": None,
            "ddconfig": {
                "double_z": False,
                "z_channels": 8,
                "resolution": 16,
                "in_channels": 1,
                "out_ch": 1,
                "ch": 8,
                "ch_mult": [1, 1],
                "num_res_blocks": 1,
                "attn_resolutions": []
                if attn_resolutions is None
                else attn_resolutions,
                "use_linear_attn": False,
                "dropout": 0.0,
            },
        },
    }


def vit_config() -> dict:
    return {
        "target": "solarchip.modules.ViT.AE_ViT",
        "params": {
            "contrastive_dim": 4,
            "ckpt_path": None,
            "ddconfig": {
                "input_dim": 1,
                "input_resolution": 16,
                "patch_size": 4,
                "hidden_dim": 16,
                "layers": 1,
                "heads": 4,
            },
        },
    }


def write_solar_checkpoint(
    path: Path,
    config: dict,
    prefix: str = "model_dict.hmi.",
    wrapped: bool = True,
) -> None:
    autoencoder = instantiate_from_config(config)
    state = {
        f"{prefix}{key}": value.detach().clone()
        for key, value in autoencoder.state_dict().items()
    }
    torch.save({"state_dict": state} if wrapped else state, path)


class SyntheticFlareDataset(Dataset):
    def __init__(
        self,
        length: int = 4,
        class_groups: tuple[str, ...] = DEFAULT_CLASS_GROUPS,
    ) -> None:
        self.length = length
        self.class_groups = normalize_class_groups(class_groups)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "hmi": torch.randn(1, 16, 16),
            "label": torch.tensor(index % 3, dtype=torch.long),
        }


class SyntheticFlareDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_groups: tuple[str, ...] = DEFAULT_CLASS_GROUPS,
        validation_groups: tuple[str, ...] = DEFAULT_CLASS_GROUPS,
    ) -> None:
        super().__init__()
        self.train_groups = train_groups
        self.validation_groups = validation_groups

    def setup(self, stage: str | None = None) -> None:
        del stage
        self.datasets = {
            "train": SyntheticFlareDataset(class_groups=self.train_groups),
            "validation": SyntheticFlareDataset(class_groups=self.validation_groups),
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets["train"], batch_size=2)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.datasets["validation"], batch_size=2)


class SolarPredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_predictor(
        self,
        config: dict,
        checkpoint: Path,
        **overrides,
    ) -> SolarPredictor:
        params = {
            "base_model": config,
            "pretrained_ckpt_path": checkpoint,
            "class_groups": list(DEFAULT_CLASS_GROUPS),
            "representation_dim": 12,
            "head_hidden_dim": 10,
            "attention_heads": 2,
            "use_contrastive_residual": True,
            "scheduler": "none",
            "freeze_encoder_epochs": 0,
        }
        params.update(overrides)
        return SolarPredictor(**params)

    def test_shipped_yaml_uses_one_class_group_contract(self) -> None:
        config = OmegaConf.load("downstream/flare/solar_predictor_cnn.yaml")
        model_groups = normalize_class_groups(config.model.params.class_groups)
        train_groups = normalize_class_groups(
            config.data.params.train.params.class_groups
        )
        validation_groups = normalize_class_groups(
            config.data.params.validation.params.class_groups
        )

        self.assertEqual(model_groups, DEFAULT_CLASS_GROUPS)
        self.assertEqual(train_groups, model_groups)
        self.assertEqual(validation_groups, model_groups)

    def test_cnn_forward_keeps_only_hmi_encoder_mapper_and_head(self) -> None:
        config = cnn_config()
        checkpoint = self.root / "cnn.ckpt"
        write_solar_checkpoint(checkpoint, config)
        model = self.make_predictor(config, checkpoint)

        features = model.encode_features(torch.randn(2, 1, 16, 16))
        logits = model(torch.randn(2, 1, 16, 16))

        self.assertEqual(tuple(features.shape), (2, 12))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(model.class_groups, DEFAULT_CLASS_GROUPS)
        self.assertEqual(model.num_classes, 4)
        state_keys = set(model.state_dict())
        self.assertTrue(any(key.startswith("hmi_encoder.") for key in state_keys))
        self.assertFalse(any("decoder" in key for key in state_keys))
        self.assertFalse(any("0094" in key for key in state_keys))
        self.assertIn("contrastive_projector", state_keys)

    def test_vit_uses_raw_cls_main_feature_and_contrastive_residual(self) -> None:
        config = vit_config()
        checkpoint = self.root / "vit.ckpt"
        write_solar_checkpoint(checkpoint, config)
        model = self.make_predictor(config, checkpoint)

        self.assertEqual(model.backbone_kind, "vit")
        self.assertEqual(
            tuple(model.encode_features(torch.randn(2, 1, 16, 16)).shape),
            (2, 12),
        )
        self.assertFalse(any("decoder" in key for key in model.state_dict()))

    def test_optional_residual_can_be_removed_completely(self) -> None:
        config = cnn_config()
        checkpoint = self.root / "cnn-no-residual.ckpt"
        write_solar_checkpoint(checkpoint, config)
        model = self.make_predictor(
            config,
            checkpoint,
            use_contrastive_residual=False,
        )

        state_keys = set(model.state_dict())
        self.assertFalse(any("contrastive" in key for key in state_keys))
        self.assertFalse(any("cnn_cls_proj" in key for key in state_keys))
        self.assertEqual(tuple(model(torch.randn(1, 1, 16, 16)).shape), (1, 4))

    def test_custom_groups_define_head_width_and_class_weight_contract(self) -> None:
        config = cnn_config()
        checkpoint = self.root / "binary-groups.ckpt"
        write_solar_checkpoint(checkpoint, config)
        model = self.make_predictor(
            config,
            checkpoint,
            class_groups=["0ABC", "MX"],
            class_weights=[1.0, 2.0],
        )

        self.assertEqual(model.class_groups, ("0ABC", "MX"))
        self.assertEqual(model.num_classes, 2)
        self.assertEqual(model.classifier[-1].out_features, 2)
        self.assertEqual(tuple(model(torch.randn(1, 1, 16, 16)).shape), (1, 2))

        with self.assertRaisesRegex(ValueError, "must have 2 values"):
            self.make_predictor(
                config,
                checkpoint,
                class_groups=["0ABC", "MX"],
                class_weights=[1.0, 1.0, 1.0, 1.0],
            )
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self.make_predictor(
                config,
                checkpoint,
                class_groups=["0ABC", "MX"],
                class_weights=[0.0, 1.0],
            )

    def test_strict_loader_rejects_wrong_encoder_config(self) -> None:
        checkpoint = self.root / "strict.ckpt"
        write_solar_checkpoint(checkpoint, cnn_config())

        with self.assertRaises(PretrainedCheckpointError):
            self.make_predictor(
                cnn_config(attn_resolutions=[8]),
                checkpoint,
            )

    def test_raw_state_and_module_prefix_are_supported(self) -> None:
        config = cnn_config()
        checkpoint = self.root / "raw.ckpt"
        autoencoder = instantiate_from_config(config)
        raw_state = {
            f"module.model_dict.hmi.{key}": value.detach().clone()
            for key, value in autoencoder.state_dict().items()
        }
        torch.save(raw_state, checkpoint)

        model = self.make_predictor(config, checkpoint)
        self.assertEqual(tuple(model(torch.randn(1, 1, 16, 16)).shape), (1, 4))

    def test_explicit_non_hmi_modal_prefix_is_rejected(self) -> None:
        config = cnn_config()
        checkpoint = self.root / "wrong-modal.ckpt"
        write_solar_checkpoint(
            checkpoint,
            config,
            prefix="model_dict.0094.",
        )

        with self.assertRaises(PretrainedCheckpointError):
            self.make_predictor(
                config,
                checkpoint,
                pretrained_prefix="model_dict.0094",
            )

    def test_freeze_phase_skips_encoder_gradients_but_trains_new_mapping(self) -> None:
        config = cnn_config()
        checkpoint = self.root / "freeze.ckpt"
        write_solar_checkpoint(checkpoint, config)
        model = self.make_predictor(
            config,
            checkpoint,
            freeze_encoder_epochs=1,
        )
        model.train()

        loss = model(torch.randn(2, 1, 16, 16)).sum()
        loss.backward()

        self.assertTrue(
            all(parameter.grad is None for parameter in model.hmi_encoder.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.main_mapper.parameters()
            )
        )
        self.assertTrue(
            all(not module.training for module in model._pretrained_modules())
        )

    def test_unfrozen_encoder_receives_gradients_and_optimizer_groups_are_static(
        self,
    ) -> None:
        config = cnn_config()
        checkpoint = self.root / "unfrozen.ckpt"
        write_solar_checkpoint(checkpoint, config)
        model = self.make_predictor(config, checkpoint, freeze_encoder_epochs=0)
        model.train()

        model(torch.randn(2, 1, 16, 16)).sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.hmi_encoder.parameters()
            )
        )
        optimizer = model.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 2)

    def test_lightning_fast_dev_run_logs_validation_metrics(self) -> None:
        config = cnn_config()
        checkpoint = self.root / "trainer.ckpt"
        write_solar_checkpoint(checkpoint, config)
        model = self.make_predictor(config, checkpoint)
        loader = DataLoader(SyntheticFlareDataset(), batch_size=2)
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )

        trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

        self.assertIn("val_loss", trainer.callback_metrics)
        self.assertIn("val_macro_f1", trainer.callback_metrics)
        self.assertIn("val_balanced_accuracy", trainer.callback_metrics)

    def test_training_rejects_mismatched_dataset_class_groups(self) -> None:
        config = cnn_config()
        checkpoint = self.root / "mismatched-groups.ckpt"
        write_solar_checkpoint(checkpoint, config)
        model = self.make_predictor(config, checkpoint)
        loader = DataLoader(
            SyntheticFlareDataset(class_groups=("0", "ABC", "M", "X")),
            batch_size=2,
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )

        with self.assertRaisesRegex(ValueError, "must match exactly"):
            trainer.fit(model, train_dataloaders=loader)

    def test_fit_checks_every_datamodule_split_before_first_batch(self) -> None:
        config = cnn_config()
        checkpoint = self.root / "mismatched-validation-groups.ckpt"
        write_solar_checkpoint(checkpoint, config)
        model = self.make_predictor(config, checkpoint)
        datamodule = SyntheticFlareDataModule(validation_groups=("0", "ABC", "M", "X"))
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )

        with self.assertRaisesRegex(ValueError, "validation"):
            trainer.fit(model, datamodule=datamodule)

    def test_downstream_checkpoint_restores_without_original_pretraining_file(
        self,
    ) -> None:
        config = cnn_config()
        base_checkpoint = self.root / "portable-base.ckpt"
        downstream_checkpoint = self.root / "portable-downstream.ckpt"
        write_solar_checkpoint(base_checkpoint, config)
        model = self.make_predictor(config, base_checkpoint)
        loader = DataLoader(SyntheticFlareDataset(), batch_size=2)
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=0,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(model, train_dataloaders=loader)

        test_input = torch.randn(2, 1, 16, 16)
        model.eval()
        with torch.no_grad():
            expected_logits = model(test_input).clone()
        trainer.save_checkpoint(downstream_checkpoint)
        completed_global_step = trainer.global_step

        with self.assertRaises(PretrainedCheckpointError):
            SolarPredictor.load_from_checkpoint(
                downstream_checkpoint,
                map_location="cpu",
                class_groups=["0", "ABC", "M", "X"],
            )

        base_checkpoint.unlink()

        restored = SolarPredictor.load_from_checkpoint(
            downstream_checkpoint,
            map_location="cpu",
        )
        restored.eval()
        with torch.no_grad():
            restored_logits = restored(test_input)
        torch.testing.assert_close(restored_logits, expected_logits)

        fresh_without_weights = self.make_predictor(config, base_checkpoint)
        with self.assertRaises(FileNotFoundError):
            fresh_without_weights(test_input)

        resume_model = self.make_predictor(config, base_checkpoint)
        resume_trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=2,
            limit_train_batches=1,
            limit_val_batches=0,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        resume_trainer.fit(
            resume_model,
            train_dataloaders=loader,
            ckpt_path=downstream_checkpoint,
        )
        self.assertGreater(resume_trainer.global_step, completed_global_step)

    def test_real_checkpoint_strictly_loads_if_available(self) -> None:
        checkpoint = Path(
            "checkpoints/solarchip/"
            "solarchip_CNN_AE_base_zscore_2026-07-10T18-03-45.ckpt"
        )
        if not checkpoint.is_file():
            self.skipTest("Repository SolarCHIP checkpoint is unavailable")
        config = {
            "target": "solarchip.modules.CNN.AE_CNN",
            "params": {
                "contrastive_dim": 32,
                "ckpt_path": None,
                "ddconfig": {
                    "double_z": False,
                    "z_channels": 32,
                    "resolution": 1024,
                    "in_channels": 1,
                    "out_ch": 1,
                    "ch": 32,
                    "ch_mult": [1, 1, 1, 2, 2, 2],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "use_linear_attn": False,
                    "dropout": 0.0,
                },
            },
        }

        model = SolarPredictor(
            base_model=config,
            pretrained_ckpt_path=checkpoint,
            pretrained_prefix="model_dict.hmi",
            use_contrastive_residual=True,
        )
        self.assertFalse(any("decoder" in key for key in model.state_dict()))
        self.assertEqual(model.backbone_kind, "cnn")
        self.assertEqual(model.num_classes, 4)


if __name__ == "__main__":
    unittest.main()
