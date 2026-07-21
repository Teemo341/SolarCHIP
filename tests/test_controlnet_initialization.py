import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from downstream.ldm.SolarLDM import SolarLDM
from tests.test_solarcontrol_optimizer import (
    TinyControlNet,
    build_control,
    optimizer_parameter_ids,
)


class MismatchedTinyControlNet(TinyControlNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_blocks = nn.ModuleList([nn.Linear(3, 3)])


def build_source_sd():
    return SolarLDM(
        solarchip_config={
            "target": "tests.test_solarldm_unconditional.TinySolarWrapper",
            "params": {"modal_list": ["hmi"]},
        },
        unet_config={
            "target": "tests.test_solarcontrol_optimizer.TinyControlledUNet",
            "params": {},
        },
        first_stage_key="hmi",
        cond_stage_key="hmi",
        cond_stage_config="__is_unconditional__",
        concat_mode=False,
        learning_rate=1.0e-4,
        image_size=4,
        channels=1,
        timesteps=10,
        linear_start=1.0e-4,
        linear_end=2.0e-2,
        use_ema=False,
        normalize_latent_per_channel=True,
        latent_mean=[3.0],
        latent_std=[2.0],
    )


def make_sd_checkpoint(path, include_ema=True, include_stats=True):
    source = build_source_sd()
    with torch.no_grad():
        for index, parameter in enumerate(
            source.model.diffusion_model.parameters(), start=1
        ):
            parameter.fill_(index / 10.0)

    state_dict = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
    }
    if not include_stats:
        for key in ("latent_mean", "latent_std", "latent_stats_initialized"):
            state_dict.pop(key)

    ema_parameters = {}
    if include_ema:
        for name, parameter in source.model.named_parameters():
            ema_value = parameter.detach().clone() + 10.0
            state_dict[f"model_ema.{name.replace('.', '')}"] = ema_value
            if name.startswith("diffusion_model."):
                ema_parameters[name.removeprefix("diffusion_model.")] = ema_value

    torch.save({"state_dict": state_dict}, path)
    raw_parameters = {
        name: parameter.detach().clone()
        for name, parameter in source.model.diffusion_model.named_parameters()
    }
    return raw_parameters, ema_parameters


def assert_module_states_equal(test_case, first, second):
    first_state = first.state_dict()
    second_state = second.state_dict()
    test_case.assertEqual(set(first_state), set(second_state))
    for key in first_state:
        test_case.assertTrue(torch.equal(first_state[key], second_state[key]), key)


class ControlNetInitializationTest(unittest.TestCase):
    def test_ema_backbone_initializes_main_and_control_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "sd.ckpt"
            _, ema_parameters = make_sd_checkpoint(checkpoint_path)

            model = build_control(
                sd_locked=True,
                sd_backbone_ckpt=str(checkpoint_path),
                sd_backbone_use_ema=True,
                normalize_latent_per_channel=True,
            )

        for name, parameter in model.model.diffusion_model.named_parameters():
            self.assertTrue(torch.equal(parameter, ema_parameters[name]), name)
        assert_module_states_equal(
            self,
            model.model.diffusion_model.time_embed,
            model.control_model.time_embed,
        )
        assert_module_states_equal(
            self,
            model.model.diffusion_model.input_blocks,
            model.control_model.input_blocks,
        )
        assert_module_states_equal(
            self,
            model.model.diffusion_model.middle_block,
            model.control_model.middle_block,
        )
        self.assertTrue(torch.equal(model.latent_mean, torch.tensor([[[[3.0]]]])))
        self.assertTrue(torch.equal(model.latent_std, torch.tensor([[[[2.0]]]])))
        self.assertTrue(model.latent_stats_initialized.item())
        self.assertFalse(model.cond_latent_stats_initialized.item())

        zero_modules = list(model.control_model.zero_convs) + [
            model.control_model.middle_block_out,
            list(model.control_model.input_hint_block.children())[-1],
        ]
        self.assertTrue(
            all(
                torch.count_nonzero(parameter).item() == 0
                for module in zero_modules
                for parameter in module.parameters()
            )
        )
        optimizer = model.configure_optimizers()
        self.assertEqual(
            optimizer_parameter_ids(optimizer),
            {id(parameter) for parameter in model.control_model.parameters()},
        )

    def test_raw_backbone_can_be_selected_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "sd.ckpt"
            raw_parameters, _ = make_sd_checkpoint(checkpoint_path)

            model = build_control(
                sd_locked=True,
                sd_backbone_ckpt=str(checkpoint_path),
                sd_backbone_use_ema=False,
                normalize_latent_per_channel=True,
            )

        for name, parameter in model.model.diffusion_model.named_parameters():
            self.assertTrue(torch.equal(parameter, raw_parameters[name]), name)

    def test_missing_ema_or_latent_stats_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            no_ema = Path(directory) / "no_ema.ckpt"
            no_stats = Path(directory) / "no_stats.ckpt"
            make_sd_checkpoint(no_ema, include_ema=False)
            make_sd_checkpoint(no_stats, include_stats=False)

            with self.assertRaisesRegex(RuntimeError, "缺少 EMA 参数"):
                build_control(
                    sd_locked=True,
                    sd_backbone_ckpt=str(no_ema),
                    normalize_latent_per_channel=True,
                )
            with self.assertRaisesRegex(RuntimeError, "缺少逐通道 HMI latent stats"):
                build_control(
                    sd_locked=True,
                    sd_backbone_ckpt=str(no_stats),
                    normalize_latent_per_channel=True,
                )

    def test_encoder_structure_mismatch_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "sd.ckpt"
            make_sd_checkpoint(checkpoint_path)

            with self.assertRaisesRegex(RuntimeError, "input_blocks.*结构不一致"):
                build_control(
                    sd_locked=True,
                    sd_backbone_ckpt=str(checkpoint_path),
                    normalize_latent_per_channel=True,
                    control_stage_config={
                        "target": (
                            "tests.test_controlnet_initialization."
                            "MismatchedTinyControlNet"
                        ),
                        "params": {},
                    },
                )


if __name__ == "__main__":
    unittest.main()
