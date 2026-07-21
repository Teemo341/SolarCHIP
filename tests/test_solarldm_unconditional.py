import unittest
from unittest import mock

import torch
import torch.nn as nn

from downstream.ldm.SolarLDM import SolarLDM


class CountingAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode_calls = 0

    def encode(self, x):
        self.encode_calls += 1
        return x

    def decode(self, z):
        return z


class TinySolarWrapper(nn.Module):
    def __init__(self, modal_list, **kwargs):
        super().__init__()
        self.model_dict = nn.ModuleDict({modal: CountingAE() for modal in modal_list})

    def get_model(self, modal):
        return self.model_dict[modal]


class TinyUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        return torch.zeros_like(x) + self.weight


def build_model(unconditional, cond_stage_key="hmi"):
    modal_list = ["hmi"] if unconditional else ["hmi", cond_stage_key]
    kwargs = {
        "solarchip_config": {
            "target": "tests.test_solarldm_unconditional.TinySolarWrapper",
            "params": {"modal_list": modal_list},
        },
        "unet_config": {
            "target": "tests.test_solarldm_unconditional.TinyUNet",
            "params": {},
        },
        "first_stage_key": "hmi",
        "cond_stage_key": cond_stage_key,
        "learning_rate": 1.0e-4,
        "image_size": 4,
        "channels": 1,
        "timesteps": 10,
        "linear_start": 1.0e-4,
        "linear_end": 2.0e-2,
        "use_ema": False,
    }
    if unconditional:
        kwargs.update(
            cond_stage_config="__is_unconditional__",
            concat_mode=False,
        )
    else:
        kwargs["conditioning_key"] = "concat"
    return SolarLDM(**kwargs)


class SolarLDMUnconditionalTest(unittest.TestCase):
    def test_unconditional_model_has_no_conditioning_stage(self):
        model = build_model(unconditional=True)

        self.assertIsNone(model.model.conditioning_key)
        self.assertIsNone(model.cond_stage_model)
        self.assertIsInstance(model.first_stage_model, CountingAE)

    def test_unconditional_get_input_never_encodes_condition(self):
        model = build_model(unconditional=True)
        batch = {"hmi": torch.randn(2, 1, 4, 4)}

        z, c = model.get_input(batch, "hmi")

        self.assertEqual(z.shape, batch["hmi"].shape)
        self.assertIsNone(c)
        self.assertEqual(model.first_stage_model.encode_calls, 1)

    def test_unconditional_log_images_still_samples(self):
        model = build_model(unconditional=True)
        batch = {"hmi": torch.randn(2, 1, 4, 4)}
        samples = torch.randn(2, 1, 4, 4)

        with mock.patch.object(
            model, "sample_log", return_value=(samples, [])
        ) as sample_log:
            logs = model.log_images(batch, N=2, sample=True)

        sample_key = "visualization/hmi/samples"
        self.assertIn(sample_key, logs)
        self.assertEqual(logs[sample_key].shape, samples.shape)
        sample_log.assert_called_once_with(
            cond=None,
            batch_size=2,
            ddim=True,
            ddim_steps=50,
            eta=1.0,
        )

    def test_conditional_path_still_encodes_requested_modality(self):
        model = build_model(unconditional=False, cond_stage_key="0094")
        batch = {
            "hmi": torch.randn(2, 1, 4, 4),
            "0094": torch.randn(2, 1, 4, 4),
        }

        _, c = model.get_input(batch, "hmi")

        self.assertEqual(model.model.conditioning_key, "concat")
        self.assertIsNotNone(c)
        self.assertEqual(model.first_stage_model.encode_calls, 1)
        self.assertEqual(model.cond_stage_model.encode_calls, 1)


if __name__ == "__main__":
    unittest.main()
