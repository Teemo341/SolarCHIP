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


def build_model(unconditional, cond_stage_key="hmi", channels=1, **model_kwargs):
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
        "channels": channels,
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
    kwargs.update(model_kwargs)
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


class SolarLDMLatentNormalizationTest(unittest.TestCase):
    def test_explicit_per_channel_stats_round_trip(self):
        model = build_model(
            unconditional=True,
            channels=2,
            normalize_latent_per_channel=True,
            latent_mean=[1.0, -2.0],
            latent_std=[2.0, 4.0],
        )
        x = torch.tensor(
            [[[[1.0, 3.0], [5.0, 7.0]], [[-2.0, 2.0], [6.0, 10.0]]]]
        )

        z, c = model.get_input({"hmi": x}, "hmi")
        xrec = model.decode_first_stage(z)

        expected = torch.tensor(
            [[[[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]]]]
        )
        self.assertIsNone(c)
        self.assertTrue(torch.allclose(z, expected))
        self.assertTrue(torch.allclose(xrec, x))
        self.assertIn("latent_mean", model.state_dict())
        self.assertIn("latent_std", model.state_dict())

        differentiable_z = z.detach().requires_grad_(True)
        model.differentiable_decode_first_stage(differentiable_z).sum().backward()
        expected_grad = torch.tensor([2.0, 4.0]).view(1, 2, 1, 1)
        self.assertTrue(
            torch.allclose(differentiable_z.grad, expected_grad.expand_as(z))
        )

    def test_full_dataloader_stats_are_channelwise_and_checkpointed(self):
        model = build_model(
            unconditional=True,
            channels=2,
            normalize_latent_per_channel=True,
        )
        x = torch.tensor(
            [
                [[[0.0, 2.0], [4.0, 6.0]], [[10.0, 14.0], [18.0, 22.0]]],
                [[[2.0, 4.0], [6.0, 8.0]], [[14.0, 18.0], [22.0, 26.0]]],
            ]
        )
        batch = {"hmi": x}

        model.initialize_latent_stats_from_dataloader(
            [{"hmi": x[:1]}, {"hmi": x[1:]}]
        )
        z, _ = model.get_input(batch, "hmi")

        expected_mean = x.mean(dim=(0, 2, 3), keepdim=True)
        expected_std = x.std(dim=(0, 2, 3), correction=0, keepdim=True)
        self.assertTrue(model.latent_stats_initialized.item())
        self.assertTrue(torch.allclose(model.latent_mean, expected_mean))
        self.assertTrue(torch.allclose(model.latent_std, expected_std))
        self.assertTrue(
            torch.allclose(z.mean(dim=(0, 2, 3)), torch.zeros(2), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                z.std(dim=(0, 2, 3), correction=0), torch.ones(2), atol=1e-6
            )
        )
        self.assertTrue(torch.allclose(model.decode_first_stage(z), x))

        restored = build_model(
            unconditional=True,
            channels=2,
            normalize_latent_per_channel=True,
        )
        restored.load_state_dict(model.state_dict())
        restored_z, _ = restored.get_input(batch, "hmi")
        self.assertTrue(torch.allclose(restored_z, z))

    def test_condition_uses_its_own_channel_stats(self):
        model = build_model(
            unconditional=False,
            cond_stage_key="0094",
            normalize_latent_per_channel=True,
            latent_mean=[1.0],
            latent_std=[2.0],
            cond_latent_mean=[2.0],
            cond_latent_std=[4.0],
        )
        batch = {
            "hmi": torch.full((2, 1, 2, 2), 5.0),
            "0094": torch.full((2, 1, 2, 2), 10.0),
        }

        z, c = model.get_input(batch, "hmi")

        self.assertTrue(torch.allclose(z, torch.full_like(z, 2.0)))
        self.assertTrue(torch.allclose(c, torch.full_like(c, 2.0)))
        self.assertTrue(model.cond_latent_stats_initialized.item())

    def test_full_dataloader_initializes_target_and_condition_separately(self):
        model = build_model(
            unconditional=False,
            cond_stage_key="0094",
            normalize_latent_per_channel=True,
        )
        batch = {
            "hmi": torch.tensor([[[[0.0, 2.0]]]]),
            "0094": torch.tensor([[[[10.0, 14.0]]]]),
        }

        model.initialize_latent_stats_from_dataloader([batch])
        z, c = model.get_input(batch, "hmi")

        self.assertTrue(
            torch.allclose(model.latent_mean.flatten(), torch.tensor([1.0]))
        )
        self.assertTrue(
            torch.allclose(model.latent_std.flatten(), torch.tensor([1.0]))
        )
        self.assertTrue(
            torch.allclose(model.cond_latent_mean.flatten(), torch.tensor([12.0]))
        )
        self.assertTrue(
            torch.allclose(model.cond_latent_std.flatten(), torch.tensor([2.0]))
        )
        self.assertTrue(torch.allclose(z.flatten(), torch.tensor([-1.0, 1.0])))
        self.assertTrue(torch.allclose(c.flatten(), torch.tensor([-1.0, 1.0])))

    def test_missing_stats_fail_loudly_outside_training_calibration(self):
        model = build_model(
            unconditional=True,
            normalize_latent_per_channel=True,
        ).eval()

        with self.assertRaisesRegex(RuntimeError, "stats 尚未初始化"):
            model.get_input({"hmi": torch.zeros(1, 1, 2, 2)}, "hmi")


if __name__ == "__main__":
    unittest.main()
