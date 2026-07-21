import unittest

import torch
import torch.nn as nn

from downstream.ldm.SolarControl import SolarControl


class TinyControlledUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(2, 2))
        self.input_blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.middle_block = nn.Linear(2, 2)
        self.output_blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.out = nn.Linear(2, 2)

    def forward(self, x, timesteps=None, context=None, control=None, **kwargs):
        backbone_gain = 1.0 + sum(
            parameter.square().mean() for parameter in self.parameters()
        )
        control_residual = 0.0 if control is None else sum(control)
        return (x + control_residual) * backbone_gain


class TinyControlNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(2, 2))
        self.input_blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.zero_convs = nn.ModuleList([nn.Linear(2, 2)])
        self.middle_block = nn.Linear(2, 2)
        self.middle_block_out = nn.Linear(2, 2)
        self.input_hint_block = nn.Sequential(
            nn.Linear(2, 2),
            nn.Linear(2, 2),
        )

    def forward(self, x, hint, timesteps=None, context=None, **kwargs):
        control_gain = 1.0 + sum(
            parameter.mean() for parameter in self.parameters()
        )
        residual = hint * control_gain
        return [residual, residual]


def build_control(sd_locked, **model_kwargs):
    constructor_locked = (
        sd_locked and model_kwargs.get("sd_backbone_ckpt") is not None
    )
    kwargs = dict(
        control_stage_config={
            "target": "tests.test_solarcontrol_optimizer.TinyControlNet",
            "params": {},
        },
        solarchip_config={
            "target": "tests.test_solarldm_unconditional.TinySolarWrapper",
            "params": {"modal_list": ["hmi", "0094"]},
        },
        unet_config={
            "target": "tests.test_solarcontrol_optimizer.TinyControlledUNet",
            "params": {},
        },
        first_stage_key="hmi",
        cond_stage_key="0094",
        learning_rate=1.0e-2,
        image_size=4,
        channels=1,
        timesteps=10,
        linear_start=1.0e-4,
        linear_end=2.0e-2,
        use_ema=False,
        sd_locked=constructor_locked,
    )
    kwargs.update(model_kwargs)
    model = SolarControl(**kwargs)
    if sd_locked and not constructor_locked:
        # Optimizer unit tests isolate the locking mechanism. Direct locked
        # construction is covered with a real temporary SD checkpoint below.
        model.sd_locked = True
        model._set_sd_backbone_trainable(False)
    return model


def optimizer_parameter_ids(optimizer):
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


class SolarControlOptimizerTest(unittest.TestCase):
    def test_locked_optimizer_contains_only_controlnet(self):
        model = build_control(sd_locked=True)
        backbone_parameters = list(model.model.parameters())
        control_parameters = list(model.control_model.parameters())

        model.train()
        optimizer = model.configure_optimizers()

        self.assertTrue(all(not p.requires_grad for p in backbone_parameters))
        self.assertTrue(all(p.requires_grad for p in control_parameters))
        self.assertFalse(model.model.training)
        self.assertTrue(model.control_model.training)
        self.assertEqual(
            optimizer_parameter_ids(optimizer),
            {id(parameter) for parameter in control_parameters},
        )

    def test_locked_step_updates_controlnet_but_not_backbone(self):
        model = build_control(sd_locked=True)
        optimizer = model.configure_optimizers()
        backbone_before = [p.detach().clone() for p in model.model.parameters()]
        control_before = [
            p.detach().clone() for p in model.control_model.parameters()
        ]

        x = torch.ones(2, 1, 2, 2)
        hint = torch.ones_like(x)
        timesteps = torch.zeros(2, dtype=torch.long)
        loss = model.apply_model(
            x,
            timesteps,
            {"c_concat": [hint]},
        ).sum()
        loss.backward()

        self.assertTrue(
            all(parameter.grad is None for parameter in model.model.parameters())
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.control_model.parameters()
            )
        )
        optimizer.step()

        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(backbone_before, model.model.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    control_before, model.control_model.parameters()
                )
            )
        )

    def test_unlocked_optimizer_contains_controlnet_and_full_backbone(self):
        model = build_control(sd_locked=False)
        model.train()
        optimizer = model.configure_optimizers()
        backbone_parameters = list(model.model.parameters())
        control_parameters = list(model.control_model.parameters())

        self.assertTrue(all(p.requires_grad for p in backbone_parameters))
        self.assertTrue(model.model.training)
        self.assertEqual(
            optimizer_parameter_ids(optimizer),
            {
                id(parameter)
                for parameter in control_parameters + backbone_parameters
            },
        )


if __name__ == "__main__":
    unittest.main()
