"""Focused contract tests for the A1 implementation.

Run with:
    pytest -q compare/flare/yi2023_dqn/test_module.py
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
from torch import nn

from .architecture import (
    YiDenseNet,
    cumulative_inconsistency,
    cumulative_targets,
    decode_cumulative_actions,
)
from .module import Yi2023DQN
from .replay import ReplayBuffer, ReplayTransition


class TinyQNetwork(nn.Module):
    """Fast replacement used only to exercise the Lightning/DQN state machine."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(1, 6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(inputs).flatten(1)).reshape(-1, 3, 2)


def test_exact_dense_shapes_and_three_heads() -> None:
    model = YiDenseNet().eval()
    with torch.no_grad():
        features, shapes = model.forward_features(
            torch.zeros(1, 1, 512, 512), return_stage_shapes=True
        )
        values = model(torch.zeros(1, 1, 512, 512))
    assert shapes == (
        (26, 256, 256),
        (65, 128, 128),
        (104, 64, 64),
        (143, 32, 32),
        (182, 16, 16),
        (221, 8, 8),
    )
    assert features.shape == (1, 3536)
    assert values.shape == (1, 3, 2)


def test_targets_highest_positive_decode_and_inconsistency() -> None:
    labels = torch.tensor([0, 1, 2, 3])
    assert cumulative_targets(labels).tolist() == [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]
    actions = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    )
    assert decode_cumulative_actions(actions).tolist() == [0, 1, 2, 3, 3]
    assert cumulative_inconsistency(actions).tolist() == [
        False,
        False,
        True,
        True,
        False,
    ]


def test_replay_and_rng_round_trip() -> None:
    replay = ReplayBuffer(capacity=3, seed=7)
    for index in range(3):
        replay.append(
            ReplayTransition(
                state=torch.full((1, 2, 2), index, dtype=torch.float16),
                actions=torch.tensor([0, 1, 0]),
                rewards=torch.tensor([1.0, 8.0, 1.0]),
                next_state=torch.full((1, 2, 2), index + 1, dtype=torch.float16),
                done=index == 2,
            )
        )
    saved = copy.deepcopy(replay.state_dict())
    expected = replay.sample(2, torch.device("cpu"))
    restored = ReplayBuffer(capacity=3, seed=999)
    restored.load_state_dict(saved)
    actual = restored.sample(2, torch.device("cpu"))
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))


def test_dataloader_order_episode_and_checkpoint_state() -> None:
    model = Yi2023DQN(
        replay_capacity=8,
        replay_batch_size=1,
        replay_warmup=1,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay_steps=10,
    )
    model.online_network = TinyQNetwork()
    model.target_network = TinyQNetwork().requires_grad_(False).eval()
    model._synchronize_target()
    states = torch.arange(3.0).reshape(3, 1, 1, 1)
    labels = torch.tensor([0, 1, 3])
    model._observe_batch(states, labels)
    assert len(model.replay_buffer) == 2
    assert model._pending_transition is not None
    model._finish_episode()
    assert len(model.replay_buffer) == 3
    assert model.replay_buffer._storage[-1].done is True

    checkpoint: dict = {}
    model.on_save_checkpoint(checkpoint)
    restored = Yi2023DQN(
        replay_capacity=8,
        replay_batch_size=1,
        replay_warmup=1,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay_steps=10,
    )
    restored.on_load_checkpoint(checkpoint)
    assert len(restored.replay_buffer) == 3
    assert restored._environment_steps == 3
    assert restored.epsilon == model.epsilon
    restored.online_network = copy.deepcopy(model.online_network)
    probe = torch.zeros(5, 1, 2, 2)
    original_actions, _ = model._epsilon_greedy_actions(probe)
    restored_actions, _ = restored._epsilon_greedy_actions(probe)
    assert torch.equal(original_actions, restored_actions)


def test_validation_boundary_is_checkpoint_ready() -> None:
    model = Yi2023DQN(
        replay_capacity=8,
        replay_batch_size=1,
        replay_warmup=1,
        target_sync_epochs=2,
    )
    model.online_network = TinyQNetwork()
    model.target_network = TinyQNetwork().requires_grad_(False).eval()
    with torch.no_grad():
        model.online_network.head.weight.fill_(2.0)
        model.target_network.head.weight.zero_()
    model._trainer = SimpleNamespace(current_epoch=1)
    model.on_train_epoch_start()
    model._observe_batch(
        torch.arange(2.0).reshape(2, 1, 1, 1), torch.tensor([0, 3])
    )

    # Validation end is where ModelCheckpoint normally saves.  The preceding
    # validation-start hook must therefore finish the episode and synchronize
    # the target before serialization can occur.
    model.on_validation_epoch_start()
    assert model._pending_transition is None
    assert len(model.replay_buffer) == 2
    assert model.replay_buffer._storage[-1].done is True
    assert model._target_synced_this_epoch is True
    assert torch.equal(
        model.online_network.head.weight, model.target_network.head.weight
    )


def test_supervised_control_uses_same_output_contract() -> None:
    model = Yi2023DQN(training_mode="supervised")
    values = torch.randn(4, 3, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    loss, per_head = model._supervised_loss(values, labels)
    loss.backward()
    assert loss.ndim == 0
    assert per_head.shape == (3,)
    assert values.grad is not None
