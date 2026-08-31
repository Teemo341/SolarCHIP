"""Checkpointable replay storage for the Yi 2023 DQN adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ReplayTransition:
    state: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_state: torch.Tensor
    done: bool


@dataclass
class PendingTransition:
    state: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    """Fixed-size CPU ring buffer with its own checkpointable RNG."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        self.capacity = int(capacity)
        self._storage: list[ReplayTransition] = []
        self._next_index = 0
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self._storage)

    def append(self, transition: ReplayTransition) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._storage:
            raise RuntimeError("Cannot sample an empty replay buffer")
        if batch_size <= 0:
            raise ValueError("replay_batch_size must be positive")
        indices = torch.randint(
            len(self._storage),
            (batch_size,),
            generator=self._generator,
        ).tolist()
        sampled = [self._storage[index] for index in indices]
        states = torch.stack([item.state for item in sampled]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        actions = torch.stack([item.actions for item in sampled]).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        rewards = torch.stack([item.rewards for item in sampled]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        next_states = torch.stack([item.next_state for item in sampled]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        dones = torch.tensor(
            [item.done for item in sampled], device=device, dtype=torch.bool
        )
        return states, actions, rewards, next_states, dones

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "next_index": self._next_index,
            "generator_state": self._generator.get_state(),
            "transitions": [
                {
                    "state": item.state,
                    "actions": item.actions,
                    "rewards": item.rewards,
                    "next_state": item.next_state,
                    "done": item.done,
                }
                for item in self._storage
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_capacity = int(state["capacity"])
        if saved_capacity != self.capacity:
            raise ValueError(
                "Checkpoint replay_capacity does not match the model: "
                f"{saved_capacity} != {self.capacity}"
            )
        raw_transitions = state.get("transitions", [])
        if not isinstance(raw_transitions, list) or len(raw_transitions) > self.capacity:
            raise ValueError("Checkpoint contains an invalid replay transition list")
        restored: list[ReplayTransition] = []
        for raw in raw_transitions:
            if not isinstance(raw, Mapping):
                raise ValueError("Each saved replay transition must be a mapping")
            restored.append(
                ReplayTransition(
                    state=raw["state"].cpu(),
                    actions=raw["actions"].cpu().long(),
                    rewards=raw["rewards"].cpu().float(),
                    next_state=raw["next_state"].cpu(),
                    done=bool(raw["done"]),
                )
            )
        self._storage = restored
        self._next_index = int(state["next_index"])
        if not 0 <= self._next_index < self.capacity:
            raise ValueError("Checkpoint contains an invalid replay write index")
        generator_state = state.get("generator_state")
        if not torch.is_tensor(generator_state):
            raise ValueError("Checkpoint replay RNG state is missing or invalid")
        self._generator.set_state(generator_state.cpu())


__all__ = ["PendingTransition", "ReplayBuffer", "ReplayTransition"]
