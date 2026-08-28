"""Lightning implementation of the SolarCHIP Yi 2023 comparison."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

from .architecture import (
    CLASS_NAMES,
    HEAD_NAMES,
    YiDenseNet,
    cumulative_inconsistency,
    cumulative_targets,
    decode_cumulative_actions,
)
from .metrics import YiEvaluationMetrics
from .replay import PendingTransition, ReplayBuffer, ReplayTransition


IMPLEMENTATION_VERSION = 1


class Yi2023DQN(pl.LightningModule):
    """Vanilla DQN with three cumulative heads, or its supervised control.

    The primary ``training_mode='dqn'`` path uses manual Lightning
    optimization, experience replay, epsilon-greedy actions and a frozen target
    network.  In the requested SolarCHIP adaptation, samples adjacent in the
    existing DataLoader order form transitions (also across batch boundaries),
    and the last sample of each epoch is terminal.

    ``training_mode='supervised'`` trains the identical shared trunk and three
    heads with mean binary cross entropy.  It is an engineering control, not a
    result reported by Yi et al.
    """

    def __init__(
        self,
        training_mode: Literal["dqn", "supervised"] = "dqn",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20_000,
        replay_capacity: int = 128,
        replay_batch_size: int = 8,
        replay_warmup: int = 32,
        gradient_updates_per_batch: int = 1,
        checkpoint_replay: bool = True,
        rmsprop_alpha: float = 0.99,
        rmsprop_epsilon: float = 1e-8,
        rmsprop_momentum: float = 0.0,
        target_sync_epochs: int = 2,
        reward_tp: float = 8.0,
        reward_fp: float = -64.0,
        reward_fn: float = -8.0,
        reward_tn: float = 1.0,
        max_epochs: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if training_mode not in {"dqn", "supervised"}:
            raise ValueError("training_mode must be 'dqn' or 'supervised'")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0,1]")
        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")
        if epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        if replay_batch_size <= 0 or replay_warmup < replay_batch_size:
            raise ValueError("replay_warmup must be >= replay_batch_size > 0")
        if replay_warmup > replay_capacity:
            raise ValueError("replay_warmup cannot exceed replay_capacity")
        if gradient_updates_per_batch <= 0:
            raise ValueError("gradient_updates_per_batch must be positive")
        if target_sync_epochs <= 0:
            raise ValueError("target_sync_epochs must be positive")

        self.training_mode = training_mode
        self.online_network = YiDenseNet(num_heads=3)
        self.target_network: YiDenseNet | None
        if training_mode == "dqn":
            self.target_network = YiDenseNet(num_heads=3)
            self._synchronize_target()
        else:
            self.target_network = None

        self.replay_buffer = ReplayBuffer(replay_capacity, seed + 1)
        self._exploration_generator = torch.Generator(device="cpu")
        self._exploration_generator.manual_seed(int(seed))
        self._environment_steps = 0
        self._pending_transition: PendingTransition | None = None
        # ModelCheckpoint normally saves at validation end, which precedes
        # LightningModule.on_train_epoch_end.  Keep the environment boundary
        # explicit so an epoch-end checkpoint never carries a transition into
        # the following epoch and includes any target-network synchronization
        # due at that boundary.
        self._training_epoch_active = False
        self._epoch_boundary_finalized = False
        self._target_synced_this_epoch = False
        self.metric_modules = nn.ModuleDict(
            {
                f"{split}_split": YiEvaluationMetrics()
                for split in ("train", "val", "test")
            }
        )
        self.automatic_optimization = training_mode != "dqn"
        self.save_hyperparameters()

    @property
    def epsilon(self) -> float:
        progress = min(
            float(self._environment_steps) / float(self.hparams.epsilon_decay_steps),
            1.0,
        )
        return float(
            self.hparams.epsilon_start
            + progress * (self.hparams.epsilon_end - self.hparams.epsilon_start)
        )

    @staticmethod
    def prepare_hmi(hmi: torch.Tensor) -> torch.Tensor:
        """Validate the existing loader output and area-resize inside the model."""

        if hmi.ndim == 3:
            hmi = hmi.unsqueeze(1)
        if hmi.ndim != 4 or hmi.shape[1] != 1:
            raise ValueError(
                "batch['hmi'] must be [B,1,H,W] (or [B,H,W]), got "
                f"{tuple(hmi.shape)}"
            )
        hmi = hmi.float()
        if hmi.shape[-2:] != (512, 512):
            hmi = F.interpolate(hmi, size=(512, 512), mode="area")
        return hmi

    @staticmethod
    def _labels(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if "label" not in batch:
            raise KeyError("Batch must contain grouped flare 'label'")
        labels = batch["label"].long().reshape(-1)
        cumulative_targets(labels)
        return labels

    def forward(self, hmi: torch.Tensor) -> torch.Tensor:
        """Return C+/M+/X+ action values shaped ``[B,3,2]``."""

        return self.online_network(self.prepare_hmi(hmi))

    @staticmethod
    def decode(values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (3, 2):
            raise ValueError(f"Expected values [B,3,2], got {tuple(values.shape)}")
        return decode_cumulative_actions(values.argmax(dim=-1))

    def _synchronize_target(self) -> None:
        if self.target_network is None:
            return
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.requires_grad_(False)
        self.target_network.eval()

    def _reward(
        self, actions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if actions.shape != targets.shape:
            raise ValueError("actions and cumulative targets must have equal shapes")
        rewards = torch.full(
            actions.shape,
            float(self.hparams.reward_tn),
            device=actions.device,
            dtype=torch.float32,
        )
        rewards[(actions == 1) & (targets == 1)] = float(self.hparams.reward_tp)
        rewards[(actions == 1) & (targets == 0)] = float(self.hparams.reward_fp)
        rewards[(actions == 0) & (targets == 1)] = float(self.hparams.reward_fn)
        return rewards

    @staticmethod
    def _cpu_image(image: torch.Tensor) -> torch.Tensor:
        # Explicit storage approximation: sampled tensors return to float32.
        return image.detach().to(device="cpu", dtype=torch.float16).contiguous()

    @staticmethod
    def _cpu_vector(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return values.detach().to(device="cpu", dtype=dtype).contiguous()

    @torch.no_grad()
    def _epsilon_greedy_actions(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.online_network.training
        self.online_network.eval()
        values = self.online_network(states)
        self.online_network.train(was_training)
        greedy = values.argmax(dim=-1)
        explore = torch.rand(
            greedy.shape,
            generator=self._exploration_generator,
            device="cpu",
        ).lt(self.epsilon)
        random_actions = torch.randint(
            0,
            2,
            greedy.shape,
            generator=self._exploration_generator,
            device="cpu",
        )
        actions = torch.where(
            explore.to(greedy.device),
            random_actions.to(greedy.device),
            greedy,
        )
        return actions, values

    def _append_transition(
        self,
        pending: PendingTransition,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.replay_buffer.append(
            ReplayTransition(
                state=pending.state,
                actions=pending.actions,
                rewards=pending.rewards,
                next_state=self._cpu_image(next_state),
                done=done,
            )
        )

    @torch.no_grad()
    def _observe_batch(
        self, states: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, acting_values = self._epsilon_greedy_actions(states)
        rewards = self._reward(actions, cumulative_targets(labels))

        if self._pending_transition is not None:
            self._append_transition(self._pending_transition, states[0], done=False)
        for index in range(states.shape[0] - 1):
            pending = PendingTransition(
                state=self._cpu_image(states[index]),
                actions=self._cpu_vector(actions[index], torch.long),
                rewards=self._cpu_vector(rewards[index], torch.float32),
            )
            self._append_transition(pending, states[index + 1], done=False)

        last = states.shape[0] - 1
        self._pending_transition = PendingTransition(
            state=self._cpu_image(states[last]),
            actions=self._cpu_vector(actions[last], torch.long),
            rewards=self._cpu_vector(rewards[last], torch.float32),
        )
        self._environment_steps += int(states.shape[0])
        return acting_values, actions, rewards

    def _finish_episode(self) -> None:
        if self._pending_transition is None:
            return
        terminal = torch.zeros_like(self._pending_transition.state)
        self._append_transition(self._pending_transition, terminal, done=True)
        self._pending_transition = None

    def _dqn_replay_loss(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.target_network is None:
            raise RuntimeError("DQN replay loss is unavailable in supervised mode")
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            int(self.hparams.replay_batch_size), self.device
        )
        values = self.online_network(states)
        chosen_values = values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            self.target_network.eval()
            next_values = self.target_network(next_states).amax(dim=-1)
            td_targets = rewards + float(self.hparams.gamma) * (
                ~dones
            ).float().unsqueeze(1) * next_values
        per_head_loss = (chosen_values - td_targets).square().mean(dim=0)
        loss = per_head_loss.mean()
        mean_absolute_td_error = (chosen_values - td_targets).abs().mean()
        return loss, per_head_loss, mean_absolute_td_error

    @staticmethod
    def _supervised_loss(
        values: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = cumulative_targets(labels)
        per_head = torch.stack(
            [
                F.cross_entropy(values[:, index], targets[:, index])
                for index in range(3)
            ]
        )
        return per_head.mean(), per_head

    def training_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        if "hmi" not in batch:
            raise KeyError("Batch must contain 'hmi'")
        states = self.prepare_hmi(batch["hmi"])
        labels = self._labels(batch)
        if states.shape[0] != labels.shape[0] or states.shape[0] == 0:
            raise ValueError("HMI and label batch sizes must match and be non-empty")

        if self.training_mode == "supervised":
            values = self.online_network(states)
            loss, per_head = self._supervised_loss(values, labels)
            self.metric_modules["train_split"].update(values.detach(), labels)
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=labels.shape[0],
            )
            for index, name in enumerate(HEAD_NAMES):
                self.log(
                    f"train_{name}_loss",
                    per_head[index],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=labels.shape[0],
                )
            return loss

        acting_values, _, rewards = self._observe_batch(states, labels)
        self.metric_modules["train_split"].update(acting_values, labels)
        self.log(
            "train_mean_reward",
            rewards.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=labels.shape[0],
        )
        self.log("train_epsilon", self.epsilon, on_step=True, on_epoch=False)
        self.log(
            "train_replay_size",
            float(len(self.replay_buffer)),
            on_step=True,
            on_epoch=False,
        )
        if len(self.replay_buffer) < int(self.hparams.replay_warmup):
            return states.new_zeros(())

        optimizer = self.optimizers()
        losses: list[torch.Tensor] = []
        for _ in range(int(self.hparams.gradient_updates_per_batch)):
            optimizer.zero_grad()
            loss, per_head, td_error = self._dqn_replay_loss()
            self.manual_backward(loss)
            optimizer.step()
            losses.append(loss.detach())
            self.log(
                "train_td_error",
                td_error.detach(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=int(self.hparams.replay_batch_size),
            )
            for index, name in enumerate(HEAD_NAMES):
                self.log(
                    f"train_{name}_td_loss",
                    per_head[index].detach(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=int(self.hparams.replay_batch_size),
                )
        mean_loss = torch.stack(losses).mean()
        self.log(
            "train_loss",
            mean_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=int(self.hparams.replay_batch_size),
        )
        return mean_loss

    def _evaluation_step(
        self,
        batch: Mapping[str, torch.Tensor],
        split: Literal["val", "test"],
    ) -> torch.Tensor:
        if "hmi" not in batch:
            raise KeyError("Batch must contain 'hmi'")
        labels = self._labels(batch)
        values = self.online_network(self.prepare_hmi(batch["hmi"]))
        predictions, actions = self.metric_modules[f"{split}_split"].update(
            values, labels
        )
        if self.training_mode == "supervised":
            loss, _ = self._supervised_loss(values, labels)
            self.log(
                f"{split}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=labels.shape[0],
            )
        else:
            rewards = self._reward(actions, cumulative_targets(labels))
            self.log(
                f"{split}_mean_reward",
                rewards.mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=labels.shape[0],
            )
        return predictions

    def validation_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        return self._evaluation_step(batch, "val")

    def test_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        return self._evaluation_step(batch, "test")

    def predict_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        del batch_idx, dataloader_idx
        if "hmi" not in batch:
            raise KeyError("Batch must contain 'hmi'")
        values = self.online_network(self.prepare_hmi(batch["hmi"]))
        actions = values.argmax(dim=-1)
        output = {
            "q_values" if self.training_mode == "dqn" else "logits": values,
            "cumulative_actions": actions,
            "prediction": decode_cumulative_actions(actions),
            "inconsistent_heads": cumulative_inconsistency(actions),
        }
        if "date_id" in batch:
            output["date_id"] = batch["date_id"]
        return output

    def _log_and_reset_metrics(self, split: str) -> None:
        self.log_dict(
            self.metric_modules[f"{split}_split"].compute(split),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.metric_modules[f"{split}_split"].reset()

    def on_train_epoch_start(self) -> None:
        self._training_epoch_active = True
        self._epoch_boundary_finalized = False
        self._target_synced_this_epoch = False
        if self.target_network is not None:
            self.target_network.eval()

    def _finalize_training_epoch_boundary(self) -> None:
        if (
            self.training_mode != "dqn"
            or not self._training_epoch_active
            or self._epoch_boundary_finalized
        ):
            return
        self._finish_episode()
        if (self.current_epoch + 1) % int(self.hparams.target_sync_epochs) == 0:
            self._synchronize_target()
            self._target_synced_this_epoch = True
        self._epoch_boundary_finalized = True
        self._training_epoch_active = False

    def on_validation_epoch_start(self) -> None:
        # With the shipped epoch-level validation schedule, this is the last
        # hook before the checkpoint callback may serialize the model.
        self._finalize_training_epoch_boundary()

    def on_train_epoch_end(self) -> None:
        if self.training_mode == "dqn":
            # Fallback for runs without validation; otherwise this is a no-op.
            self._finalize_training_epoch_boundary()
            if self._target_synced_this_epoch:
                self.log(
                    "train_target_sync", 1.0, on_step=False, on_epoch=True
                )
        self._log_and_reset_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_and_reset_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_and_reset_metrics("test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.training_mode == "dqn":
            return torch.optim.RMSprop(
                self.online_network.parameters(),
                lr=float(self.hparams.learning_rate),
                alpha=float(self.hparams.rmsprop_alpha),
                eps=float(self.hparams.rmsprop_epsilon),
                momentum=float(self.hparams.rmsprop_momentum),
                weight_decay=float(self.hparams.weight_decay),
            )
        return torch.optim.Adam(
            self.online_network.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )

    @staticmethod
    def _pending_state_dict(
        pending: PendingTransition | None,
    ) -> dict[str, torch.Tensor] | None:
        if pending is None:
            return None
        return {
            "state": pending.state,
            "actions": pending.actions,
            "rewards": pending.rewards,
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["yi2023_metadata"] = {
            "implementation_version": IMPLEMENTATION_VERSION,
            "training_mode": self.training_mode,
            "class_names": list(CLASS_NAMES),
            "head_names": list(HEAD_NAMES),
            "transition_order": "current_dataloader_order_epoch_terminal",
        }
        if self.training_mode != "dqn":
            return
        dqn_state: dict[str, Any] = {
            "environment_steps": self._environment_steps,
            "exploration_generator_state": self._exploration_generator.get_state(),
            "pending_transition": self._pending_state_dict(self._pending_transition),
        }
        if bool(self.hparams.checkpoint_replay):
            dqn_state["replay_buffer"] = self.replay_buffer.state_dict()
        checkpoint["yi2023_dqn_state"] = dqn_state

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        metadata = checkpoint.get("yi2023_metadata")
        if isinstance(metadata, Mapping):
            saved_mode = metadata.get("training_mode")
            if saved_mode != self.training_mode:
                raise ValueError(
                    "Checkpoint Yi training_mode does not match this model: "
                    f"{saved_mode!r} != {self.training_mode!r}"
                )
        if self.training_mode != "dqn":
            return
        dqn_state = checkpoint.get("yi2023_dqn_state")
        if not isinstance(dqn_state, Mapping):
            # Legacy/weights-only checkpoints intentionally start fresh replay.
            return
        self._environment_steps = int(dqn_state.get("environment_steps", 0))
        exploration_state = dqn_state.get("exploration_generator_state")
        if torch.is_tensor(exploration_state):
            self._exploration_generator.set_state(exploration_state.cpu())
        raw_pending = dqn_state.get("pending_transition")
        if isinstance(raw_pending, Mapping):
            self._pending_transition = PendingTransition(
                state=raw_pending["state"].cpu(),
                actions=raw_pending["actions"].cpu().long(),
                rewards=raw_pending["rewards"].cpu().float(),
            )
        else:
            self._pending_transition = None
        replay_state = dqn_state.get("replay_buffer")
        if isinstance(replay_state, Mapping):
            self.replay_buffer.load_state_dict(replay_state)


class Yi2023Supervised(Yi2023DQN):
    """Convenience import target for the supervised architecture control."""

    def __init__(self, **kwargs: Any) -> None:
        if "training_mode" in kwargs and kwargs["training_mode"] != "supervised":
            raise ValueError("Yi2023Supervised always uses supervised mode")
        kwargs.pop("training_mode", None)
        super().__init__(training_mode="supervised", **kwargs)


__all__ = ["Yi2023DQN", "Yi2023Supervised", "YiDenseNet"]
