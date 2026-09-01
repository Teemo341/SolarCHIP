"""Focused CPU contract tests for the DeepSWM Lightning module.

These tests do not launch a distributed process group. Instead, DDP tests
inject other ranks through mocked ``all_reduce`` and ``all_gather_object``.

Run with:
    pytest -q compare/flare/deepswm/test_module.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import module as deepswm_module
from .module import DeepSWM, SplitAccumulator, inverse_frequency_weights


class TinyNetwork(nn.Module):
    """Cheap network replacement; these tests exercise only module contracts."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.projection = nn.Linear(1, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = inputs.mean(dim=tuple(range(1, inputs.ndim)), keepdim=False)
        return self.projection(pooled[:, None])


def _make_module(monkeypatch, **kwargs) -> DeepSWM:
    monkeypatch.setattr(deepswm_module, "HMIOnlyDeepSWM", TinyNetwork)
    return DeepSWM(
        train_class_counts=[1000, 100, 10, 1],
        gmgs_weight=0.0,
        bss_weight=0.0,
        **kwargs,
    )


def test_unweighted_ce_is_the_plain_batch_mean(monkeypatch) -> None:
    model = _make_module(monkeypatch, class_weight_mode="none")
    logits = torch.tensor(
        [
            [8.0, 0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, -2.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 3])

    components = model.loss_components(logits, labels)
    expected = F.cross_entropy(logits, labels, reduction="mean")
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    legacy_weights = inverse_frequency_weights(
        torch.tensor([1000.0, 100.0, 10.0, 1.0])
    )[labels]
    weighted = (per_sample * legacy_weights).sum() / legacy_weights.sum()

    torch.testing.assert_close(model.class_weights, torch.ones(4))
    torch.testing.assert_close(
        model.training_class_probabilities,
        torch.tensor([1000.0, 100.0, 10.0, 1.0]) / 1111.0,
    )
    torch.testing.assert_close(components["ce"], expected)
    # Protect against an implementation that accepts the new option but still
    # silently follows the old inverse-frequency path.
    assert not torch.isclose(components["ce"], weighted)


def test_synchronized_sums_sufficient_statistics_without_mutating_local_rank(
    monkeypatch,
) -> None:
    accumulator = SplitAccumulator()
    local_confusion = torch.tensor(
        [
            [2, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    remote_confusion = torch.tensor(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.long,
    )
    accumulator.confusion.copy_(local_confusion)
    accumulator.mplus_brier_sum.fill_(0.5)
    accumulator.sample_count.fill_(5)

    remote_values = iter(
        (
            remote_confusion,
            torch.tensor(0.25, dtype=torch.float64),
            torch.tensor(4, dtype=torch.long),
        )
    )
    calls: list[torch.Tensor] = []

    def fake_all_reduce(tensor: torch.Tensor, op=None) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        calls.append(tensor)
        tensor.add_(next(remote_values).to(device=tensor.device, dtype=tensor.dtype))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    confusion, brier_sum, sample_count = accumulator.synchronized()

    assert len(calls) == 3
    torch.testing.assert_close(confusion, local_confusion + remote_confusion)
    torch.testing.assert_close(brier_sum, torch.tensor(0.75, dtype=torch.float64))
    torch.testing.assert_close(sample_count, torch.tensor(9, dtype=torch.long))
    # Synchronizing for metrics must not contaminate the rank-local running
    # buffers or make a second epoch double-count other ranks.
    torch.testing.assert_close(accumulator.confusion, local_confusion)
    torch.testing.assert_close(
        accumulator.mplus_brier_sum, torch.tensor(0.5, dtype=torch.float64)
    )
    torch.testing.assert_close(
        accumulator.sample_count, torch.tensor(5, dtype=torch.long)
    )


def test_unique_date_records_remove_sampler_padding() -> None:
    accumulator = SplitAccumulator()
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 2, 0])
    # date 100 is the kind of repeated item DistributedSampler appends when
    # the validation length is not divisible by world size.
    accumulator.update(logits, labels, torch.tensor([100, 101, 100]))

    confusion, brier_sum, sample_count = accumulator.synchronized()

    expected_confusion = torch.zeros(4, 4, dtype=torch.long)
    expected_confusion[0, 0] = 1
    expected_confusion[2, 2] = 1
    probabilities = logits.softmax(dim=-1)
    expected_brier = probabilities[0, 2:].sum().double().square()
    expected_brier += (probabilities[1, 2:].sum().double() - 1.0).square()
    torch.testing.assert_close(confusion, expected_confusion)
    torch.testing.assert_close(brier_sum, expected_brier)
    torch.testing.assert_close(sample_count, torch.tensor(2, dtype=torch.long))


def test_unique_date_records_are_deduplicated_across_ranks(monkeypatch) -> None:
    accumulator = SplitAccumulator()
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 2])
    accumulator.update(logits, labels, torch.tensor([100, 101]))
    local_records = list(accumulator.records)
    remote_records = [
        local_records[0],
        (102, 3, (0.0, 0.0, 0.0, 1.0)),
    ]

    def fake_all_gather_object(output, local) -> None:
        output[0] = list(local)
        output[1] = remote_records

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", fake_all_gather_object
    )

    confusion, _, sample_count = accumulator.synchronized()

    expected_confusion = torch.zeros(4, 4, dtype=torch.long)
    expected_confusion[0, 0] = 1
    expected_confusion[2, 2] = 1
    expected_confusion[3, 3] = 1
    torch.testing.assert_close(confusion, expected_confusion)
    torch.testing.assert_close(sample_count, torch.tensor(3, dtype=torch.long))


def test_epoch_metrics_are_computed_from_summed_statistics(monkeypatch) -> None:
    model = _make_module(monkeypatch, class_weight_mode="none")
    confusion = torch.tensor(
        [
            [2, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 2, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.long,
    )

    metrics = model._epoch_metric_values(
        confusion,
        torch.tensor(0.75, dtype=torch.float64),
        torch.tensor(9, dtype=torch.long),
    )

    torch.testing.assert_close(
        metrics["accuracy"], torch.tensor(2.0 / 3.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        metrics["macro_f1"], torch.tensor(17.0 / 24.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        metrics["tss_mplus"], torch.tensor(0.55, dtype=torch.float64)
    )
    # Train climatology above has M+X probability 11/1111, so calculate the
    # exact BSS reference from the model's persistent train-only statistics.
    climatology = model.training_class_probabilities[2:].sum().to(torch.float64)
    expected_bss = 1.0 - (0.75 / 9.0) / (climatology * (1.0 - climatology))
    torch.testing.assert_close(metrics["bss_mplus"], expected_bss)
