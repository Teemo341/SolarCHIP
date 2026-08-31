"""Shared confusion-matrix reductions and flare skill metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .class_groups import BASE_CLASS_SYMBOLS, normalize_class_groups


def _group_mapping(
    class_groups: Sequence[str],
    raw_bucket: Mapping[str, int],
    description: str,
) -> tuple[int, ...]:
    mapping: list[int] = []
    for group in class_groups:
        buckets = {raw_bucket[symbol] for symbol in group}
        if len(buckets) != 1:
            raise ValueError(
                f"class group {group!r} crosses the {description} boundary; "
                "the requested metric cannot be recovered from grouped predictions"
            )
        mapping.append(next(iter(buckets)))
    return tuple(mapping)


def class_reduction_mappings(
    class_groups: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    """Map grouped classes to the paper's 0/C/M, C+, and M+ tasks.

    A grouped output cannot be reduced across a boundary that it already
    merges.  For example, ``0ABC`` crosses the C+ boundary, so C+ predictions
    cannot be recovered from that classifier's logits.
    """

    groups = normalize_class_groups(class_groups)
    ranks = {symbol: index for index, symbol in enumerate(BASE_CLASS_SYMBOLS)}
    return {
        "overall": _group_mapping(
            groups,
            {
                symbol: 0 if rank < ranks["C"] else 1 if rank < ranks["M"] else 2
                for symbol, rank in ranks.items()
            },
            "0/C/M three-class",
        ),
        "c_plus": _group_mapping(
            groups,
            {symbol: int(rank >= ranks["C"]) for symbol, rank in ranks.items()},
            "C+",
        ),
        "m_plus": _group_mapping(
            groups,
            {symbol: int(rank >= ranks["M"]) for symbol, rank in ranks.items()},
            "M+",
        ),
    }


def threshold_reduction_mappings(
    class_groups: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    """Return only the C+ and M+ mappings used during training."""

    return {
        threshold: threshold_reduction_mapping(class_groups, threshold)
        for threshold in ("c_plus", "m_plus")
    }


def threshold_reduction_mapping(
    class_groups: Sequence[str],
    threshold: str,
) -> tuple[int, ...]:
    """Return one binary threshold mapping for grouped output classes."""

    groups = normalize_class_groups(class_groups)
    ranks = {symbol: index for index, symbol in enumerate(BASE_CLASS_SYMBOLS)}
    definitions = {
        "c_plus": ("C", "C+"),
        "m_plus": ("M", "M+"),
    }
    if threshold not in definitions:
        raise ValueError(
            f"threshold must be one of {list(definitions)}, got {threshold!r}"
        )
    boundary_symbol, description = definitions[threshold]
    return _group_mapping(
        groups,
        {
            symbol: int(rank >= ranks[boundary_symbol])
            for symbol, rank in ranks.items()
        },
        description,
    )


def collapse_confusion(
    confusion: torch.Tensor,
    mapping: Sequence[int],
    output_classes: int,
) -> torch.Tensor:
    """Collapse a square confusion matrix while preserving its device."""

    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")
    if len(mapping) != confusion.shape[0]:
        raise ValueError("mapping length must match confusion size")
    if output_classes <= 0:
        raise ValueError("output_classes must be positive")
    if any(value < 0 or value >= output_classes for value in mapping):
        raise ValueError("mapping contains an output class outside the valid range")

    source = confusion.to(torch.float64)
    collapsed = source.new_zeros((output_classes, output_classes))
    for true_class, mapped_true in enumerate(mapping):
        for predicted_class, mapped_prediction in enumerate(mapping):
            collapsed[mapped_true, mapped_prediction] += source[
                true_class, predicted_class
            ]
    return collapsed


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def binary_metric_values(confusion: torch.Tensor) -> dict[str, float]:
    """Return the paper's binary metrics for layout [[TN, FP], [FN, TP]]."""

    if tuple(confusion.shape) != (2, 2):
        raise ValueError("binary confusion must have shape [2,2]")
    tn, fp = (float(value) for value in confusion[0])
    fn, tp = (float(value) for value in confusion[1])
    total = tn + fp + fn + tp
    hss_denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
        "pod": _safe_ratio(tp, tp + fn),
        "csi": _safe_ratio(tp, tp + fp + fn),
        "far": _safe_ratio(fp, tp + fp),
        "hss": _safe_ratio(2.0 * (tp * tn - fp * fn), hss_denominator),
        "tss": _safe_ratio(tp, tp + fn) - _safe_ratio(fp, fp + tn),
        "acc": _safe_ratio(tp + tn, total),
    }


def binary_true_skill_statistic(
    confusion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return TSS and whether both true classes have non-zero support.

    TSS is undefined when a validation epoch contains no positive or no
    negative examples.  The returned score is conservatively set to zero in
    that case, and the validity tensor lets callers expose the condition.
    """

    if tuple(confusion.shape) != (2, 2):
        raise ValueError("binary confusion must have shape [2,2]")
    source = confusion.to(torch.float64)
    tn, fp = source[0]
    fn, tp = source[1]
    positive_support = tp + fn
    negative_support = fp + tn
    valid = (positive_support > 0) & (negative_support > 0)
    tss = tp / positive_support.clamp_min(1.0) - fp / negative_support.clamp_min(1.0)
    return torch.where(valid, tss, torch.zeros_like(tss)), valid.to(source.dtype)
