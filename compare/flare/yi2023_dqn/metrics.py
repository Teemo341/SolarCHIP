"""Metrics used by the Yi comparison and its three cumulative heads."""

from __future__ import annotations

import torch
from torch import nn
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
)

from .architecture import (
    HEAD_NAMES,
    cumulative_inconsistency,
    cumulative_targets,
    decode_cumulative_actions,
)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


class YiEvaluationMetrics(nn.Module):
    """Four-class accuracy/macro-F1 plus Yi's metrics for every binary head."""

    def __init__(self) -> None:
        super().__init__()
        self.multiclass = MulticlassConfusionMatrix(num_classes=4)
        self.binary = nn.ModuleList(BinaryConfusionMatrix() for _ in range(3))
        self.inconsistency = MeanMetric()

    @torch.no_grad()
    def update(
        self, values: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = values.argmax(dim=-1)
        predictions = decode_cumulative_actions(actions)
        targets = cumulative_targets(labels)
        self.multiclass.update(predictions, labels)
        for index, metric in enumerate(self.binary):
            metric.update(actions[:, index], targets[:, index])
        self.inconsistency.update(cumulative_inconsistency(actions).float())
        return predictions, actions

    def compute(self, prefix: str) -> dict[str, torch.Tensor]:
        confusion = self.multiclass.compute().float()
        total = confusion.sum()
        true_positive = confusion.diag()
        support = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        recall = true_positive / support.clamp_min(1.0)
        precision = true_positive / predicted.clamp_min(1.0)
        class_f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        output = {
            f"{prefix}_accuracy": _safe_ratio(true_positive.sum(), total),
            f"{prefix}_macro_f1": class_f1.mean(),
            f"{prefix}_inconsistent_head_rate": self.inconsistency.compute(),
        }

        for head_name, metric in zip(HEAD_NAMES, self.binary):
            binary = metric.compute().float()
            tn, fp = binary[0, 0], binary[0, 1]
            fn, tp = binary[1, 0], binary[1, 1]
            positives = tp + fn
            negatives = tn + fp
            binary_total = positives + negatives
            f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
            tss = _safe_ratio(tp, positives) - _safe_ratio(fp, negatives)
            hss_denominator = (
                (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
            )
            hss = _safe_ratio(2 * (tp * tn - fp * fn), hss_denominator)

            # Yi uses the always-no-event forecast as the Appleman reference.
            accuracy = _safe_ratio(tp + tn, binary_total)
            reference_accuracy = _safe_ratio(negatives, binary_total)
            apss = _safe_ratio(
                accuracy - reference_accuracy, 1 - reference_accuracy
            )
            output.update(
                {
                    f"{prefix}_{head_name}_f1": f1,
                    f"{prefix}_{head_name}_tss": tss,
                    f"{prefix}_{head_name}_hss": hss,
                    f"{prefix}_{head_name}_apss": apss,
                    f"{prefix}_{head_name}_positive_rate": _safe_ratio(
                        tp + fp, binary_total
                    ),
                }
            )
        return output

    def reset(self) -> None:
        self.multiclass.reset()
        for metric in self.binary:
            metric.reset()
        self.inconsistency.reset()


__all__ = ["YiEvaluationMetrics"]
