from __future__ import annotations

from dataclasses import dataclass, field

import torch


def dice_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-6) -> float:
    intersection = torch.sum(pred_mask * true_mask).item()
    union = torch.sum(pred_mask).item() + torch.sum(true_mask).item()
    return (2 * intersection + eps) / (union + eps)


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-6) -> float:
    intersection = torch.sum(pred_mask * true_mask).item()
    union = torch.sum((pred_mask + true_mask) > 0).item()
    return (intersection + eps) / (union + eps)


@dataclass
class SegmentationMeter:
    history: dict[str, list[float]] = field(
        default_factory=lambda: {
            "dice_disc": [],
            "dice_cup": [],
            "iou_disc": [],
            "iou_cup": [],
        }
    )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        pred_disc = ((preds == 1) | (preds == 2)).float()
        true_disc = ((targets == 1) | (targets == 2)).float()
        pred_cup = (preds == 2).float()
        true_cup = (targets == 2).float()

        self.history["dice_disc"].append(dice_score(pred_disc, true_disc))
        self.history["iou_disc"].append(iou_score(pred_disc, true_disc))
        self.history["dice_cup"].append(dice_score(pred_cup, true_cup))
        self.history["iou_cup"].append(iou_score(pred_cup, true_cup))

    def compute(self) -> dict[str, float]:
        results = {key: sum(values) / max(len(values), 1) for key, values in self.history.items()}
        results["mean_dice"] = (results["dice_disc"] + results["dice_cup"]) / 2
        results["mean_iou"] = (results["iou_disc"] + results["iou_cup"]) / 2
        return results
