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
    class_ids: tuple[int, ...] = (1, 2)
    history: dict[str, list[float]] = field(
        default_factory=lambda: {
            "dice_disc": [],
            "dice_cup": [],
            "iou_disc": [],
            "iou_cup": [],
        }
    )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        for class_id, dice_key, iou_key in [
            (1, "dice_disc", "iou_disc"),
            (2, "dice_cup", "iou_cup"),
        ]:
            pred_mask = (preds == class_id).float()
            true_mask = (targets == class_id).float()
            self.history[dice_key].append(dice_score(pred_mask, true_mask))
            self.history[iou_key].append(iou_score(pred_mask, true_mask))

    def compute(self) -> dict[str, float]:
        results = {key: sum(values) / max(len(values), 1) for key, values in self.history.items()}
        results["mean_dice"] = (results["dice_disc"] + results["dice_cup"]) / 2
        results["mean_iou"] = (results["iou_disc"] + results["iou_cup"]) / 2
        return results

