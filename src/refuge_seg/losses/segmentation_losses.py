from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target_oh = one_hot(target, logits.shape[1])
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dims)
        union = torch.sum(probs + target_oh, dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def topology_penalty(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    background = probs[:, 0:1]
    cup = probs[:, 2:3]
    cup_max = F.max_pool2d(cup, kernel_size=3, stride=1, padding=1)
    cup_min = -F.max_pool2d(-cup, kernel_size=3, stride=1, padding=1)
    cup_boundary = torch.relu(cup_max - cup_min)
    nearby_background = F.max_pool2d(background, kernel_size=9, stride=1, padding=4)
    return (cup_boundary * nearby_background).mean()


class TopologyAwareLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, lambda_topology: float = 0.3) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.lambda_topology = lambda_topology

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.base_loss(logits, target) + self.lambda_topology * topology_penalty(logits)


def build_loss(name: str, lambda_topology: float = 0.3) -> nn.Module:
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "dice":
        return DiceLoss()
    if name == "dice_ce":
        return DiceCrossEntropyLoss()
    if name == "topology":
        return TopologyAwareLoss(DiceCrossEntropyLoss(), lambda_topology=lambda_topology)
    raise ValueError(f"Unsupported loss: {name}")
