from __future__ import annotations

import numpy as np
from scipy import ndimage


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    component_sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    largest_component = int(np.argmax(component_sizes)) + 1
    return (labeled == largest_component).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(mask).astype(np.uint8)


def postprocess_prediction(pred: np.ndarray) -> np.ndarray:
    processed = np.zeros_like(pred, dtype=np.uint8)
    disc = np.isin(pred, [1, 2]).astype(np.uint8)
    cup = (pred == 2).astype(np.uint8)

    disc = _fill_holes(_keep_largest_component(disc))
    cup = _fill_holes(_keep_largest_component(cup))
    cup = cup * disc

    processed[disc == 1] = 1
    processed[cup == 1] = 2
    return processed


def diagnose_prediction(pred: np.ndarray) -> dict[str, int | bool]:
    disc = np.isin(pred, [1, 2]).astype(np.uint8)
    cup = (pred == 2).astype(np.uint8)
    disc_components = ndimage.label(disc)[1]
    cup_components = ndimage.label(cup)[1]
    disc_holes = int(ndimage.binary_fill_holes(disc).sum() - disc.sum())
    cup_holes = int(ndimage.binary_fill_holes(cup).sum() - cup.sum())
    cup_outside_disc = bool(np.any((cup == 1) & (disc == 0)))
    return {
        "disc_components": disc_components,
        "cup_components": cup_components,
        "disc_holes_pixels": disc_holes,
        "cup_holes_pixels": cup_holes,
        "cup_outside_disc": cup_outside_disc,
    }
