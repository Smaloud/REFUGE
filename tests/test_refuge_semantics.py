import numpy as np
import torch

from refuge_seg.utils.metrics import SegmentationMeter
from refuge_seg.utils.postprocess import diagnose_prediction, postprocess_prediction


def test_disc_metrics_include_cup_region() -> None:
    target = torch.tensor([[[0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 1, 0]]])
    pred = torch.tensor([[[0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 1, 0]]])

    meter = SegmentationMeter()
    meter.update(pred, target)
    metrics = meter.compute()

    assert metrics["dice_disc"] == 1.0
    assert metrics["dice_cup"] == 1.0


def test_empty_prediction_does_not_get_high_disc_score() -> None:
    target = torch.tensor([[[0, 1, 1, 0], [0, 1, 2, 0], [0, 1, 1, 0]]])
    pred = torch.zeros_like(target)

    meter = SegmentationMeter()
    meter.update(pred, target)
    metrics = meter.compute()

    assert metrics["dice_disc"] < 1e-5
    assert metrics["dice_cup"] < 1e-5


def test_postprocess_keeps_cup_inside_complete_disc() -> None:
    pred = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    processed = postprocess_prediction(pred)
    diagnosis = diagnose_prediction(processed)

    assert processed[2, 2] == 2
    assert diagnosis["disc_components"] == 1
    assert diagnosis["cup_components"] == 1
    assert not diagnosis["cup_outside_disc"]
