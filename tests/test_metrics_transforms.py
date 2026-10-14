import pytest
import torch

from tinymedvision.calibration import calibration_metrics
from tinymedvision.evaluation import bootstrap_auc_ci, evaluate_predictions
from tinymedvision.transforms import corrupt


def test_known_binary_metrics():
    result = evaluate_predictions([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.4], threshold=0.5)
    assert result["tn"] == 1 and result["fp"] == 1
    assert result["tp"] == 1 and result["fn"] == 1
    assert result["roc_auc"] == 0.75
    assert result["sensitivity"] == 0.5


def test_single_class_auc_is_undefined():
    result = evaluate_predictions([1, 1], [0.2, 0.9])
    assert result["roc_auc"] is None
    assert bootstrap_auc_ci([0, 0], [0.3, 0.6]) is None


def test_constant_scores_and_invalid_metric_inputs():
    assert evaluate_predictions([0, 1], [0.5, 0.5])["roc_auc"] == 0.5
    with pytest.raises(ValueError):
        evaluate_predictions([0, 1], [0.2])
    with pytest.raises(ValueError):
        evaluate_predictions([0, 1], [float("nan"), 0.8])


def test_calibration_brier_known_values():
    result = calibration_metrics([0, 1], [0.0, 1.0], bins=2)
    assert result["brier"] == 0
    assert result["ece"] == 0


def test_corruptions_are_seeded_and_validated():
    image = torch.full((1, 1, 16, 16), 0.5)
    assert torch.equal(corrupt(image, "noise", 0.4, seed=1), corrupt(image, "noise", 0.4, seed=1))
    assert corrupt(image, "blur", 0.7).shape == image.shape
    with pytest.raises(ValueError):
        corrupt(image, "noise", 1.2)
