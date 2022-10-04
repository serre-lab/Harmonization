import numpy as np
from harmonization.evaluation import dice, spearman_correlation, intersection_over_union

from ..utils import almost_equal

def test_dice_metric():
    """
    Test the Dice metric
    """
    heatmaps_a = np.random.rand(10, 224, 224)
    heatmaps_b = np.random.rand(10, 224, 224)

    dice_scores = dice(heatmaps_a, heatmaps_b)

    assert dice_scores.shape == (10,)
    assert np.all(dice_scores >= 0.0)
    assert np.all(dice_scores <= 1.0)

    perfect_score = dice(heatmaps_a, heatmaps_a)
    assert almost_equal(perfect_score, 1.0)

    worst_score = dice(heatmaps_a, 1.0 - heatmaps_a)
    assert almost_equal(worst_score, 0.0)


def test_spearman_correlation_metric():
    """
    Test the Spearman correlation
    """
    heatmaps_a = np.random.rand(10, 224, 224)
    heatmaps_b = np.random.rand(10, 224, 224)

    spearman_scores = spearman_correlation(heatmaps_a, heatmaps_b)

    assert spearman_scores.shape == (10,)
    assert np.all(spearman_scores >= -1.0)
    assert np.all(spearman_scores <= 1.0)

    perfect_score = spearman_correlation(heatmaps_a, heatmaps_a)
    assert almost_equal(perfect_score, 1.0)

    worst_score = spearman_correlation(heatmaps_a, -heatmaps_a)
    assert almost_equal(worst_score, -1.0)


def test_intersection_over_union_metric():
    """
    Test the Intersection over Union metric
    """
    heatmaps_a = np.random.rand(10, 224, 224)
    heatmaps_b = np.random.rand(10, 224, 224)

    iou_scores = intersection_over_union(heatmaps_a, heatmaps_b)

    assert iou_scores.shape == (10,)
    assert np.all(iou_scores >= 0.0)
    assert np.all(iou_scores <= 1.0)

    perfect_score = intersection_over_union(heatmaps_a, heatmaps_a)
    assert almost_equal(perfect_score, 1.0)

    worst_score = intersection_over_union(heatmaps_a, 1.0 - heatmaps_a)
    assert almost_equal(worst_score, 0.0)
