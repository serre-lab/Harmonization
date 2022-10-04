"""
Module related to the Human Alignment metrics
"""

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

EPSILON = 1e-9


def spearman_correlation(heatmaps_a, heatmaps_b):
    """
    Computes the Spearman correlation between two sets of heatmaps.

    Parameters
    ----------
    heatmaps_a
        First set of heatmaps.
        Expected shape (N, W, H).
    heatmaps_b
        Second set of heatmaps.
        Expected shape (N, W, H).

    Returns
    -------
    spearman_correlations
        Array of Spearman correlation score between the two sets of heatmaps.
    """
    assert heatmaps_a.shape == heatmaps_b.shape, "The two sets of heatmaps must" \
                                                 "have the same shape."
    assert len(heatmaps_a.shape) == 3, "The two sets of heatmaps must have shape (N, W, H)."

    scores = []

    heatmaps_a = tf.cast(heatmaps_a, tf.float32).numpy()
    heatmaps_b = tf.cast(heatmaps_b, tf.float32).numpy()

    for ha, hb in zip(heatmaps_a, heatmaps_b):
        rho, _ = spearmanr(ha.flatten(), hb.flatten())
        scores.append(rho)

    return np.array(scores)


def intersection_over_union(heatmaps_a, heatmaps_b, percentile = 10):
    """
    Computes the Intersection over Union (IoU) between two sets of heatmaps.
    Use a percentile threshold to binarize the heatmaps.

    Parameters
    ----------
    heatmaps_a
        First set of heatmaps.
        Expected shape (N, W, H).
    heatmaps_b
        Second set of heatmaps.
        Expected shape (N, W, H).

    Returns
    -------
    ious_scores
        Array of IoU scores between the two sets of heatmaps.
    """
    assert heatmaps_a.shape == heatmaps_b.shape, "The two sets of heatmaps must" \
                                                 "have the same shape."
    assert len(heatmaps_a.shape) == 3, "The two sets of heatmaps must have shape (N, W, H)."

    scores = []

    heatmaps_a = tf.cast(heatmaps_a, tf.float32).numpy()
    heatmaps_b = tf.cast(heatmaps_b, tf.float32).numpy()

    for ha, hb in zip(heatmaps_a, heatmaps_b):
        ha = (ha > np.percentile(ha, 100-percentile, (0, 1))).astype(np.float32)
        hb = (hb > np.percentile(hb, 100-percentile, (0, 1))).astype(np.float32)

        iou_inter = np.sum(np.logical_and(ha, hb))
        iou_union = np.sum(np.logical_or(ha, hb))

        iou_score = iou_inter / (iou_union + EPSILON)

        scores.append(iou_score)

    return np.array(scores)


def dice(heatmaps_a, heatmaps_b, percentile = 10):
    """
    Computes the Sorensen-Dice score between two sets of heatmaps.
    Use a percentile threshold to binarize the heatmaps.

    Parameters
    ----------
    heatmaps_a
        First set of heatmaps.
        Expected shape (N, W, H).
    heatmaps_b
        Second set of heatmaps.
        Expected shape (N, W, H).

    Returns
    -------
    dice_scores
        Array of dice scores between the two sets of heatmaps.
    """
    scores = []

    heatmaps_a = tf.cast(heatmaps_a, tf.float32).numpy()
    heatmaps_b = tf.cast(heatmaps_b, tf.float32).numpy()

    for ha, hb in zip(heatmaps_a, heatmaps_b):
        ha = (ha > np.percentile(ha, 100-percentile, (0, 1))).astype(np.float32)
        hb = (hb > np.percentile(hb, 100-percentile, (0, 1))).astype(np.float32)

        dice_score = 2.0 * np.sum(ha * hb) / (np.sum(ha + hb) + EPSILON)

        scores.append(dice_score)

    return np.array(scores)
