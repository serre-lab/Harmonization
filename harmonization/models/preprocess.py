"""
Module related to image preprocessing
"""

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess_input(images):
    """
    Preprocesses images for the harmonized models.
    The images are expected to be in RGB format with values in the range [0, 255].

    Parameters
    ----------
    images
        Tensor or numpy array to be preprocessed.
        Expected shape (N, W, H, C).

    Returns
    -------
    preprocessed_images
        Images preprocessed for the harmonized models.
    """
    images = images / 255.0

    images = images - IMAGENET_MEAN
    images = images / IMAGENET_STD

    return images
