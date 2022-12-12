"""
Module related to the Pyramidal representation
"""

import numpy as np
import tensorflow as tf


def _downsample(image, kernel):
    return tf.nn.conv2d(input=image, filters=kernel, strides=[1, 2, 2, 1], padding="SAME")


def _binomial_kernel(num_channels):
    kernel = np.array((1., 4., 6., 4., 1.), dtype=np.float32)
    kernel = np.outer(kernel, kernel)
    kernel /= np.sum(kernel)
    kernel = kernel[:, :, np.newaxis, np.newaxis]
    return tf.constant(kernel, dtype=tf.float32) * tf.eye(num_channels, dtype=tf.float32)


def pyramidal_representation(image, num_levels):
    """
    Compute the pyramidal representation of an image.

    Parameters
    ----------
    image : tf.Tensor
        The image to compute the pyramidal representation.
    num_levels : int
        The number of levels to use in the pyramid.

    Returns
    -------
    list of tf.Tensor
        The pyramidal representation.
    """
    kernel = _binomial_kernel(tf.shape(input=image)[3])
    levels = [image]
    for _ in range(num_levels):
        image = _downsample(image, kernel)
        levels.append(image)
    return levels
