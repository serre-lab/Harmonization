"""
Gaussian kernel & associated Gaussian blur utils
"""

import tensorflow as tf
import numpy as np

BLUR_KERNEL_SIZE = 10
BLUR_SIGMA = 10

def gaussian_kernel(size = BLUR_KERNEL_SIZE, sigma = BLUR_SIGMA):
    """
    Generates a 2D Gaussian kernel.

    Parameters
    ----------
    size : int, optional
      Kernel size, by default BLUR_KERNEL_SIZE
    sigma : int, optional
      Kernel sigma, by default BLUR_SIGMA

    Returns
    -------
    kernel : tf.Tensor
      A Gaussian kernel.
    """
    x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = tf.meshgrid(x_range, y_range)
    kernel = tf.exp(-(xs**2 + ys**2)/(2*(sigma**2))) / (2*np.pi*(sigma**2))

    kernel = tf.cast(kernel / tf.reduce_sum(kernel), tf.float32)

    return tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)


def gaussian_blur(heatmap, kernel):
    """
    Blurs a heatmap with a Gaussian kernel.

    Parameters
    ----------
    heatmap : tf.Tensor
      The heatmap to blur.
    kernel : tf.Tensor
      The Gaussian kernel.

    Returns
    -------
    heatmap : tf.Tensor
      The blurred heatmap.
    """
    heatmap = tf.nn.conv2d(heatmap[None, :, :, :], kernel, [1, 1, 1, 1], 'SAME')

    return heatmap[0]
