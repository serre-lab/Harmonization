import tensorflow as tf
import numpy as np
from harmonization.training import (standardize_cut,
                                    pyramidal_mse_with_tokens,
                                    harmonizer_loss)
from ..utils import almost_equal


def test_standardize_cut():
    """
    Test the standardize_cut function
    """
    heatmaps = tf.random.uniform((10, 224, 224))

    preprocessed_heatmaps = standardize_cut(heatmaps)

    assert preprocessed_heatmaps.shape == (10, 224, 224)
    assert tf.reduce_min(preprocessed_heatmaps) >= 0.0


def test_pyramidal_mse_with_tokens():
    """
    Test the pyramidal_mse_with_tokens function
    """
    heatmaps = tf.random.uniform((10, 224, 224))
    predicted_heatmaps = tf.random.uniform((10, 224, 224))
    tokens = tf.random.uniform((10,))

    loss = pyramidal_mse_with_tokens(heatmaps, predicted_heatmaps, tokens)

    assert isinstance(loss, tf.Tensor)
    assert loss.shape == ()

    loss_none = pyramidal_mse_with_tokens(heatmaps, predicted_heatmaps, tf.zeros((10,)))
    assert almost_equal(loss_none, 0.0)

    loss_all = pyramidal_mse_with_tokens(heatmaps, predicted_heatmaps, tf.ones((10,)))
    assert loss_all >= loss >= loss_none


def test_harmonizer_loss():
    """
    Test the harmonizer_loss function
    """
    images = tf.random.uniform((10, 224, 224, 3))
    tokens = tf.random.uniform((10,))
    labels = tf.random.uniform((10, 10))
    true_heatmaps = tf.random.uniform((10, 224, 224))

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((10, 10), strides=(10, 10)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='linear')
    ])

    gradients = harmonizer_loss(model, images, tokens, labels, true_heatmaps)
    assert len(gradients) == len(model.trainable_variables)
