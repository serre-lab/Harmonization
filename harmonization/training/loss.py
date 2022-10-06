"""
Harmonization loss
"""

import tensorflow as tf
from .pyramid import pyramidal_representation


Cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                        reduction=tf.keras.losses.Reduction.NONE,
                                                        label_smoothing=0.1)


def standardize_cut(heatmaps, axes=(1, 2), epsilon=1e-5):
    """
    Standardize the heatmaps (zero mean, unit variance) and apply ReLU.

    Parameters
    ----------
    heatmaps : tf.Tensor
        The heatmaps to standardize.
    axes : tuple
        The axes to compute the mean and variance.
    epsilon : float
        A small value to avoid division by zero.

    Returns
    -------
    tf.Tensor
        The positive part of the standardized heatmaps.
    """
    means = tf.reduce_mean(heatmaps, axes, keepdims=True)
    stds = tf.math.reduce_std(heatmaps, axes, keepdims=True)

    heatmaps = heatmaps - means
    heatmaps = heatmaps / (stds + epsilon)

    heatmaps = tf.nn.relu(heatmaps)

    return heatmaps


def _mse_with_tokens(heatmaps_a, heatmaps_b, tokens):
    """
    Compute the MSE between two set of heatmaps, weighted by the tokens.

    Parameters
    ----------
    heatmap_a : tf.Tensor
        The first heatmap.
    heatmap_b : tf.Tensor
        The second heatmap.
    token : tf.Tensor
        The token to weight the MSE.

    Returns
    -------
    tf.Tensor
        The weighted MSE.
    """
    return tf.reduce_mean(tf.square(heatmaps_a - heatmaps_b) * tokens[:, None, None, None])


def pyramidal_mse_with_tokens(true_heatmaps, predicted_heatmaps, tokens, nb_levels=5):
    """
    Compute mean squared error between two set heatmaps on a pyramidal representation.

    Parameters
    ----------
    true_heatmaps : tf.Tensor
        The true heatmaps.
    predicted_heatmaps : tf.Tensor
        The predicted heatmaps.
    tokens : tf.Tensor
        The tokens to weight the MSE.
    nb_levels : int
        The number of levels to use in the pyramid.

    Returns
    -------
    tf.Tensor
        The weighted MSE.

    """
    pyramid_y     = pyramidal_representation(true_heatmaps[:, :, :, None], nb_levels)
    pyramid_y_pred = pyramidal_representation(predicted_heatmaps[:, :, :, None], nb_levels)

    loss = tf.reduce_mean([
                            _mse_with_tokens(pyramid_y[i], pyramid_y_pred[i], tokens)
                            for i in range(nb_levels)])

    return loss


def harmonizer_loss(model, images, tokens, labels, true_heatmaps,
                    cross_entropy = Cross_entropy, lambda_weights=1e-5, lambda_harmonization=1.0):
    """
    Compute the harmonization loss: cross entropy + pyramidal mse of standardized-cut heatmaps.

    Parameters
    ----------
    model : tf.keras.Model
        The model to train.
    images : tf.Tensor
        The batch of images to train on.
    tokens : tf.Tensor
        The batch of tokens to weight the MSE.
    labels : tf.Tensor
        The batch of labels.
    true_heatmaps : tf.Tensor
        The batch of true heatmaps (e.g Click-me maps) to align the model on.
    cross_entropy : tf.keras.losses.Loss
        The cross entropy loss to use.

    Returns
    -------
    gradients : tf.Tensor
        The gradients on the given batch according to the harmonization loss.
    """
    with tf.GradientTape() as tape:
        tape.watch(images)
        tape.watch(model.trainable_variables)

        with tf.GradientTape() as tape_metapred:
            tape_metapred.watch(images)
            tape_metapred.watch(model.trainable_variables)

            y_pred = model(images, training=True)
            loss_metapred = tf.reduce_sum(y_pred * labels, -1)

        # compute the saliency maps
        sa_maps = tf.cast(tape_metapred.gradient(loss_metapred, images), tf.float32)
        sa_maps = tf.reduce_mean(sa_maps, -1)
        # apply the standardization-cut procedure on heatmaps
        sa_maps_preprocess = standardize_cut(sa_maps)
        heatmaps_preprocess = standardize_cut(true_heatmaps)

        # re-normalize before pyramidal
        _hm_max = tf.math.reduce_max(heatmaps_preprocess, (1, 2), keepdims=True) + 1e-6
        _sa_max = tf.stop_gradient(tf.math.reduce_max(sa_maps_preprocess, (1, 2),
                                                keepdims=True))  + 1e-6
        # normalize the true heatmaps according to the saliency maps
        heatmaps_preprocess = heatmaps_preprocess / _hm_max * _sa_max

        # compute and combine the losses
        harmonization_loss = pyramidal_mse_with_tokens(sa_maps_preprocess,
                                                       heatmaps_preprocess, tokens)
        cce_loss = tf.reduce_mean(cross_entropy(labels, y_pred))
        weight_loss = tf.add_n([tf.nn.l2_loss(v)
                                for v in model.trainable_variables \
                                if 'bn/' not in v.name
                                and 'normalization' not in v.name
                                and 'embed' not in v.name
                                and 'Norm' not in v.name
                                and 'norm' not in v.name
                                and 'class_token' not in v.name
                                ])

        loss = tf.cast(cce_loss, tf.float32) + \
               tf.cast(weight_loss * lambda_weights, tf.float32) + \
               tf.cast(harmonization_loss * lambda_harmonization, tf.float32)

    gradients = tape.gradient(loss, model.trainable_variables)

    return gradients
