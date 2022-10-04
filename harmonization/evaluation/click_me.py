"""
Module related to the click-me dataset
"""

import tensorflow as tf
from xplique.attributions import Saliency

from ..models import preprocess_input
from .utils import gaussian_kernel, gaussian_blur
from .metrics import spearman_correlation, dice, intersection_over_union


CLICKME_BASE_URL = 'https://storage.googleapis.com/serrelab/prj_harmonization/dataset/click-me'
NB_VAL_SHARDS = 17
AUTO = tf.data.AUTOTUNE

GAUSSIAN_KERNEL = tf.cast(gaussian_kernel(), tf.float32)
FEATURE_DESCRIPTION = {
      "image"       : tf.io.FixedLenFeature([], tf.string, default_value=''),
      "heatmap"     : tf.io.FixedLenFeature([], tf.string, default_value=''),
      "label"       : tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


def parse_clickme_prototype(prototype, training=False):
    """
    Parses a Click-me prototype.

    Parameters
    ----------
    prototype : tf.Tensor
        The Click-me prototype to parse.
    training : bool, optional
        Whether the prototype is from the training set, by default False

    Returns
    -------
    image : tf.Tensor
        The image.
    heatmap : tf.Tensor
        The heatmap.
    label : tf.Tensor
        The label.
    """
    # parse a single sample
    data = tf.io.parse_single_example(prototype, FEATURE_DESCRIPTION)

    if not training:
        # load & preprocess image
        image   = tf.io.decode_jpeg(data['image'])
        image   = tf.reshape(image, (256, 256, 3))
        image   = tf.cast(image, tf.float32)
        image   = tf.image.resize(image, (224, 224), method='bilinear')

        # load & blur the heatmap
        heatmap = tf.io.decode_jpeg(data['heatmap'])
        heatmap = tf.reshape(heatmap, (256, 256, 1))
        heatmap = tf.cast(heatmap, tf.float32)
        heatmap = tf.image.resize(heatmap, (64, 64), method="bilinear")
        heatmap = gaussian_blur(heatmap, GAUSSIAN_KERNEL)
        heatmap = tf.image.resize(heatmap, (224, 224), method="bilinear")
    else:
        raise NotImplementedError

    label   = tf.cast(data['label'], tf.int32)
    label   = tf.one_hot(label, 1_000)

    return image, heatmap, label


def load_clickme_val(shards_paths = None, batch_size = 64):
    """
    Loads the click-me validation set.

    Parameters
    ----------
    batch_size : int, optional
        Batch size, by default 64

    Returns
    -------
    dataset
        A `tf.dataset` of the Click-me validation dataset.
        Each element contains a batch of (images, heatmaps, labels).
    """

    if shards_paths is None:
        shards_paths = [
            tf.keras.utils.get_file(f"clickme_val_{i}", f"{CLICKME_BASE_URL}/val/val-{i}.tfrecords",
                                    cache_subdir="datasets/click-me") for i in range(NB_VAL_SHARDS)
        ]

    deterministic_order = tf.data.Options()
    deterministic_order.experimental_deterministic = True

    dataset = tf.data.TFRecordDataset(shards_paths, num_parallel_reads=AUTO)
    dataset = dataset.with_options(deterministic_order)

    dataset = dataset.map(parse_clickme_prototype, num_parallel_calls=AUTO)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)

    return dataset


def evaluate_tf_model(model, clickme_val_dataset = None, preprocess_inputs = preprocess_input):
    """
    Evaluates a model on the Click-me validation set.

    Parameters
    ----------
    model : tf.keras.Model
        The model to evaluate.
    preprocess_inputs : function, optional
        The preprocessing function to apply to the inputs, by default `preprocess_input`
    batch_size : int, optional
        Batch size, by default 64

    Returns
    -------
    scores : dict
        The Human Alignements metrics (Spearman, Dice, IoU) on the Click-me validation set.
    """
    if clickme_val_dataset is None:
        clickme_val_dataset = load_clickme_val()

    clickme_val_dataset = clickme_val_dataset.map(lambda x, y, z: (preprocess_inputs(x), y, z),
                                                   num_parallel_calls=AUTO)

    explainer = Saliency(model)

    metrics = {
        'spearman': [],
        'dice': [],
        'iou': [],
    }

    for images_batch, heatmaps_batch, label_batch in clickme_val_dataset:

        # tfel: use xplique saliency maps (max over channels of absolute values)
        saliency_maps = explainer(images_batch, label_batch)

        if len(saliency_maps.shape) == 4:
            saliency_maps = tf.reduce_mean(saliency_maps, -1)
        if len(heatmaps_batch.shape) == 4:
            heatmaps_batch = tf.reduce_mean(heatmaps_batch, -1)

        spearman_batch = spearman_correlation(saliency_maps, heatmaps_batch)
        dice_batch = dice(saliency_maps, heatmaps_batch)
        iou_batch = intersection_over_union(saliency_maps, heatmaps_batch)

        metrics['spearman'] += list(spearman_batch)
        metrics['dice']     += list(dice_batch)
        metrics['iou']      += list(iou_batch)

    return metrics
