"""
Module related to the click-me dataset
"""

import tensorflow as tf

from .blur import gaussian_kernel, gaussian_blur

CLICKME_BASE_URL = 'https://storage.googleapis.com/serrelab/prj_harmonization/dataset/click-me'
NB_VAL_SHARDS = 17
NB_TRAIN_SHARDS = 318

AUTO = tf.data.AUTOTUNE
GAUSSIAN_KERNEL = tf.cast(gaussian_kernel(), tf.float32)
FEATURE_DESCRIPTION = {
      "image"       : tf.io.FixedLenFeature([], tf.string, default_value=''),
      "heatmap"     : tf.io.FixedLenFeature([], tf.string, default_value=''),
      "label"       : tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


def parse_clickme_prototype(prototype):
    """
    Parses a Click-me prototype.

    Parameters
    ----------
    prototype : tf.Tensor
        The Click-me prototype to parse.

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

    label   = tf.cast(data['label'], tf.int32)
    label   = tf.one_hot(label, 1_000)

    return image, heatmap, label


def load_clickme(shards_paths, batch_size):
    """
    Loads the click-me dataset (training or validation).

    Parameters
    ----------
    shards_paths : list of str
        The path to the shards to load.
    batch_size : int, optional
        Batch size, by default 64

    Returns
    -------
    dataset
        A `tf.dataset` of the Click-me dataset.
        Each element contains a batch of (images, heatmaps, labels).
    """
    deterministic_order = tf.data.Options()
    deterministic_order.experimental_deterministic = True

    dataset = tf.data.TFRecordDataset(shards_paths, num_parallel_reads=AUTO)
    dataset = dataset.with_options(deterministic_order)

    dataset = dataset.map(parse_clickme_prototype, num_parallel_calls=AUTO)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)

    return dataset


def load_clickme_train(batch_size = 64):
    """
    Loads the click-me training set.

    Parameters
    ----------
    batch_size : int, optional
        Batch size, by default 64

    Returns
    -------
    dataset
        A `tf.dataset` of the Click-me training dataset.
        Each element contains a batch of (images, heatmaps, labels).
    """

    shards_paths = [
        tf.keras.utils.get_file(f"clickme_train_{i}",
                                f"{CLICKME_BASE_URL}/train/train-{i}.tfrecords",
                               cache_subdir="datasets/click-me") for i in range(NB_TRAIN_SHARDS)
    ]

    return load_clickme(shards_paths, batch_size)


def load_clickme_val(batch_size = 64):
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

    shards_paths = [
        tf.keras.utils.get_file(f"clickme_val_{i}",
                                f"{CLICKME_BASE_URL}/val/val-{i}.tfrecords",
                                cache_subdir="datasets/click-me") for i in range(NB_VAL_SHARDS)
    ]

    return load_clickme(shards_paths, batch_size)
