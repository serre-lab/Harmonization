import tensorflow as tf

from harmonization.common.clickme_dataset import load_clickme, CLICKME_BASE_URL


def test_loading_clickme_val():
    """
    Test that the click-me validation dataset is publicly accessible
    and that the dataset is correctly preprocessed.
    """
    bs = 32
    single_shard = tf.keras.utils.get_file(f"clickme_val_17", f"{CLICKME_BASE_URL}/val/val-17.tfrecords",
                                           cache_subdir="datasets/click-me")
    dataset = load_clickme(shards_paths=[single_shard], batch_size = bs)

    for images, heatmaps, labels in dataset.take(1):
        assert images.shape == (bs, 224, 224, 3)
        assert heatmaps.shape == (bs, 224, 224, 1)
        assert labels.shape == (bs, 1000)
        assert tf.reduce_min(images) >= 0.0
        assert tf.reduce_max(images) <= 255.0
        assert tf.reduce_min(heatmaps) >= 0.0


def test_loading_clickme_train():
    """
    Test that the click-me training dataset is publicly accessible
    and that the dataset is correctly preprocessed.
    """
    bs = 32
    single_shard = tf.keras.utils.get_file(f"clickme_train_318", f"{CLICKME_BASE_URL}/train/train-318.tfrecords",
                                           cache_subdir="datasets/click-me")
    dataset = load_clickme(shards_paths=[single_shard], batch_size = bs)

    for images, heatmaps, labels in dataset.take(1):
        assert images.shape == (bs, 224, 224, 3)
        assert heatmaps.shape == (bs, 224, 224, 1)
        assert labels.shape == (bs, 1000)
        assert tf.reduce_min(images) >= 0.0
        assert tf.reduce_max(images) <= 255.0
        assert tf.reduce_min(heatmaps) >= 0.0
