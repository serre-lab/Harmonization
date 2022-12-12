import tensorflow as tf
import numpy as np

from harmonization.evaluation import evaluate_clickme
from harmonization.common.clickme_dataset import load_clickme, CLICKME_BASE_URL


def test_evaluate_tf_model():

    # load a single click-me shard
    batch_size = 32
    nb_elements = 256 # number of elements in the the smallest shard

    single_shard = tf.keras.utils.get_file(f"clickme_val_17", f"{CLICKME_BASE_URL}/val/val-17.tfrecords",
                                           cache_subdir="datasets/click-me")
    dataset = load_clickme(shards_paths=[single_shard], batch_size = batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((10, 10), strides=(10, 10)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='linear')
    ])

    def identity(x):
        return x

    scores = evaluate_clickme(model, clickme_val_dataset=dataset, preprocess_inputs=identity, explainer=None)

    assert 'spearman' in scores
    assert 'dice' in scores
    assert 'iou' in scores

    assert len(scores['spearman']) == nb_elements
    assert len(scores['dice']) == nb_elements
    assert len(scores['iou']) == nb_elements

    assert tf.reduce_min(scores['spearman']) >= -1.0
    assert tf.reduce_max(scores['spearman']) <= 1.0

    assert tf.reduce_min(scores['dice']) >= 0.0
    assert tf.reduce_max(scores['dice']) <= 1.0

    assert tf.reduce_min(scores['iou']) >= 0.0
    assert tf.reduce_max(scores['iou']) <= 1.0

    assert np.isnan(scores['spearman']).sum() == 0
    assert np.isnan(scores['dice']).sum() == 0
    assert np.isnan(scores['iou']).sum() == 0
