"""
Module related to the Harmonized ResNet50 model
"""

import tensorflow as tf
from keras_cv_attention_models.levit import LeViT128

HARMONIZED_LEVIT_SMALL_WEIGHTS = ('https://storage.googleapis.com/serrelab/prj_harmonization/'
                                  'models/levit_small_harmonized.h5')


def load_LeViT_small():
    """
    Loads the (small) Harmonized LeViT.

    Returns
    -------
    model
        Harmonized LeViT keras model.
    """
    weights_path = tf.keras.utils.get_file("levit_small_harmonized", HARMONIZED_LEVIT_SMALL_WEIGHTS,
                                            cache_subdir="models")

    model = LeViT128(classifier_activation = None, use_distillation = False)
    model.load_weights(weights_path)

    return model
