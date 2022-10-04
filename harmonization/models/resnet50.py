"""
Module related to the Harmonized ResNet50 model
"""

import tensorflow as tf

HARMONIZED_RESNET50_WEIGHTS = ('https://storage.googleapis.com/serrelab/prj_harmonization/'
                               'models/resnet50v2_harmonized.h5')


def load_ResNet50():
    """
    Loads the Harmonized ResNet50.

    Returns
    -------
    model
        Harmonized ResNet50 keras model.
    """
    weights_path = tf.keras.utils.get_file("resnet50v2_harmonized", HARMONIZED_RESNET50_WEIGHTS,
                                            cache_subdir="models")
    model = tf.keras.models.load_model(weights_path)

    return model
