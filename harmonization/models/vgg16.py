"""
Module related to the Harmonized VGG model
"""

import tensorflow as tf

HARMONIZED_VGG16_WEIGHTS = ('https://storage.googleapis.com/serrelab/prj_harmonization'
                            '/models/vgg16_harmonized.h5')


def load_VGG16():
    """
    Loads the Harmonized VGG16.

    Returns
    -------
    model
        Harmonized VGG16 keras model.
    """
    weights_path = tf.keras.utils.get_file("vgg16_harmonized", HARMONIZED_VGG16_WEIGHTS,
                                            cache_subdir="models")
    model = tf.keras.models.load_model(weights_path)

    return model
