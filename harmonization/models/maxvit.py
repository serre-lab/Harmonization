"""
Module related to the Harmonized ResNet50 model
"""

import tensorflow as tf
from keras_cv_attention_models.maxvit import  MaxViT_Tiny

HARMONIZED_TINY_MAXVIT_WEIGHTS = ('https://storage.googleapis.com/serrelab/prj_harmonization/'
                                  'models/maxvit_tiny_harmonized.h5')


def load_tiny_MaxViT():
    """
    Loads the (tiny) Harmonized MaxViT.

    Returns
    -------
    model
        Harmonized MaxViT keras model.
    """
    weights_path = tf.keras.utils.get_file("maxvit_tiny_harmonized", HARMONIZED_TINY_MAXVIT_WEIGHTS,
                                            cache_subdir="models")

    model = MaxViT_Tiny(classifier_activation = None, pretrained = None)
    model.load_weights(weights_path)

    return model
