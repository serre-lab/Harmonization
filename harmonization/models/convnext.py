"""
Module related to the Harmonized ResNet50 model
"""

import tensorflow as tf
from keras_cv_attention_models.convnext import ConvNeXtTiny

HARMONIZED_TINY_CONVNEXT_WEIGHTS = ('https://storage.googleapis.com/serrelab/prj_harmonization/'
                                    'models/convnext_tiny_harmonized.h5')


def load_tiny_ConvNeXT():
    """
    Loads the (tiny) Harmonized ConvNeXT.

    Returns
    -------
    model
        Harmonized ConvNeXT keras model.
    """
    weights_path = tf.keras.utils.get_file("tiny_convnext_harmonized",
                                            HARMONIZED_TINY_CONVNEXT_WEIGHTS,
                                            cache_subdir="models")

    model = ConvNeXtTiny(classifier_activation=None, pretrained=None)
    model.load_weights(weights_path)

    return model
