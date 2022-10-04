"""
Module related to the Harmonized EfficientNet model
"""

# tfel: let keras autoload the custom layers needed by efficientnet
import efficientnet.keras # pylint: disable=W0611
import tensorflow as tf

HARMONIZED_EFFICIENTNETB0_WEIGHTS = ('https://storage.googleapis.com/serrelab/prj_harmonization/'
                                     'models/efficientnetB0_harmonized.h5')


def load_EfficientNetB0():
    """
    Loads the Harmonized EfficientNetB0.

    Returns
    -------
    model
        Harmonized EfficientNetB0 keras model.
    """
    weights_path = tf.keras.utils.get_file("efficientnetB0_harmonized",
                                           HARMONIZED_EFFICIENTNETB0_WEIGHTS,
                                           cache_subdir="models")
    model = tf.keras.models.load_model(weights_path)

    return model
