"""
Module related to the Harmonized ResNet50 model
"""

import tensorflow as tf
from vit_keras import vit

HARMONIZED_VITB16_WEIGHTS = ('https://storage.googleapis.com/serrelab/prj_harmonization/'
                             'models/vit-b16_harmonized.h5')


def load_ViT_B16():
    """
    Loads the Harmonized ViT-B16.

    Returns
    -------
    model
        Harmonized ViT-B16 keras model.
    """
    weights_path = tf.keras.utils.get_file("vit-b16_harmonized", HARMONIZED_VITB16_WEIGHTS,
                                            cache_subdir="models")

    model = vit.vit_b16(
        image_size=224,
        activation='linear',
        pretrained=False,
        include_top=True,
        pretrained_top=False
    )
    model.load_weights(weights_path)

    return model
