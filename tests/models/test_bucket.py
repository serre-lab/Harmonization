"""
Ensure the bucket storing the models is publicly accessible
"""

import tensorflow as tf

DUMMY_MODEL_WEIGHTS = ('https://storage.googleapis.com/serrelab/prj_harmonization/'
                       'models/dummy_model.h5')


def test_load_dummy_model():
    """
    Test that the dummy model is publicly accessible
    """
    weights_path = tf.keras.utils.get_file("dummy_model", DUMMY_MODEL_WEIGHTS,
                                            cache_subdir="models")
    model = tf.keras.models.load_model(weights_path)

    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) > 1
