"""
Sample model, used in tests and as a starter for writing ones own.
"""
from typing import Tuple

import tensorflow as tf


def get_model(
    num_classes: int, input_shape: Tuple[int, int, int] = (28, 28, 1)
) -> tf.keras.Sequential:
    """
    Gets the model.

    :param num_classes: Number of classes (e.g., 10 for MNIST, 2 for cat v dog)

    :param input_shape: The shape of the input, 3 in image context usually means (height, width, channels)

    :return: Keras model configured with the input
    """
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def hack(self, model):
    """ TODO: Delete """

    print(model.input_shape)

    print(model.summary())

    # model = ml_params_tensorflow.example_model.get_model(self.ds_info.features["label"].num_classes)

    print("model.input_shape:", model.input_shape, ";")
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(224, 224, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, kernel_size=(10, 10), padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(64, (3, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(
                self.ds_info.features["label"].num_classes, activation="softmax"
            ),
        ]
    )

    dot_img_file = "/tmp/model.png"
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    dot_img_file = "/tmp/model_show_shapes.png"
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    dot_img_file = "/tmp/model_show_shapes_show_dtype.png"
    tf.keras.utils.plot_model(
        model, to_file=dot_img_file, show_shapes=True, show_dtype=True
    )
    dot_img_file = "/tmp/model_false.png"
    tf.keras.utils.plot_model(
        model, to_file=dot_img_file, show_shapes=False, show_dtype=False
    )
    return model


__all__ = ["get_model"]
