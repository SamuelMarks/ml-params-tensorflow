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


__all__ = ["get_model"]
