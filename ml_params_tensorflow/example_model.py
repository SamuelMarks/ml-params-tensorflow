"""
Sample model, used in tests and as a starter for writing ones own.
"""

import tensorflow as tf


def get_model(num_classes, input_shape=(28, 28, 1)):
    """
    Gets the model.

    :param num_classes: Number of classes (e.g., 10 for MNIST, 2 for cat v dog)
    :type num_classes: ```int```

    :param input_shape: The shape of the input, 3 in image context usually means (height, width, channels)
    :type input_shape: ```Tuple[int, int, int]```

    :returns: Keras model configured with the input
    :rtype: ```keras.Sequential```
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
