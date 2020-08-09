"""
Sample model, used in tests and as a starter for writing ones own.
"""

from os import environ

if environ.get("TF_KERAS", True):
    from tensorflow import keras
else:
    import keras


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
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


__all__ = ["get_model"]
