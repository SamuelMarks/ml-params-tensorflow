""" Generated Callback config classes """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class binary_accuracyConfig(object):
    """
    Calculates how often predictions matches binary labels.

    Standalone usage:
    >>> y_true = [[1], [1], [0], [0]]
    >>> y_pred = [[1], [1], [0], [0]]
    >>> m = tf.keras.metrics.binary_accuracy(y_true, y_pred)
    >>> assert m.shape == (4,)
    >>> m.numpy()
    array([1., 1., 1., 1.], dtype=float32)

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar threshold: (Optional) Float representing the threshold for deciding whether
    prediction values are 1 or 0. Defaults to 0.5
    :cvar return_type: Binary accuracy values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.equal(y_true, y_pred), axis=-1)```"""

    y_true = None
    y_pred = None
    threshold: float = 0.5
    return_type = "```K.mean(math_ops.equal(y_true, y_pred), axis=-1)```"


class binary_crossentropyConfig(object):
    """
    Computes the binary crossentropy loss.

    Standalone usage:

    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([0.916 , 0.714], dtype=float32)

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
    we assume that `y_pred` encodes a probability distribution. Defaults to False
    :cvar label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. Defaults to 0
    :cvar return_type: Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```(y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing)```"""

    y_true = None
    y_pred = None
    from_logits: bool = False
    label_smoothing: int = 0
    return_type = "```(y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing)```"


class categorical_accuracyConfig(object):
    """
    Calculates how often predictions matches one-hot labels.

    Standalone usage:
    >>> y_true = [[0, 0, 1], [0, 1, 0]]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([0., 1.], dtype=float32)

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    :cvar y_true: One-hot ground truth values.
    :cvar y_pred: The prediction values.
    :cvar return_type: Categorical accuracy values. Defaults to ```math_ops.cast(math_ops.equal(math_ops.argmax(y_true, axis=-1), math_ops.
    argmax(y_pred, axis=-1)), K.floatx())```"""

    y_true = None
    y_pred = None
    return_type = """```math_ops.cast(math_ops.equal(math_ops.argmax(y_true, axis=-1), math_ops.
    argmax(y_pred, axis=-1)), K.floatx())```"""


class categorical_crossentropyConfig(object):
    """
    Computes the categorical crossentropy loss.

    Standalone usage:

    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([0.0513, 2.303], dtype=float32)

    :cvar y_true: Tensor of one-hot true targets.
    :cvar y_pred: Tensor of predicted targets.
    :cvar from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
    we assume that `y_pred` encodes a probability distribution. Defaults to False
    :cvar label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. Defaults to 0
    :cvar return_type: Categorical crossentropy loss value. Defaults to ```(y_true * (1.0 - label_smoothing) + label_smoothing / num_classes)```"""

    y_true = None
    y_pred = None
    from_logits: bool = False
    label_smoothing: int = 0
    return_type = (
        "```(y_true * (1.0 - label_smoothing) + label_smoothing / num_classes)```"
    )


class hingeConfig(object):
    """
    Computes the hinge loss between `y_true` and `y_pred`.

    `loss = mean(maximum(1 - y_true * y_pred, 0), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.choice([-1, 1], size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.hinge(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(),
    ...     np.mean(np.maximum(1. - y_true * y_pred, 0.), axis=-1))

    :cvar y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
    If binary (0 or 1) labels are provided they will be converted to -1 or 1.
    shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Hinge loss values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = "```K.mean(math_ops.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)```"


class kl_divergenceConfig(object):
    """
    Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

    `loss = y_true * log(y_true / y_pred)`

    See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float64)
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = tf.keras.backend.clip(y_true, 1e-7, 1)
    >>> y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.sum(y_true * np.log(y_true / y_pred), axis=-1))

    :cvar y_true: Tensor of true targets.
    :cvar y_pred: Tensor of predicted targets.
    :cvar return_type: A `Tensor` with loss. Defaults to ```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = (
        "```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```"
    )


class kldConfig(object):
    """
    Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

    `loss = y_true * log(y_true / y_pred)`

    See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float64)
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = tf.keras.backend.clip(y_true, 1e-7, 1)
    >>> y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.sum(y_true * np.log(y_true / y_pred), axis=-1))

    :cvar y_true: Tensor of true targets.
    :cvar y_pred: Tensor of predicted targets.
    :cvar return_type: A `Tensor` with loss. Defaults to ```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = (
        "```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```"
    )


class kullback_leibler_divergenceConfig(object):
    """
    Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

    `loss = y_true * log(y_true / y_pred)`

    See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float64)
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = tf.keras.backend.clip(y_true, 1e-7, 1)
    >>> y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.sum(y_true * np.log(y_true / y_pred), axis=-1))

    :cvar y_true: Tensor of true targets.
    :cvar y_pred: Tensor of predicted targets.
    :cvar return_type: A `Tensor` with loss. Defaults to ```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = (
        "```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```"
    )


class maeConfig(object):
    """
    Computes the mean absolute error between labels and predictions.

    `loss = mean(abs(y_true - y_pred), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.mean(np.abs(y_true - y_pred), axis=-1))

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.abs(y_pred - y_true), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = "```K.mean(math_ops.abs(y_pred - y_true), axis=-1)```"


class mapeConfig(object):
    """
    Computes the mean absolute percentage error between `y_true` and `y_pred`.

    `loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.random(size=(2, 3))
    >>> y_true = np.maximum(y_true, 1e-7)  # Prevent division by zero
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(),
    ...     100. * np.mean(np.abs((y_true - y_pred) / y_true), axis=-1))

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```(100.0 * K.mean(diff, axis=-1))```"""

    y_true = None
    y_pred = None
    return_type = "```(100.0 * K.mean(diff, axis=-1))```"


class mean_absolute_errorConfig(object):
    """
    Computes the mean absolute error between labels and predictions.

    `loss = mean(abs(y_true - y_pred), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.mean(np.abs(y_true - y_pred), axis=-1))

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.abs(y_pred - y_true), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = "```K.mean(math_ops.abs(y_pred - y_true), axis=-1)```"


class mean_absolute_percentage_errorConfig(object):
    """
    Computes the mean absolute percentage error between `y_true` and `y_pred`.

    `loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.random(size=(2, 3))
    >>> y_true = np.maximum(y_true, 1e-7)  # Prevent division by zero
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(),
    ...     100. * np.mean(np.abs((y_true - y_pred) / y_true), axis=-1))

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```(100.0 * K.mean(diff, axis=-1))```"""

    y_true = None
    y_pred = None
    return_type = "```(100.0 * K.mean(diff, axis=-1))```"


class mean_squared_errorConfig(object):
    """
    Computes the mean squared error between labels and predictions.

    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.

    `loss = mean(square(y_true - y_pred), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Mean squared error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = "```K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)```"


class mean_squared_logarithmic_errorConfig(object):
    """
    Computes the mean squared logarithmic error between `y_true` and `y_pred`.

    `loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = np.maximum(y_true, 1e-7)
    >>> y_pred = np.maximum(y_pred, 1e-7)
    >>> assert np.array_equal(
    ...     loss.numpy(),
    ...     np.mean(
    ...         np.square(np.log(y_true + 1.) - np.log(y_pred + 1.)), axis=-1))

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Mean squared logarithmic error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = (
        "```K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)```"
    )


class mseConfig(object):
    """
    Computes the mean squared error between labels and predictions.

    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.

    `loss = mean(square(y_true - y_pred), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Mean squared error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = "```K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)```"


class msleConfig(object):
    """
    Computes the mean squared logarithmic error between `y_true` and `y_pred`.

    `loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = np.maximum(y_true, 1e-7)
    >>> y_pred = np.maximum(y_pred, 1e-7)
    >>> assert np.array_equal(
    ...     loss.numpy(),
    ...     np.mean(
    ...         np.square(np.log(y_true + 1.) - np.log(y_pred + 1.)), axis=-1))

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Mean squared logarithmic error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = (
        "```K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)```"
    )


class poissonConfig(object):
    """
    Computes the Poisson loss between y_true and y_pred.

    The Poisson loss is the mean of the elements of the `Tensor`
    `y_pred - y_true * log(y_pred)`.

    Standalone usage:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.poisson(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_pred = y_pred + 1e-7
    >>> assert np.allclose(
    ...     loss.numpy(), np.mean(y_pred - y_true * np.log(y_pred), axis=-1),
    ...     atol=1e-5)

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Poisson loss value. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = (
        "```K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)```"
    )


class sparse_categorical_accuracyConfig(object):
    """
    Calculates how often predictions matches integer labels.

    Standalone usage:
    >>> y_true = [2, 1]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([0., 1.], dtype=float32)

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    :cvar y_true: Integer ground truth values.
    :cvar y_pred: The prediction values.
    :cvar return_type: Sparse categorical accuracy values. Defaults to ```math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())```"""

    y_true = None
    y_pred = None
    return_type = "```math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())```"


class sparse_categorical_crossentropyConfig(object):
    """
    Computes the sparse categorical crossentropy loss.

    Standalone usage:

    >>> y_true = [1, 2]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([0.0513, 2.303], dtype=float32)

    :cvar y_true: Ground truth values.
    :cvar y_pred: The predicted values.
    :cvar from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
    we assume that `y_pred` encodes a probability distribution. Defaults to False
    :cvar axis: (Optional)The dimension along which the entropy is
    computed. Defaults to -1
    :cvar return_type: Sparse categorical crossentropy loss value. Defaults to ```K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits,
    axis=axis)```"""

    y_true = None
    y_pred = None
    from_logits: bool = False
    axis: int = -1
    return_type = """```K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits,
    axis=axis)```"""


class sparse_top_k_categorical_accuracyConfig(object):
    """
     Computes how often integer targets are in the top `K` predictions.

     Standalone usage:
     >>> y_true = [2, 1]
     >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
     >>> m = tf.keras.metrics.sparse_top_k_categorical_accuracy(
     ...     y_true, y_pred, k=3)
     >>> assert m.shape == (2,)
     >>> m.numpy()
     array([1., 1.], dtype=float32)

     :cvar y_true: tensor of true targets.
     :cvar y_pred: tensor of predicted targets.
     :cvar k: (Optional) Number of top elements to look at for computing accuracy.
    . Defaults to 5
     :cvar return_type: Sparse top K categorical accuracy value. Defaults to ```math_ops.cast(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), k), K.
     floatx())```"""

    y_true = None
    y_pred = None
    k: int = 5
    return_type = """```math_ops.cast(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), k), K.
    floatx())```"""


class squared_hingeConfig(object):
    """
    Computes the squared hinge loss between `y_true` and `y_pred`.

    `loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)`

    Standalone usage:

    >>> y_true = np.random.choice([-1, 1], size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.squared_hinge(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> assert np.array_equal(
    ...     loss.numpy(),
    ...     np.mean(np.square(np.maximum(1. - y_true * y_pred, 0.)), axis=-1))

    :cvar y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
    If binary (0 or 1) labels are provided we will convert them to -1 or 1.
    shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Squared hinge loss values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.square(math_ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = "```K.mean(math_ops.square(math_ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)```"


class top_k_categorical_accuracyConfig(object):
    """
     Computes how often targets are in the top `K` predictions.

     Standalone usage:
     >>> y_true = [[0, 0, 1], [0, 1, 0]]
     >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
     >>> m = tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
     >>> assert m.shape == (2,)
     >>> m.numpy()
     array([1., 1.], dtype=float32)

     :cvar y_true: The ground truth values.
     :cvar y_pred: The prediction values.
     :cvar k: (Optional) Number of top elements to look at for computing accuracy.
    . Defaults to 5
     :cvar return_type: Top K categorical accuracy value. Defaults to ```math_ops.cast(nn.in_top_k(y_pred, math_ops.argmax(y_true, axis=-1), k), K.
     floatx())```"""

    y_true = None
    y_pred = None
    k: int = 5
    return_type = """```math_ops.cast(nn.in_top_k(y_pred, math_ops.argmax(y_true, axis=-1), k), K.
    floatx())```"""