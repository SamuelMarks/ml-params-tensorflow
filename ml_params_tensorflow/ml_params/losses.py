""" Generated Callback config classes """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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


class categorical_hingeConfig(object):
    """
    Computes the categorical hinge loss between `y_true` and `y_pred`.

    `loss = maximum(neg - pos + 1, 0)`
    where `neg=maximum((1-y_true)*y_pred) and pos=sum(y_true*y_pred)`

    Standalone usage:

    >>> y_true = np.random.randint(0, 3, size=(2,))
    >>> y_true = tf.keras.utils.to_categorical(y_true, num_classes=3)
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.categorical_hinge(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> pos = np.sum(y_true * y_pred, axis=-1)
    >>> neg = np.amax((1. - y_true) * y_pred, axis=-1)
    >>> assert np.array_equal(loss.numpy(), np.maximum(0., neg - pos + 1.))

    :cvar y_true: The ground truth values. `y_true` values are expected to be 0 or 1.
    :cvar y_pred: The predicted values.
    :cvar return_type: Categorical hinge loss values. Defaults to ```math_ops.maximum(neg - pos + 1.0, zero)```"""

    y_true = None
    y_pred = None
    return_type = "```math_ops.maximum(neg - pos + 1.0, zero)```"


class cosine_similarityConfig(object):
    """
    Computes the cosine similarity between labels and predictions.

    Note that it is a number between -1 and 1. When it is a negative number
    between -1 and 0, 0 indicates orthogonality and values closer to -1
    indicate greater similarity. The values closer to 1 indicate greater
    dissimilarity. This makes it usable as a loss function in a setting
    where you try to maximize the proximity between predictions and
    targets. If either `y_true` or `y_pred` is a zero vector, cosine
    similarity will be 0 regardless of the proximity between predictions
    and targets.

    `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`

    Standalone usage:

    >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
    >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
    >>> loss = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=1)
    >>> loss.numpy()
    array([-0., -0.999, 0.999], dtype=float32)

    :cvar y_true: Tensor of true targets.
    :cvar y_pred: Tensor of predicted targets.
    :cvar axis: Axis along which to determine similarity. Defaults to -1
    :cvar return_type: Cosine similarity tensor. Defaults to ```(-math_ops.reduce_sum(y_true * y_pred, axis=axis))```"""

    y_true = None
    y_pred = None
    axis: int = -1
    return_type = "```(-math_ops.reduce_sum(y_true * y_pred, axis=axis))```"


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


class log_coshConfig(object):
    """
    Logarithm of the hyperbolic cosine of the prediction error.

    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.

    Standalone usage:

    >>> y_true = np.random.random(size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.logcosh(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> x = y_pred - y_true
    >>> assert np.allclose(
    ...     loss.numpy(),
    ...     np.mean(x + np.log(np.exp(-2. * x) + 1.) - math_ops.log(2.), axis=-1),
    ...     atol=1e-5)

    :cvar y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    :cvar y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    :cvar return_type: Logcosh error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```(x + nn.softplus(-2.0 * x) - math_ops.cast(math_ops.log(2.0), x.dtype))```"""

    y_true = None
    y_pred = None
    return_type = (
        "```(x + nn.softplus(-2.0 * x) - math_ops.cast(math_ops.log(2.0), x.dtype))```"
    )


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


__all__ = [
    "binary_crossentropyConfig",
    "categorical_crossentropyConfig",
    "categorical_hingeConfig",
    "cosine_similarityConfig",
    "kl_divergenceConfig",
    "kullback_leibler_divergenceConfig",
    "log_coshConfig",
    "mean_absolute_errorConfig",
    "mean_absolute_percentage_errorConfig",
    "mean_squared_errorConfig",
    "mean_squared_logarithmic_errorConfig",
    "sparse_categorical_crossentropyConfig",
    "squared_hingeConfig",
]
