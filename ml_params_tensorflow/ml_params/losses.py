"""" Loss config classes. Generated by:
```
import ast
from ast import Import, ImportFrom
from inspect import getfile

from doctrans import parse, emit
from doctrans.source_transformer import to_code
from doctrans.ast_utils import get_at_root
from ml_params_tensorflow.ml_params.type_generators import exposed_loss

import tensorflow as tf

with open(getfile(tf.keras.losses.Loss), "rt") as f:
    imports = "".join(
        map(to_code, get_at_root(ast.parse(f.read()), (Import, ImportFrom)))
    )

content = "import tensorflow as tf\n{}\n{}".format(
    imports,  # TODO: Optimize imports programatically (rather than just with IDE)
    "\n\n".join(
        to_code(
            emit.class_(
                parse.class_(obj),
                emit_call=True,
                class_bases=("tf.keras.losses.Loss",),
                class_name="{name}Config".format(name=name),
            )
        )
        for name, obj in exposed_loss.items()
    ),
)

with open("losses.py", "a") as f:
    f.write(content)
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


class binary_crossentropyConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```(y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing)```"""

    y_true = None
    y_pred = None
    from_logits: bool = False
    label_smoothing: int = 0
    return_type = "```(y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing)```"

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)
        self.label_smoothing = ops.convert_to_tensor_v2(
            self.label_smoothing, dtype=K.floatx()
        )

        def _smooth_labels():
            return (
                self.y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )

        self.y_true = smart_cond.smart_cond(
            self.label_smoothing, _smooth_labels, lambda: self.y_true
        )
        return K.mean(
            K.binary_crossentropy(
                self.y_true, self.y_pred, from_logits=self.from_logits
            ),
            axis=-1,
        )


class categorical_crossentropyConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   Categorical crossentropy loss value. Defaults to ```(y_true * (1.0 - label_smoothing) + label_smoothing / num_classes)```"""

    y_true = None
    y_pred = None
    from_logits: bool = False
    label_smoothing: int = 0
    return_type = (
        "```(y_true * (1.0 - label_smoothing) + label_smoothing / num_classes)```"
    )

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)
        self.label_smoothing = ops.convert_to_tensor_v2(
            self.label_smoothing, dtype=K.floatx()
        )

        def _smooth_labels():
            num_classes = math_ops.cast(
                array_ops.shape(self.y_true)[-1], self.y_pred.dtype
            )
            return (
                self.y_true * (1.0 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )

        self.y_true = smart_cond.smart_cond(
            self.label_smoothing, _smooth_labels, lambda: self.y_true
        )
        return K.categorical_crossentropy(
            self.y_true, self.y_pred, from_logits=self.from_logits
        )


class categorical_hingeConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   Categorical hinge loss values. Defaults to ```math_ops.maximum(neg - pos + 1.0, zero)```"""

    y_true = None
    y_pred = None
    return_type = "```math_ops.maximum(neg - pos + 1.0, zero)```"

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)
        pos = math_ops.reduce_sum(self.y_true * self.y_pred, axis=-1)
        neg = math_ops.reduce_max((1.0 - self.y_true) * self.y_pred, axis=-1)
        zero = math_ops.cast(0.0, self.y_pred.dtype)
        return math_ops.maximum(neg - pos + 1.0, zero)


class cosine_similarityConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   Cosine similarity tensor. Defaults to ```(-math_ops.reduce_sum(y_true * y_pred, axis=axis))```"""

    y_true = None
    y_pred = None
    axis: int = -1
    return_type = "```(-math_ops.reduce_sum(y_true * y_pred, axis=axis))```"

    def __call__(self):
        self.y_true = nn.l2_normalize(self.y_true, axis=self.axis)
        self.y_pred = nn.l2_normalize(self.y_pred, axis=self.axis)
        return -math_ops.reduce_sum(self.y_true * self.y_pred, axis=self.axis)


class kl_divergenceConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   A `Tensor` with loss. Defaults to ```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = (
        "```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```"
    )

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)
        self.y_true = K.clip(self.y_true, K.epsilon(), 1)
        self.y_pred = K.clip(self.y_pred, K.epsilon(), 1)
        return math_ops.reduce_sum(
            self.y_true * math_ops.log(self.y_true / self.y_pred), axis=-1
        )


class kullback_leibler_divergenceConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   A `Tensor` with loss. Defaults to ```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = (
        "```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```"
    )

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)
        self.y_true = K.clip(self.y_true, K.epsilon(), 1)
        self.y_pred = K.clip(self.y_pred, K.epsilon(), 1)
        return math_ops.reduce_sum(
            self.y_true * math_ops.log(self.y_true / self.y_pred), axis=-1
        )


class log_coshConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   Logcosh error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```(x + nn.softplus(-2.0 * x) - math_ops.cast(math_ops.log(2.0), x.dtype))```"""

    y_true = None
    y_pred = None
    return_type = (
        "```(x + nn.softplus(-2.0 * x) - math_ops.cast(math_ops.log(2.0), x.dtype))```"
    )

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)

        def _logcosh(x):
            return x + nn.softplus(-2.0 * x) - math_ops.cast(math_ops.log(2.0), x.dtype)

        return K.mean(_logcosh(self.y_pred - self.y_true), axis=-1)


class mean_absolute_errorConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.abs(y_pred - y_true), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = "```K.mean(math_ops.abs(y_pred - y_true), axis=-1)```"

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)
        return K.mean(math_ops.abs(self.y_pred - self.y_true), axis=-1)


class mean_absolute_percentage_errorConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```(100.0 * K.mean(diff, axis=-1))```"""

    y_true = None
    y_pred = None
    return_type = "```(100.0 * K.mean(diff, axis=-1))```"

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)
        diff = math_ops.abs(
            (self.y_true - self.y_pred)
            / K.maximum(math_ops.abs(self.y_true), K.epsilon())
        )
        return 100.0 * K.mean(diff, axis=-1)


class mean_squared_errorConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   Mean squared error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = "```K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)```"

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)
        return K.mean(math_ops.squared_difference(self.y_pred, self.y_true), axis=-1)


class mean_squared_logarithmic_errorConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   Mean squared logarithmic error values. shape = `[batch_size, d0, .. dN-1]`. Defaults to ```K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)```"""

    y_true = None
    y_pred = None
    return_type = (
        "```K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)```"
    )

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)
        first_log = math_ops.log(K.maximum(self.y_pred, K.epsilon()) + 1.0)
        second_log = math_ops.log(K.maximum(self.y_true, K.epsilon()) + 1.0)
        return K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)


class sparse_categorical_crossentropyConfig(tf.keras.losses.Loss):
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
    :cvar return_type:   Sparse categorical crossentropy loss value. Defaults to ```K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits,
    axis=axis)```"""

    y_true = None
    y_pred = None
    from_logits: bool = False
    axis: int = -1
    return_type = """```K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits,
    axis=axis)```"""

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)
        return K.sparse_categorical_crossentropy(
            self.y_true, self.y_pred, from_logits=self.from_logits, axis=self.axis
        )


class squared_hingeConfig(tf.keras.losses.Loss):
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
    :cvar return_type: None"""

    y_true = None
    y_pred = None
    return_type = "```K.mean(math_ops.square(math_ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)```"

    def __call__(self):
        self.y_pred = ops.convert_to_tensor_v2(self.y_pred)
        self.y_true = math_ops.cast(self.y_true, self.y_pred.dtype)
        self.y_true = _maybe_convert_labels(self.y_true)
        return K.mean(
            math_ops.square(math_ops.maximum(1.0 - self.y_true * self.y_pred, 0.0)),
            axis=-1,
        )