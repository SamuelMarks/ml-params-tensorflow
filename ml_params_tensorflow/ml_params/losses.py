""" Generated Loss CLI parsers """
from __future__ import absolute_import, division, print_function


def binary_crossentropyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the binary crossentropy loss.

Standalone usage:

>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
>>> loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss.numpy()
array([0.916 , 0.714], dtype=float32)"""
    argument_parser.add_argument(
        "--y_true",
        type=bool,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=int,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--from_logits",
        type=bool,
        help="""Whether `y_pred` is expected to be a logits tensor. By default,
    we assume that `y_pred` encodes a probability distribution.""",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--label_smoothing",
        type=int,
        help="Float in [0, 1]. If > `0` then smooth the labels.",
        required=True,
        default=0,
    )
    return (
        argument_parser,
        "```K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)```",
    )


def categorical_crossentropyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Categorical crossentropy loss value.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the categorical crossentropy loss.

Standalone usage:

>>> y_true = [[0, 1, 0], [0, 0, 1]]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss.numpy()
array([0.0513, 2.303], dtype=float32)"""
    argument_parser.add_argument(
        "--y_true",
        type=bool,
        help="Tensor of one-hot true targets.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=int,
        help="Tensor of predicted targets.",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--from_logits",
        type=bool,
        help="""Whether `y_pred` is expected to be a logits tensor. By default,
    we assume that `y_pred` encodes a probability distribution.""",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--label_smoothing",
        type=int,
        help="Float in [0, 1]. If > `0` then smooth the labels.",
        required=True,
        default=0,
    )
    return (
        argument_parser,
        "```K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)```",
    )


def categorical_hingeConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Categorical hinge loss values.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the categorical hinge loss between `y_true` and `y_pred`.

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
>>> assert np.array_equal(loss.numpy(), np.maximum(0., neg - pos + 1.))"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="The ground truth values. `y_true` values are expected to be 0 or 1.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred", type=str, help="The predicted values.", required=True
    )
    return argument_parser, "```math_ops.maximum(neg - pos + 1.0, zero)```"


def cosine_similarityConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Cosine similarity tensor.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the cosine similarity between labels and predictions.

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
array([-0., -0.999, 0.999], dtype=float32)"""
    argument_parser.add_argument(
        "--y_true", help="Tensor of true targets.", required=True, default="```(-1)```"
    )
    argument_parser.add_argument(
        "--y_pred", type=str, help="Tensor of predicted targets.", required=True
    )
    argument_parser.add_argument(
        "--axis",
        type=int,
        help="Axis along which to determine similarity.",
        required=True,
        default=-1,
    )
    return (argument_parser, "```(-math_ops.reduce_sum(y_true * y_pred, axis=axis))```")


def hingeConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Hinge loss values. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the hinge loss between `y_true` and `y_pred`.

`loss = mean(maximum(1 - y_true * y_pred, 0), axis=-1)`

Standalone usage:

>>> y_true = np.random.choice([-1, 1], size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.hinge(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> assert np.array_equal(
...     loss.numpy(),
...     np.mean(np.maximum(1. - y_true * y_pred, 0.), axis=-1))"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="""The ground truth values. `y_true` values are expected to be -1 or 1.
    If binary (0 or 1) labels are provided they will be converted to -1 or 1.
    shape = `[batch_size, d0, .. dN]`.""",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (
        argument_parser,
        "```K.mean(math_ops.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)```",
    )


def huberConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Tensor with one scalar loss entry per sample.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes Huber loss value.

For each value x in `error = y_true - y_pred`:

```
loss = 0.5 * x^2                  if |x| <= d
loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
```
where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss"""
    argument_parser.add_argument(
        "--y_true",
        type=float,
        help="tensor of true targets.",
        required=True,
        default=1.0,
    )
    argument_parser.add_argument(
        "--y_pred", type=str, help="tensor of predicted targets.", required=True
    )
    argument_parser.add_argument(
        "--delta",
        type=float,
        help="""A float, the point where the Huber loss function changes from a
    quadratic to linear.""",
        required=True,
        default=1.0,
    )
    return (
        argument_parser,
        """```K.mean(array_ops.where_v2(abs_error <= delta, half * math_ops.pow(error, 2),
    half * math_ops.pow(delta, 2) + delta * (abs_error - delta)), axis=-1)```""",
    )


def kldConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, A `Tensor` with loss.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

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
...     loss.numpy(), np.sum(y_true * np.log(y_true / y_pred), axis=-1))"""
    argument_parser.add_argument(
        "--y_true", type=str, help="Tensor of true targets.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", type=str, help="Tensor of predicted targets.", required=True
    )
    return (
        argument_parser,
        "```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```",
    )


def kl_divergenceConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, A `Tensor` with loss.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

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
...     loss.numpy(), np.sum(y_true * np.log(y_true / y_pred), axis=-1))"""
    argument_parser.add_argument(
        "--y_true", type=str, help="Tensor of true targets.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", type=str, help="Tensor of predicted targets.", required=True
    )
    return (
        argument_parser,
        "```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```",
    )


def kullback_leibler_divergenceConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, A `Tensor` with loss.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

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
...     loss.numpy(), np.sum(y_true * np.log(y_true / y_pred), axis=-1))"""
    argument_parser.add_argument(
        "--y_true", type=str, help="Tensor of true targets.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", type=str, help="Tensor of predicted targets.", required=True
    )
    return (
        argument_parser,
        "```math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)```",
    )


def logcoshConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Logcosh error values. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Logarithm of the hyperbolic cosine of the prediction error.

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
...     atol=1e-5)"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return argument_parser, "```K.mean(_logcosh(y_pred - y_true), axis=-1)```"


def LossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Loss base class.

To be implemented by subclasses:
* `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

Example subclass implementation:

```python
class MeanSquaredError(Loss):

  def call(self, y_true, y_pred):
    y_pred = tf.convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
```

When used with `tf.distribute.Strategy`, outside of built-in training loops
such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
types, and reduce losses explicitly in your training loop. Using 'AUTO' or
'SUM_OVER_BATCH_SIZE' will raise an error.

Please see this custom training [tutorial](
  https://www.tensorflow.org/tutorials/distribute/custom_training) for more
details on this.

You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
```python
with strategy.scope():
  loss_obj = tf.keras.losses.CategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)
  ....
  loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
          (1. / global_batch_size))
```"""
    argument_parser.add_argument(
        "--reduction",
        help="""(Optional) Type of `tf.keras.losses.Reduction` to apply to
    loss. Default value is `AUTO`. `AUTO` indicates that the reduction
    option will be determined by the usage context. For almost all cases
    thisWhen used with
    `tf.distribute.Strategy`, outside of built-in training loops such as
    `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
    will raise an error. Please see this custom training [tutorial](
      https://www.tensorflow.org/tutorials/distribute/custom_training)
    for more details.""",
        default="```losses_utils.ReductionV2```",
    )
    argument_parser.add_argument(
        "--name", type=str, help="Optional name for the op.", required=True
    )
    return argument_parser


def maeConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the mean absolute error between labels and predictions.

`loss = mean(abs(y_true - y_pred), axis=-1)`

Standalone usage:

>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> assert np.array_equal(
...     loss.numpy(), np.mean(np.abs(y_true - y_pred), axis=-1))"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (argument_parser, "```K.mean(math_ops.abs(y_pred - y_true), axis=-1)```")


def mapeConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the mean absolute percentage error between `y_true` and `y_pred`.

`loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`

Standalone usage:

>>> y_true = np.random.random(size=(2, 3))
>>> y_true = np.maximum(y_true, 1e-7)  # Prevent division by zero
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> assert np.array_equal(
...     loss.numpy(),
...     100. * np.mean(np.abs((y_true - y_pred) / y_true), axis=-1))"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return argument_parser, "```(100.0 * K.mean(diff, axis=-1))```"


def mean_absolute_errorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the mean absolute error between labels and predictions.

`loss = mean(abs(y_true - y_pred), axis=-1)`

Standalone usage:

>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> assert np.array_equal(
...     loss.numpy(), np.mean(np.abs(y_true - y_pred), axis=-1))"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (argument_parser, "```K.mean(math_ops.abs(y_pred - y_true), axis=-1)```")


def mean_absolute_percentage_errorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the mean absolute percentage error between `y_true` and `y_pred`.

`loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`

Standalone usage:

>>> y_true = np.random.random(size=(2, 3))
>>> y_true = np.maximum(y_true, 1e-7)  # Prevent division by zero
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> assert np.array_equal(
...     loss.numpy(),
...     100. * np.mean(np.abs((y_true - y_pred) / y_true), axis=-1))"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return argument_parser, "```(100.0 * K.mean(diff, axis=-1))```"


def mean_squared_errorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the mean squared error between labels and predictions.

After computing the squared distance between the inputs, the mean value over
the last dimension is returned.

`loss = mean(square(y_true - y_pred), axis=-1)`

Standalone usage:

>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> assert np.array_equal(
...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (
        argument_parser,
        "```K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)```",
    )


def mean_squared_logarithmic_errorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Mean squared logarithmic error values. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

`loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)`

Standalone usage:

>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> y_true = np.maximum(y_true, 1e-7)
>>> y_pred = np.maximum(y_pred, 1e-7)
>>> assert np.allclose(
...     loss.numpy(),
...     np.mean(
...         np.square(np.log(y_true + 1.) - np.log(y_pred + 1.)), axis=-1))"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (
        argument_parser,
        "```K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)```",
    )


def mseConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the mean squared error between labels and predictions.

After computing the squared distance between the inputs, the mean value over
the last dimension is returned.

`loss = mean(square(y_true - y_pred), axis=-1)`

Standalone usage:

>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> assert np.array_equal(
...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (
        argument_parser,
        "```K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)```",
    )


def msleConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Mean squared logarithmic error values. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

`loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)`

Standalone usage:

>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> y_true = np.maximum(y_true, 1e-7)
>>> y_pred = np.maximum(y_pred, 1e-7)
>>> assert np.allclose(
...     loss.numpy(),
...     np.mean(
...         np.square(np.log(y_true + 1.) - np.log(y_pred + 1.)), axis=-1))"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (
        argument_parser,
        "```K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)```",
    )


def poissonConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Poisson loss value. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the Poisson loss between y_true and y_pred.

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
...     atol=1e-5)"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (
        argument_parser,
        "```K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)```",
    )


def ReductionConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Types of loss reduction.

Contains the following values:

* `AUTO`: Indicates that the reduction option will be determined by the usage
   context. For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. When
   used with `tf.distribute.Strategy`, outside of built-in training loops such
   as `tf.keras` `compile` and `fit`, we expect reduction value to be
   `SUM` or `NONE`. Using `AUTO` in that case will raise an error.
* `NONE`: Weighted losses with one dimension reduced (axis=-1, or axis
   specified by loss function). When this reduction type used with built-in
   Keras training loops like `fit`/`evaluate`, the unreduced vector loss is
   passed to the optimizer but the reported loss will be a scalar value.
* `SUM`: Scalar sum of weighted losses.
* `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
   This reduction type is not supported when used with
   `tf.distribute.Strategy` outside of built-in training loops like `tf.keras`
   `compile`/`fit`.

   You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
   ```
   with strategy.scope():
     loss_obj = tf.keras.losses.CategoricalCrossentropy(
         reduction=tf.keras.losses.Reduction.NONE)
     ....
     loss = tf.reduce_sum(loss_obj(labels, predictions)) *
         (1. / global_batch_size)
   ```

Please see the
[custom training guide](https://www.tensorflow.org/tutorials/distribute/custom_training)  # pylint: disable=line-too-long
for more details on this."""
    argument_parser.add_argument("--AUTO", type=str, required=True, default="auto")
    argument_parser.add_argument("--NONE", type=str, required=True, default="none")
    argument_parser.add_argument("--SUM", type=str, required=True, default="sum")
    argument_parser.add_argument(
        "--SUM_OVER_BATCH_SIZE", type=str, required=True, default="sum_over_batch_size"
    )
    return argument_parser


def sparse_categorical_crossentropyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Sparse categorical crossentropy loss value.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the sparse categorical crossentropy loss.

Standalone usage:

>>> y_true = [1, 2]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss.numpy()
array([0.0513, 2.303], dtype=float32)"""
    argument_parser.add_argument(
        "--y_true", type=bool, help="Ground truth values.", required=True, default=False
    )
    argument_parser.add_argument(
        "--y_pred", help="The predicted values.", required=True, default="```(-1)```"
    )
    argument_parser.add_argument(
        "--from_logits",
        type=bool,
        help="""Whether `y_pred` is expected to be a logits tensor. By default,
    we assume that `y_pred` encodes a probability distribution.""",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--axis",
        type=int,
        help="""(Optional)The dimension along which the entropy is
    computed.""",
        default=-1,
    )
    return (
        argument_parser,
        """```K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits,
    axis=axis)```""",
    )


def squared_hingeConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Squared hinge loss values. shape = `[batch_size, d0, .. dN-1]`.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the squared hinge loss between `y_true` and `y_pred`.

`loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)`

Standalone usage:

>>> y_true = np.random.choice([-1, 1], size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.squared_hinge(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> assert np.array_equal(
...     loss.numpy(),
...     np.mean(np.square(np.maximum(1. - y_true * y_pred, 0.)), axis=-1))"""
    argument_parser.add_argument(
        "--y_true",
        type=str,
        help="""The ground truth values. `y_true` values are expected to be -1 or 1.
    If binary (0 or 1) labels are provided we will convert them to -1 or 1.
    shape = `[batch_size, d0, .. dN]`.""",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (
        argument_parser,
        "```K.mean(math_ops.square(math_ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)```",
    )


__all__ = [
    "binary_crossentropyConfig",
    "categorical_crossentropyConfig",
    "categorical_hingeConfig",
    "cosine_similarityConfig",
    "hingeConfig",
    "huberConfig",
    "kldConfig",
    "kl_divergenceConfig",
    "kullback_leibler_divergenceConfig",
    "logcoshConfig",
    "LossConfig",
    "maeConfig",
    "mapeConfig",
    "mean_absolute_errorConfig",
    "mean_absolute_percentage_errorConfig",
    "mean_squared_errorConfig",
    "mean_squared_logarithmic_errorConfig",
    "mseConfig",
    "msleConfig",
    "poissonConfig",
    "ReductionConfig",
    "sparse_categorical_crossentropyConfig",
    "squared_hingeConfig",
]
