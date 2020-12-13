""" Generated Callback config classes """
from __future__ import absolute_import, division, print_function

NoneType = type(None)


def binary_accuracyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Binary accuracy values. shape = `[batch_size, d0, .. dN-1]`
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Calculates how often predictions matches binary labels.

Standalone usage:
>>> y_true = [[1], [1], [0], [0]]
>>> y_pred = [[1], [1], [0], [0]]
>>> m = tf.keras.metrics.binary_accuracy(y_true, y_pred)
>>> assert m.shape == (4,)
>>> m.numpy()
array([1., 1., 1., 1.], dtype=float32)"""
    argument_parser.add_argument(
        "--y_true",
        type=float,
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
        default=0.5,
    )
    argument_parser.add_argument(
        "--y_pred",
        type=str,
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--threshold",
        type=float,
        help="""(Optional) Float representing the threshold for deciding whether
    prediction values are 1 or 0.""",
        required=True,
        default=0.5,
    )
    return (argument_parser, "```K.mean(math_ops.equal(y_true, y_pred), axis=-1)```")


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


def categorical_accuracyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Categorical accuracy values.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Calculates how often predictions matches one-hot labels.

Standalone usage:
>>> y_true = [[0, 0, 1], [0, 1, 0]]
>>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
>>> m = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
>>> assert m.shape == (2,)
>>> m.numpy()
array([0., 1.], dtype=float32)

You can provide logits of classes as `y_pred`, since argmax of
logits and probabilities are same."""
    argument_parser.add_argument(
        "--y_true", type=str, help="One-hot ground truth values.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", type=str, help="The prediction values.", required=True
    )
    return (
        argument_parser,
        """```math_ops.cast(math_ops.equal(math_ops.argmax(y_true, axis=-1), math_ops.
    argmax(y_pred, axis=-1)), K.floatx())```""",
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
>>> assert np.array_equal(
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
>>> assert np.array_equal(
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


def sparse_categorical_accuracyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Sparse categorical accuracy values.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Calculates how often predictions matches integer labels.

Standalone usage:
>>> y_true = [2, 1]
>>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
>>> m = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
>>> assert m.shape == (2,)
>>> m.numpy()
array([0., 1.], dtype=float32)

You can provide logits of classes as `y_pred`, since argmax of
logits and probabilities are same."""
    argument_parser.add_argument(
        "--y_true", type=str, help="Integer ground truth values.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", type=str, help="The prediction values.", required=True
    )
    return (
        argument_parser,
        "```math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())```",
    )


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
        "--y_pred",
        type=str,
        help="The predicted values.",
        required=True,
        default="```(-1)```",
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
        required=True,
        default=-1,
    )
    return (
        argument_parser,
        """```K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits,
    axis=axis)```""",
    )


def sparse_top_k_categorical_accuracyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Sparse top K categorical accuracy value.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes how often integer targets are in the top `K` predictions.

Standalone usage:
>>> y_true = [2, 1]
>>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
>>> m = tf.keras.metrics.sparse_top_k_categorical_accuracy(
...     y_true, y_pred, k=3)
>>> assert m.shape == (2,)
>>> m.numpy()
array([1., 1.], dtype=float32)"""
    argument_parser.add_argument(
        "--y_true", type=int, help="tensor of true targets.", required=True, default=5
    )
    argument_parser.add_argument(
        "--y_pred", type=str, help="tensor of predicted targets.", required=True
    )
    argument_parser.add_argument(
        "--k",
        type=int,
        help="""(Optional) Number of top elements to look at for computing accuracy.
   """,
        required=True,
        default=5,
    )
    return (
        argument_parser,
        """```math_ops.cast(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), k), K.
    floatx())```""",
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


def top_k_categorical_accuracyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Top K categorical accuracy value.
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes how often targets are in the top `K` predictions.

Standalone usage:
>>> y_true = [[0, 0, 1], [0, 1, 0]]
>>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
>>> m = tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
>>> assert m.shape == (2,)
>>> m.numpy()
array([1., 1.], dtype=float32)"""
    argument_parser.add_argument(
        "--y_true", type=int, help="The ground truth values.", required=True, default=5
    )
    argument_parser.add_argument(
        "--y_pred", type=str, help="The prediction values.", required=True
    )
    argument_parser.add_argument(
        "--k",
        type=int,
        help="""(Optional) Number of top elements to look at for computing accuracy.
   """,
        required=True,
        default=5,
    )
    return (
        argument_parser,
        """```math_ops.cast(nn.in_top_k(y_pred, math_ops.argmax(y_true, axis=-1), k), K.
    floatx())```""",
    )


__all__ = [
    "binary_accuracyConfig",
    "binary_crossentropyConfig",
    "categorical_accuracyConfig",
    "categorical_crossentropyConfig",
    "hingeConfig",
    "kl_divergenceConfig",
    "kldConfig",
    "kullback_leibler_divergenceConfig",
    "maeConfig",
    "mapeConfig",
    "mean_absolute_errorConfig",
    "mean_absolute_percentage_errorConfig",
    "mean_squared_errorConfig",
    "mean_squared_logarithmic_errorConfig",
    "mseConfig",
    "msleConfig",
    "poissonConfig",
    "sparse_categorical_accuracyConfig",
    "sparse_categorical_crossentropyConfig",
    "sparse_top_k_categorical_accuracyConfig",
    "squared_hingeConfig",
    "top_k_categorical_accuracyConfig",
]
