""" Generated Metric CLI parsers """


def AUCConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the approximate AUC (Area under the curve) via a Riemann sum.

This metric creates four local variables, `true_positives`, `true_negatives`,
`false_positives` and `false_negatives` that are used to compute the AUC.
To discretize the AUC curve, a linearly spaced set of thresholds is used to
compute pairs of recall and precision values. The area under the ROC-curve is
therefore computed using the height of the recall values by the false positive
rate, while the area under the PR-curve is the computed using the height of
the precision values by the recall.

This value is ultimately returned as `auc`, an idempotent operation that
computes the area under a discretized curve of precision versus recall values
(computed using the aforementioned variables). The `num_thresholds` variable
controls the degree of discretization with larger numbers of thresholds more
closely approximating the true AUC. The quality of the approximation may vary
dramatically depending on `num_thresholds`. The `thresholds` parameter can be
used to manually specify thresholds which split the predictions more evenly.

For best results, `predictions` should be distributed approximately uniformly
in the range [0, 1] and not peaked around 0 or 1. The quality of the AUC
approximation may be poor if this is not the case. Setting `summation_method`
to 'minoring' or 'majoring' can help quantify the error in the approximation
by providing lower or upper bound estimate of the AUC.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.


Standalone usage:

>>> m = tf.keras.metrics.AUC(num_thresholds=3)
>>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
>>> # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
>>> # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
>>> # recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
>>> # auc = ((((1+0.5)/2)*(1-0))+ (((0.5+0)/2)*(0-0))) = 0.75
>>> m.result().numpy()
0.75

>>> m.reset_states()
>>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
...                sample_weight=[1, 0, 0, 1])
>>> m.result().numpy()
1.0

Usage with `compile()` API:

```python
model.compile(optimizer='sgd', loss='mse', metrics=[tf.keras.metrics.AUC()])
```"""
    argument_parser.add_argument(
        "--num_thresholds",
        type=int,
        help="(Optional)The number of thresholds to use when discretizing the roc curve. Values must be > 1.",
        default=200,
    )
    argument_parser.add_argument(
        "--curve",
        help="""(Optional) Specifies the name of the curve to be computed, 'ROC' [default] or 'PR' for the
Precision-Recall-curve.""",
    )
    argument_parser.add_argument(
        "--summation_method",
        help="""(Optional) Specifies the [Riemann summation method]( https://en.wikipedia.org/wiki/Riemann_sum)
used. 'interpolation' (default) applies mid-point summation scheme for `ROC`. For PR-AUC,
interpolates (true/false) positives but not the ratio that is precision (see Davis & Goadrich 2006
for details); 'minoring' applies left summation for increasing intervals and right summation for
decreasing intervals; 'majoring' does the opposite.""",
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance."
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result."
    )
    argument_parser.add_argument(
        "--thresholds",
        help="""(Optional) A list of floating point values to use as the thresholds for discretizing the curve. If
set, the `num_thresholds` parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
equal to {-epsilon, 1+epsilon} for a small positive epsilon value will be automatically included
with these to correctly handle predictions equal to exactly 0 or 1.""",
    )
    argument_parser.add_argument(
        "--multi_label",
        help="""boolean indicating whether multilabel data should be treated as such, wherein AUC is computed
separately for each label and then averaged across labels, or (when False) if the data should be
flattened into a single label before AUC computation. In the latter case, when multilabel data is
passed to AUC, each label-prediction pair is treated as an individual data point. Should be set to
False for multi-class data.""",
    )
    argument_parser.add_argument(
        "--label_weights",
        help="""(optional) list, array, or tensor of non-negative weights used to compute AUCs for multilabel data.
When `multi_label` is True, the weights are applied to the individual label AUCs when they are
averaged to produce the multi-label AUC. When it's False, they are used to weight the individual
label predictions in computing the confusion matrix on the flattened data. Note that this is unlike
class_weights in that class_weights weights the example depending on the value of its label, whereas
label_weights depends only on the index of that label before flattening; therefore `label_weights`
should not be used for multi-class data.""",
    )
    return argument_parser


def binary_accuracyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--threshold",
        type=float,
        help="(Optional) Float representing the threshold for deciding whether prediction values are 1 or 0.",
        required=True,
        default=0.5,
    )
    return (argument_parser, "```K.mean(math_ops.equal(y_true, y_pred), axis=-1)```")


def binary_crossentropyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--from_logits",
        type=bool,
        help="""Whether `y_pred` is expected to be a logits tensor. By default, we assume that `y_pred` encodes a
probability distribution.""",
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

    :returns: argument_parser
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
        "--y_true", help="One-hot ground truth values.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", help="The prediction values.", required=True
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

    :returns: argument_parser
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
        "--y_true", help="Tensor of one-hot true targets.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", help="Tensor of predicted targets.", required=True
    )
    argument_parser.add_argument(
        "--from_logits",
        type=bool,
        help="""Whether `y_pred` is expected to be a logits tensor. By default, we assume that `y_pred` encodes a
probability distribution.""",
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


def CategoricalHingeConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the categorical hinge metric between `y_true` and `y_pred`.


Standalone usage:

>>> m = tf.keras.metrics.CategoricalHinge()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
>>> m.result().numpy()
1.4000001

>>> m.reset_states()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
...                sample_weight=[1, 0])
>>> m.result().numpy()
1.2

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.CategoricalHinge()])
```"""
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance.", required=True
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result.", required=True
    )
    return argument_parser


def CosineSimilarityConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the cosine similarity between the labels and predictions.

`cosine similarity = (a . b) / ||a|| ||b||`

See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

This metric keeps the average cosine similarity between `predictions` and
`labels` over a stream of data.


Standalone usage:

>>> # l2_norm(y_true) = [[0., 1.], [1./1.414], 1./1.414]]]
>>> # l2_norm(y_pred) = [[1., 0.], [1./1.414], 1./1.414]]]
>>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
>>> # result = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
>>> #        = ((0. + 0.) +  (0.5 + 0.5)) / 2
>>> m = tf.keras.metrics.CosineSimilarity(axis=1)
>>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]])
>>> m.result().numpy()
0.49999997

>>> m.reset_states()
>>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]],
...                sample_weight=[0.3, 0.7])
>>> m.result().numpy()
0.6999999

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
```"""
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance.", required=True
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result.", required=True
    )
    argument_parser.add_argument(
        "--axis",
        type=float,
        help="(Optional)The dimension along which the cosine similarity is computed.",
        default=-1.0,
    )
    return argument_parser


def FalseNegativesConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Calculates the number of false negatives.

If `sample_weight` is given, calculates the sum of the weights of
false negatives. This metric creates one local variable, `accumulator`
that is used to keep track of the number of false negatives.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.


Standalone usage:

>>> m = tf.keras.metrics.FalseNegatives()
>>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
>>> m.result().numpy()
2.0

>>> m.reset_states()
>>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0], sample_weight=[0, 0, 1, 0])
>>> m.result().numpy()
1.0

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.FalseNegatives()])
```"""
    argument_parser.add_argument(
        "--thresholds",
        type=float,
        help="""(Optional)A float value or a python list/tuple of float threshold values in [0, 1]. A threshold is
compared with prediction values to determine the truth value of predictions (i.e., above the
threshold is `true`, below is `false`). One metric value is generated for each threshold value.""",
        default=0.5,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance."
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result."
    )
    return argument_parser


def FalsePositivesConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Calculates the number of false positives.

If `sample_weight` is given, calculates the sum of the weights of
false positives. This metric creates one local variable, `accumulator`
that is used to keep track of the number of false positives.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.


Standalone usage:

>>> m = tf.keras.metrics.FalsePositives()
>>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1])
>>> m.result().numpy()
2.0

>>> m.reset_states()
>>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1], sample_weight=[0, 0, 1, 0])
>>> m.result().numpy()
1.0

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.FalsePositives()])
```"""
    argument_parser.add_argument(
        "--thresholds",
        type=float,
        help="""(Optional)A float value or a python list/tuple of float threshold values in [0, 1]. A threshold is
compared with prediction values to determine the truth value of predictions (i.e., above the
threshold is `true`, below is `false`). One metric value is generated for each threshold value.""",
        default=0.5,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance."
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result."
    )
    return argument_parser


def hingeConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        help="""The ground truth values. `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
provided they will be converted to -1 or 1. shape = `[batch_size, d0, .. dN]`.""",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (
        argument_parser,
        "```K.mean(math_ops.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)```",
    )


def kldConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        "--y_true", help="Tensor of true targets.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", help="Tensor of predicted targets.", required=True
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

    :returns: argument_parser
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
        "--y_true", help="Tensor of true targets.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", help="Tensor of predicted targets.", required=True
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

    :returns: argument_parser
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
        "--y_true", help="Tensor of true targets.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", help="Tensor of predicted targets.", required=True
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

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return argument_parser, "```K.mean(_logcosh(y_pred - y_true), axis=-1)```"


def LogCoshErrorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the logarithm of the hyperbolic cosine of the prediction error.

`logcosh = log((exp(x) + exp(-x))/2)`, where x is the error (y_pred - y_true)


Standalone usage:

>>> m = tf.keras.metrics.LogCoshError()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
>>> m.result().numpy()
0.10844523

>>> m.reset_states()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
...                sample_weight=[1, 0])
>>> m.result().numpy()
0.21689045

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.LogCoshError()])
```"""
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance.", required=True
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result.", required=True
    )
    return argument_parser


def maeConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (argument_parser, "```K.mean(math_ops.abs(y_pred - y_true), axis=-1)```")


def mapeConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return argument_parser, "```(100.0 * K.mean(diff, axis=-1))```"


def MeanConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the (weighted) mean of the given values.

For example, if values is [1, 3, 5, 7] then the mean is 4.
If the weights were specified as [1, 1, 0, 0] then the mean would be 2.

This metric creates two variables, `total` and `count` that are used to
compute the average of `values`. This average is ultimately returned as `mean`
which is an idempotent operation that simply divides `total` by `count`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.


Standalone usage:

>>> m = tf.keras.metrics.Mean()
>>> m.update_state([1, 3, 5, 7])
>>> m.result().numpy()
4.0
>>> m.reset_states()
>>> m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
>>> m.result().numpy()
2.0

Usage with `compile()` API:

```python
model.add_metric(tf.keras.metrics.Mean(name='mean_1')(outputs))
model.compile(optimizer='sgd', loss='mse')
```"""
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance.", required=True
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result.", required=True
    )
    return argument_parser


def mean_absolute_errorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (argument_parser, "```K.mean(math_ops.abs(y_pred - y_true), axis=-1)```")


def mean_absolute_percentage_errorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return argument_parser, "```(100.0 * K.mean(diff, axis=-1))```"


def MeanIoUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the mean Intersection-Over-Union metric.

Mean Intersection-Over-Union is a common evaluation metric for semantic image
segmentation, which first computes the IOU for each semantic class and then
computes the average over classes. IOU is defined as follows:
  IOU = true_positive / (true_positive + false_positive + false_negative).
The predictions are accumulated in a confusion matrix, weighted by
`sample_weight` and the metric is then calculated from it.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.


Standalone usage:

>>> # cm = [[1, 1],
>>> #        [1, 1]]
>>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
>>> # iou = true_positives / (sum_row + sum_col - true_positives))
>>> # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
>>> m = tf.keras.metrics.MeanIoU(num_classes=2)
>>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
>>> m.result().numpy()
0.33333334

>>> m.reset_states()
>>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
...                sample_weight=[0.3, 0.3, 0.3, 0.1])
>>> m.result().numpy()
0.23809525

Usage with `compile()` API:

```python
model.compile(
  optimizer='sgd',
  loss='mse',
  metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
```"""
    argument_parser.add_argument(
        "--num_classes",
        help="""The possible number of labels the prediction task can have. This value must be provided, since a
confusion matrix of dimension = [num_classes, num_classes] will be allocated.""",
        required=True,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance.", required=True
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result.", required=True
    )
    return argument_parser


def MeanRelativeErrorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the mean relative error by normalizing with the given values.

This metric creates two local variables, `total` and `count` that are used to
compute the mean relative error. This is weighted by `sample_weight`, and
it is ultimately returned as `mean_relative_error`:
an idempotent operation that simply divides `total` by `count`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.


Standalone usage:

>>> m = tf.keras.metrics.MeanRelativeError(normalizer=[1, 3, 2, 3])
>>> m.update_state([1, 3, 2, 3], [2, 4, 6, 8])

>>> # metric = mean(|y_pred - y_true| / normalizer)
>>> #        = mean([1, 1, 4, 5] / [1, 3, 2, 3]) = mean([1, 1/3, 2, 5/3])
>>> #        = 5/4 = 1.25
>>> m.result().numpy()
1.25

Usage with `compile()` API:

```python
model.compile(
  optimizer='sgd',
  loss='mse',
  metrics=[tf.keras.metrics.MeanRelativeError(normalizer=[1, 3])])
```"""
    argument_parser.add_argument(
        "--normalizer",
        help="The normalizer values with same shape as predictions.",
        required=True,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance.", required=True
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result.", required=True
    )
    return argument_parser


def mean_squared_errorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
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

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (
        argument_parser,
        "```K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)```",
    )


def MeanTensorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the element-wise (weighted) mean of the given tensors.

`MeanTensor` returns a tensor with the same shape of the input tensors. The
mean value is updated by keeping local variables `total` and `count`. The
`total` tracks the sum of the weighted values, and `count` stores the sum of
the weighted counts.


Standalone usage:

>>> m = tf.keras.metrics.MeanTensor()
>>> m.update_state([0, 1, 2, 3])
>>> m.update_state([4, 5, 6, 7])
>>> m.result().numpy()
array([2., 3., 4., 5.], dtype=float32)

>>> m.update_state([12, 10, 8, 6], sample_weight= [0, 0.2, 0.5, 1])
>>> m.result().numpy()
array([2.       , 3.6363635, 4.8      , 5.3333335], dtype=float32)"""
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance.", required=True
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result.", required=True
    )
    return argument_parser


def mseConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
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

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
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

    :returns: argument_parser
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
        help="Ground truth values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (
        argument_parser,
        "```K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)```",
    )


def PrecisionConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the precision of the predictions with respect to the labels.

The metric creates two local variables, `true_positives` and `false_positives`
that are used to compute the precision. This value is ultimately returned as
`precision`, an idempotent operation that simply divides `true_positives`
by the sum of `true_positives` and `false_positives`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `top_k` is set, we'll calculate precision as how often on average a class
among the top-k classes with the highest predicted values of a batch entry is
correct and can be found in the label for that entry.

If `class_id` is specified, we calculate precision by considering only the
entries in the batch for which `class_id` is above the threshold and/or in the
top-k highest predictions, and computing the fraction of them for which
`class_id` is indeed a correct label.


Standalone usage:

>>> m = tf.keras.metrics.Precision()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
>>> m.result().numpy()
0.6666667

>>> m.reset_states()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
>>> m.result().numpy()
1.0

>>> # With top_k=2, it will calculate precision over y_true[:2] and y_pred[:2]
>>> m = tf.keras.metrics.Precision(top_k=2)
>>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
>>> m.result().numpy()
0.0

>>> # With top_k=4, it will calculate precision over y_true[:4] and y_pred[:4]
>>> m = tf.keras.metrics.Precision(top_k=4)
>>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
>>> m.result().numpy()
0.5

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.Precision()])
```"""
    argument_parser.add_argument(
        "--thresholds",
        help="""(Optional) A float value or a python list/tuple of float threshold values in [0, 1]. A threshold is
compared with prediction values to determine the truth value of predictions (i.e., above the
threshold is `true`, below is `false`). One metric value is generated for each threshold value. If
neither thresholds nor top_k are set, the default is to calculate precision with `thresholds=0.5`.""",
        required=True,
    )
    argument_parser.add_argument(
        "--top_k",
        help="""(Optional) Unset by default. An int value specifying the top-k predictions to consider when
calculating precision.""",
        required=True,
    )
    argument_parser.add_argument(
        "--class_id",
        help="""(Optional) Integer class ID for which we want binary metrics. This must be in the half-open interval
`[0, num_classes)`, where `num_classes` is the last dimension of predictions.""",
        required=True,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance.", required=True
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result.", required=True
    )
    return argument_parser


def PrecisionAtRecallConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes best precision where recall is >= specified value.

This metric creates four local variables, `true_positives`, `true_negatives`,
`false_positives` and `false_negatives` that are used to compute the
precision at the given recall. The threshold for the given recall
value is computed and used to evaluate the corresponding precision.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.


Standalone usage:

>>> m = tf.keras.metrics.PrecisionAtRecall(0.5)
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
>>> m.result().numpy()
0.5

>>> m.reset_states()
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
...                sample_weight=[2, 2, 2, 1, 1])
>>> m.result().numpy()
0.33333333

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
```"""
    argument_parser.add_argument(
        "--recall", help="A scalar value in range `[0, 1]`.", required=True
    )
    argument_parser.add_argument(
        "--num_thresholds",
        type=int,
        help="(Optional)The number of thresholds to use for matching the given recall.",
        default=200,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance."
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result."
    )
    return argument_parser


def RecallConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the recall of the predictions with respect to the labels.

This metric creates two local variables, `true_positives` and
`false_negatives`, that are used to compute the recall. This value is
ultimately returned as `recall`, an idempotent operation that simply divides
`true_positives` by the sum of `true_positives` and `false_negatives`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `top_k` is set, recall will be computed as how often on average a class
among the labels of a batch entry is in the top-k predictions.

If `class_id` is specified, we calculate recall by considering only the
entries in the batch for which `class_id` is in the label, and computing the
fraction of them for which `class_id` is above the threshold and/or in the
top-k predictions.


Standalone usage:

>>> m = tf.keras.metrics.Recall()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
>>> m.result().numpy()
0.6666667

>>> m.reset_states()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
>>> m.result().numpy()
1.0

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.Recall()])
```"""
    argument_parser.add_argument(
        "--thresholds",
        help="""(Optional) A float value or a python list/tuple of float threshold values in [0, 1]. A threshold is
compared with prediction values to determine the truth value of predictions (i.e., above the
threshold is `true`, below is `false`). One metric value is generated for each threshold value. If
neither thresholds nor top_k are set, the default is to calculate recall with `thresholds=0.5`.""",
        required=True,
    )
    argument_parser.add_argument(
        "--top_k",
        help="""(Optional) Unset by default. An int value specifying the top-k predictions to consider when
calculating recall.""",
        required=True,
    )
    argument_parser.add_argument(
        "--class_id",
        help="""(Optional) Integer class ID for which we want binary metrics. This must be in the half-open interval
`[0, num_classes)`, where `num_classes` is the last dimension of predictions.""",
        required=True,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance.", required=True
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result.", required=True
    )
    return argument_parser


def RecallAtPrecisionConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes best recall where precision is >= specified value.

For a given score-label-distribution the required precision might not
be achievable, in this case 0.0 is returned as recall.

This metric creates four local variables, `true_positives`, `true_negatives`,
`false_positives` and `false_negatives` that are used to compute the
recall at the given precision. The threshold for the given precision
value is computed and used to evaluate the corresponding recall.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.


Standalone usage:

>>> m = tf.keras.metrics.RecallAtPrecision(0.8)
>>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
>>> m.result().numpy()
0.5

>>> m.reset_states()
>>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
...                sample_weight=[1, 0, 0, 1])
>>> m.result().numpy()
1.0

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.RecallAtPrecision(precision=0.8)])
```"""
    argument_parser.add_argument(
        "--precision", help="A scalar value in range `[0, 1]`.", required=True
    )
    argument_parser.add_argument(
        "--num_thresholds",
        type=int,
        help="(Optional)The number of thresholds to use for matching the given precision.",
        default=200,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance."
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result."
    )
    return argument_parser


def RootMeanSquaredErrorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes root mean squared error metric between `y_true` and `y_pred`.

Standalone usage:

>>> m = tf.keras.metrics.RootMeanSquaredError()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
>>> m.result().numpy()
0.5

>>> m.reset_states()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
...                sample_weight=[1, 0])
>>> m.result().numpy()
0.70710677

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError()])
```"""
    return argument_parser


def SensitivityAtSpecificityConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes best sensitivity where specificity is >= specified value.

the sensitivity at a given specificity.

`Sensitivity` measures the proportion of actual positives that are correctly
identified as such (tp / (tp + fn)).
`Specificity` measures the proportion of actual negatives that are correctly
identified as such (tn / (tn + fp)).

This metric creates four local variables, `true_positives`, `true_negatives`,
`false_positives` and `false_negatives` that are used to compute the
sensitivity at the given specificity. The threshold for the given specificity
value is computed and used to evaluate the corresponding sensitivity.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

For additional information about specificity and sensitivity, see
[the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).


Standalone usage:

>>> m = tf.keras.metrics.SensitivityAtSpecificity(0.5)
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
>>> m.result().numpy()
0.5

>>> m.reset_states()
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
...                sample_weight=[1, 1, 2, 2, 1])
>>> m.result().numpy()
0.333333

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.SensitivityAtSpecificity()])
```"""
    argument_parser.add_argument(
        "--specificity", help="A scalar value in range `[0, 1]`.", required=True
    )
    argument_parser.add_argument(
        "--num_thresholds",
        type=int,
        help="(Optional)The number of thresholds to use for matching the given specificity.",
        default=200,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance."
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result."
    )
    return argument_parser


def sparse_categorical_accuracyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        "--y_true", help="Integer ground truth values.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", help="The prediction values.", required=True
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

    :returns: argument_parser
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
    argument_parser.add_argument("--y_true", help="Ground truth values.", required=True)
    argument_parser.add_argument(
        "--y_pred", help="The predicted values.", required=True
    )
    argument_parser.add_argument(
        "--from_logits",
        type=bool,
        help="""Whether `y_pred` is expected to be a logits tensor. By default, we assume that `y_pred` encodes a
probability distribution.""",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--axis",
        type=int,
        help="(Optional) Defaults to -1. The dimension along which the entropy is computed.",
        default=-1,
    )
    return argument_parser, "```None```"


def sparse_top_k_categorical_accuracyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        "--y_true", help="tensor of true targets.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", help="tensor of predicted targets.", required=True
    )
    argument_parser.add_argument(
        "--k",
        type=int,
        help="(Optional) Number of top elements to look at for computing accuracy. Defaults to 5.",
        default=5,
    )
    return argument_parser, "```None```"


def SpecificityAtSensitivityConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes best specificity where sensitivity is >= specified value.

`Sensitivity` measures the proportion of actual positives that are correctly
identified as such (tp / (tp + fn)).
`Specificity` measures the proportion of actual negatives that are correctly
identified as such (tn / (tn + fp)).

This metric creates four local variables, `true_positives`, `true_negatives`,
`false_positives` and `false_negatives` that are used to compute the
specificity at the given sensitivity. The threshold for the given sensitivity
value is computed and used to evaluate the corresponding specificity.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

For additional information about specificity and sensitivity, see
[the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).


Standalone usage:

>>> m = tf.keras.metrics.SpecificityAtSensitivity(0.5)
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
>>> m.result().numpy()
0.66666667

>>> m.reset_states()
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
...                sample_weight=[1, 1, 2, 2, 2])
>>> m.result().numpy()
0.5

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.SpecificityAtSensitivity()])
```"""
    argument_parser.add_argument(
        "--sensitivity", help="A scalar value in range `[0, 1]`.", required=True
    )
    argument_parser.add_argument(
        "--num_thresholds",
        type=int,
        help="(Optional)The number of thresholds to use for matching the given sensitivity.",
        default=200,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance."
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result."
    )
    return argument_parser


def squared_hingeConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        help="""The ground truth values. `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
provided we will convert them to -1 or 1. shape = `[batch_size, d0, .. dN]`.""",
        required=True,
    )
    argument_parser.add_argument(
        "--y_pred",
        help="The predicted values. shape = `[batch_size, d0, .. dN]`.",
        required=True,
    )
    return (
        argument_parser,
        "```K.mean(math_ops.square(math_ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)```",
    )


def SumConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Computes the (weighted) sum of the given values.

For example, if values is [1, 3, 5, 7] then the sum is 16.
If the weights were specified as [1, 1, 0, 0] then the sum would be 4.

This metric creates one variable, `total`, that is used to compute the sum of
`values`. This is ultimately returned as `sum`.

If `sample_weight` is `None`, weights default to 1.  Use `sample_weight` of 0
to mask values.


Standalone usage:

>>> m = tf.keras.metrics.Sum()
>>> m.update_state([1, 3, 5, 7])
>>> m.result().numpy()
16.0

Usage with `compile()` API:

```python
model.add_metric(tf.keras.metrics.Sum(name='sum_1')(outputs))
model.compile(optimizer='sgd', loss='mse')
```"""
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance.", required=True
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result.", required=True
    )
    return argument_parser


def top_k_categorical_accuracyConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
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
        "--y_true", help="The ground truth values.", required=True
    )
    argument_parser.add_argument(
        "--y_pred", help="The prediction values.", required=True
    )
    argument_parser.add_argument(
        "--k",
        type=int,
        help="(Optional) Number of top elements to look at for computing accuracy. Defaults to 5.",
        default=5,
    )
    return argument_parser, "```None```"


def TrueNegativesConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Calculates the number of true negatives.

If `sample_weight` is given, calculates the sum of the weights of
true negatives. This metric creates one local variable, `accumulator`
that is used to keep track of the number of true negatives.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.


Standalone usage:

>>> m = tf.keras.metrics.TrueNegatives()
>>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0])
>>> m.result().numpy()
2.0

>>> m.reset_states()
>>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0], sample_weight=[0, 0, 1, 0])
>>> m.result().numpy()
1.0

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.TrueNegatives()])
```"""
    argument_parser.add_argument(
        "--thresholds",
        type=float,
        help="""(Optional)A float value or a python list/tuple of float threshold values in [0, 1]. A threshold is
compared with prediction values to determine the truth value of predictions (i.e., above the
threshold is `true`, below is `false`). One metric value is generated for each threshold value.""",
        default=0.5,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance."
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result."
    )
    return argument_parser


def TruePositivesConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Calculates the number of true positives.

If `sample_weight` is given, calculates the sum of the weights of
true positives. This metric creates one local variable, `true_positives`
that is used to keep track of the number of true positives.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.


Standalone usage:

>>> m = tf.keras.metrics.TruePositives()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
>>> m.result().numpy()
2.0

>>> m.reset_states()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
>>> m.result().numpy()
1.0

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[tf.keras.metrics.TruePositives()])
```"""
    argument_parser.add_argument(
        "--thresholds",
        type=float,
        help="""(Optional)A float value or a python list/tuple of float threshold values in [0, 1]. A threshold is
compared with prediction values to determine the truth value of predictions (i.e., above the
threshold is `true`, below is `false`). One metric value is generated for each threshold value.""",
        default=0.5,
    )
    argument_parser.add_argument(
        "--name", help="(Optional) string name of the metric instance."
    )
    argument_parser.add_argument(
        "--dtype", help="(Optional) data type of the metric result."
    )
    return argument_parser


__all__ = [
    "AUCConfig",
    "binary_accuracyConfig",
    "binary_crossentropyConfig",
    "categorical_accuracyConfig",
    "categorical_crossentropyConfig",
    "CategoricalHingeConfig",
    "CosineSimilarityConfig",
    "FalseNegativesConfig",
    "FalsePositivesConfig",
    "hingeConfig",
    "kldConfig",
    "kl_divergenceConfig",
    "kullback_leibler_divergenceConfig",
    "logcoshConfig",
    "LogCoshErrorConfig",
    "maeConfig",
    "mapeConfig",
    "MeanConfig",
    "mean_absolute_errorConfig",
    "mean_absolute_percentage_errorConfig",
    "MeanIoUConfig",
    "MeanRelativeErrorConfig",
    "mean_squared_errorConfig",
    "mean_squared_logarithmic_errorConfig",
    "MeanTensorConfig",
    "mseConfig",
    "msleConfig",
    "poissonConfig",
    "PrecisionConfig",
    "PrecisionAtRecallConfig",
    "RecallConfig",
    "RecallAtPrecisionConfig",
    "RootMeanSquaredErrorConfig",
    "SensitivityAtSpecificityConfig",
    "sparse_categorical_accuracyConfig",
    "sparse_categorical_crossentropyConfig",
    "sparse_top_k_categorical_accuracyConfig",
    "SpecificityAtSensitivityConfig",
    "squared_hingeConfig",
    "SumConfig",
    "top_k_categorical_accuracyConfig",
    "TrueNegativesConfig",
    "TruePositivesConfig",
]
