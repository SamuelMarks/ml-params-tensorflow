""" Generated Optimizer CLI parsers """
from __future__ import absolute_import, division, print_function

from yaml import safe_load as loads


def AdadeltaConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Optimizer that implements the Adadelta algorithm.

Adadelta optimization is a stochastic gradient descent method that is based on
adaptive learning rate per dimension to address two drawbacks:

- The continual decay of learning rates throughout training
- The need for a manually selected global learning rate

Adadelta is a more robust extension of Adagrad that adapts learning rates
based on a moving window of gradient updates, instead of accumulating all
past gradients. This way, Adadelta continues learning even when many updates
have been done. Compared to Adagrad, in the original version of Adadelta you
don't have to set an initial learning rate. In this version, initial
learning rate can be set, as in most other Keras optimizers.

According to section 4.3 ("Effective Learning rates"), near the end of
training step sizes converge to 1 which is effectively a high learning
rate which would cause divergence. This occurs only near the end of the
training as gradients and step sizes are small, and the epsilon constant
in the numerator and denominator dominate past gradients and parameter
updates which converge the learning rate to 1.

According to section 4.4("Speech Data"),where a large neural network with
4 hidden layers was trained on a corpus of US English data, ADADELTA was
used with 100 network replicas.The epsilon used is 1e-6 with rho=0.95
which converged faster than ADAGRAD, by the following construction:
def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, decay=0., **kwargs):


Reference:
  - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)"""
    argument_parser.add_argument(
        "--learning_rate",
        type=float,
        help="""A `Tensor`, floating point value, or a schedule that is a
    `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate.
    To match the exact form in the original paper use 1.0.""",
        required=True,
        default=0.001,
    )
    argument_parser.add_argument(
        "--rho",
        type=float,
        help="A `Tensor` or a floating point value. The decay rate.",
        required=True,
        default=0.95,
    )
    argument_parser.add_argument(
        "--epsilon",
        type=float,
        help="""A `Tensor` or a floating point value.  A constant epsilon used
           to better conditioning the grad update.""",
        required=True,
        default=1e-07,
    )
    argument_parser.add_argument(
        "--name",
        type=str,
        help="""Optional name prefix for the operations created when applying
    gradients. """,
        default="Adadelta",
    )
    argument_parser.add_argument(
        "--kwargs",
        type=loads,
        help="""Keyword arguments. Allowed to be one of
    `"clipnorm"` or `"clipvalue"`.
    `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
    gradients by value.""",
    )
    argument_parser.add_argument(
        "--_HAS_AGGREGATE_GRAD", type=bool, required=True, default=True
    )
    return argument_parser


def AdagradConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Optimizer that implements the Adagrad algorithm.

Adagrad is an optimizer with parameter-specific learning rates,
which are adapted relative to how frequently a parameter gets
updated during training. The more updates a parameter receives,
the smaller the updates.


Reference:
  - [Duchi et al., 2011](
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)."""
    argument_parser.add_argument(
        "--learning_rate",
        type=float,
        help="""A `Tensor`, floating point value, or a schedule that is a
    `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate.""",
        required=True,
        default=0.001,
    )
    argument_parser.add_argument(
        "--initial_accumulator_value",
        type=float,
        help="""A floating point value.
    Starting value for the accumulators, must be non-negative.""",
        required=True,
        default=0.1,
    )
    argument_parser.add_argument(
        "--epsilon",
        type=float,
        help="A small floating point value to avoid zero denominator.",
        required=True,
        default=1e-07,
    )
    argument_parser.add_argument(
        "--name",
        type=str,
        help="""Optional name prefix for the operations created when applying
    gradients. """,
        default="Adagrad",
    )
    argument_parser.add_argument(
        "--kwargs",
        type=loads,
        help="""Keyword arguments. Allowed to be one of
    `"clipnorm"` or `"clipvalue"`.
    `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
    gradients by value.""",
    )
    argument_parser.add_argument(
        "--_HAS_AGGREGATE_GRAD", type=bool, required=True, default=True
    )
    return argument_parser


def AdamConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Optimizer that implements the Adam algorithm.

Adam optimization is a stochastic gradient descent method that is based on
adaptive estimation of first-order and second-order moments.

According to
[Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
the method is "*computationally
efficient, has little memory requirement, invariant to diagonal rescaling of
gradients, and is well suited for problems that are large in terms of
data/parameters*".


Usage:

>>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
>>> var1 = tf.Variable(10.0)
>>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
>>> step_count = opt.minimize(loss, [var1]).numpy()
>>> # The first step is `-learning_rate*sign(grad)`
>>> var1.numpy()
9.9

Reference:
  - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
  - [Reddi et al., 2018](
      https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

Notes:

The default value of 1e-7 for epsilon might not be a good default in
general. For example, when training an Inception network on ImageNet a
current good choice is 1.0 or 0.1. Note that since Adam uses the
formulation just before Section 2.1 of the Kingma and Ba paper rather than
the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
hat" in the paper.

The sparse implementation of this algorithm (used when the gradient is an
IndexedSlices object, typically because of `tf.gather` or an embedding
lookup in the forward pass) does apply momentum to variable slices even if
they were not used in the forward pass (meaning they have a gradient equal
to zero). Momentum decay (beta1) is also applied to the entire momentum
accumulator. This means that the sparse behavior is equivalent to the dense
behavior (in contrast to some momentum implementations which ignore momentum
unless a variable slice was actually used)."""
    argument_parser.add_argument(
        "--learning_rate",
        type=float,
        help="""A `Tensor`, floating point value, or a schedule that is a
    `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
    that takes no arguments and returns the actual value to use, The
    learning rate.""",
        required=True,
        default=0.001,
    )
    argument_parser.add_argument(
        "--beta_1",
        type=float,
        help="""A float value or a constant float tensor, or a callable
    that takes no arguments and returns the actual value to use. The
    exponential decay rate for the 1st moment estimates.""",
        required=True,
        default=0.9,
    )
    argument_parser.add_argument(
        "--beta_2",
        type=float,
        help="""A float value or a constant float tensor, or a callable
    that takes no arguments and returns the actual value to use, The
    exponential decay rate for the 2nd moment estimates.""",
        required=True,
        default=0.999,
    )
    argument_parser.add_argument(
        "--epsilon",
        type=float,
        help="""A small constant for numerical stability. This epsilon is
    "epsilon hat" in the Kingma and Ba paper (in the formula just before
    Section 2.1), not the epsilon in Algorithm 1 of the paper.""",
        required=True,
        default=1e-07,
    )
    argument_parser.add_argument(
        "--amsgrad",
        type=bool,
        help="""Boolean. Whether to apply AMSGrad variant of this algorithm from
    the paper "On the Convergence of Adam and beyond".""",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--name",
        type=str,
        help="""Optional name for the operations created when applying gradients.
   """,
        default="Adam",
    )
    argument_parser.add_argument(
        "--kwargs",
        type=loads,
        help="""Keyword arguments. Allowed to be one of
    `"clipnorm"` or `"clipvalue"`.
    `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
    gradients by value.""",
    )
    argument_parser.add_argument(
        "--_HAS_AGGREGATE_GRAD", type=bool, required=True, default=True
    )
    return argument_parser


def AdamaxConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Optimizer that implements the Adamax algorithm.

It is a variant of Adam based on the infinity norm.
Default parameters follow those provided in the paper.
Adamax is sometimes superior to adam, specially in models with embeddings.

Initialization:

```python
m = 0  # Initialize initial 1st moment vector
v = 0  # Initialize the exponentially weighted infinity norm
t = 0  # Initialize timestep
```

The update rule for parameter `w` with gradient `g` is
described at the end of section 7.1 of the paper:

```python
t += 1
m = beta1 * m + (1 - beta) * g
v = max(beta2 * v, abs(g))
current_lr = learning_rate / (1 - beta1 ** t)
w = w - current_lr * m / (v + epsilon)
```

Similarly to `Adam`, the epsilon is added for numerical stability
(especially to get rid of division by zero when `v_t == 0`).

In contrast to `Adam`, the sparse implementation of this algorithm
(used when the gradient is an IndexedSlices object, typically because of
`tf.gather` or an embedding lookup in the forward pass) only updates
variable slices and corresponding `m_t`, `v_t` terms when that part of
the variable was used in the forward pass. This means that the sparse
behavior is contrast to the dense behavior (similar to some momentum
implementations which ignore momentum unless a variable slice was actually
used).


Reference:
  - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)"""
    argument_parser.add_argument(
        "--learning_rate",
        type=float,
        help="""A `Tensor`, floating point value, or a schedule that is a
    `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate.""",
        required=True,
        default=0.001,
    )
    argument_parser.add_argument(
        "--beta_1",
        type=float,
        help="""A float value or a constant float tensor. The exponential decay
    rate for the 1st moment estimates.""",
        required=True,
        default=0.9,
    )
    argument_parser.add_argument(
        "--beta_2",
        type=float,
        help="""A float value or a constant float tensor. The exponential decay
    rate for the exponentially weighted infinity norm.""",
        required=True,
        default=0.999,
    )
    argument_parser.add_argument(
        "--epsilon",
        type=float,
        help="A small constant for numerical stability.",
        required=True,
        default=1e-07,
    )
    argument_parser.add_argument(
        "--name",
        type=str,
        help="""Optional name for the operations created when applying gradients.
   """,
        default="Adamax",
    )
    argument_parser.add_argument(
        "--kwargs",
        type=loads,
        help="""Keyword arguments. Allowed to be one of
    `"clipnorm"` or `"clipvalue"`.
    `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
    gradients by value.""",
    )
    argument_parser.add_argument(
        "--_HAS_AGGREGATE_GRAD", type=bool, required=True, default=True
    )
    return argument_parser


def FtrlConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Optimizer that implements the FTRL algorithm.

See Algorithm 1 of this
[paper](https://research.google.com/pubs/archive/41159.pdf).
This version has support for both online L2 (the L2 penalty given in the paper
above) and shrinkage-type L2 (which is the addition of an L2 penalty to the
loss function).

Initialization:
$$t = 0$$
$$n_{0} = 0$$
$$\\sigma_{0} = 0$$
$$z_{0} = 0$$

Update ($$i$$ is variable index, $$\\alpha$$ is the learning rate):
$$t = t + 1$$
$$n_{t,i} = n_{t-1,i} + g_{t,i}^{2}$$
$$\\sigma_{t,i} = (\\sqrt{n_{t,i}} - \\sqrt{n_{t-1,i}}) / \\alpha$$
$$z_{t,i} = z_{t-1,i} + g_{t,i} - \\sigma_{t,i} * w_{t,i}$$
$$w_{t,i} = - ((\\beta+\\sqrt{n_{t,i}}) / \\alpha + 2 * \\lambda_{2})^{-1} *
            (z_{i} - sgn(z_{i}) * \\lambda_{1}) if \\abs{z_{i}} > \\lambda_{i}
                                               else 0$$

Check the documentation for the l2_shrinkage_regularization_strength
parameter for more details when shrinkage is enabled, in which case gradient
is replaced with gradient_with_shrinkage.


Reference:
  - [paper](
    https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)"""
    argument_parser.add_argument(
        "--learning_rate",
        type=float,
        help="""A `Tensor`, floating point value, or a schedule that is a
    `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate.""",
        required=True,
        default=0.001,
    )
    argument_parser.add_argument(
        "--learning_rate_power",
        type=float,
        help="""A float value, must be less or equal to zero.
    Controls how the learning rate decreases during training. Use zero for
    a fixed learning rate.""",
        required=True,
        default=-0.5,
    )
    argument_parser.add_argument(
        "--initial_accumulator_value",
        type=float,
        help="""The starting value for accumulators.
    Only zero or positive values are allowed.""",
        required=True,
        default=0.1,
    )
    argument_parser.add_argument(
        "--l1_regularization_strength",
        type=float,
        help="""A float value, must be greater than or
    equal to zero.""",
        required=True,
        default=0.0,
    )
    argument_parser.add_argument(
        "--l2_regularization_strength",
        type=float,
        help="""A float value, must be greater than or
    equal to zero.""",
        required=True,
        default=0.0,
    )
    argument_parser.add_argument(
        "--name",
        type=str,
        help="""Optional name prefix for the operations created when applying
    gradients. """,
        default="Ftrl",
    )
    argument_parser.add_argument(
        "--l2_shrinkage_regularization_strength",
        type=float,
        help="""A float value, must be greater than
    or equal to zero. This differs from L2 above in that the L2 above is a
    stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
    When input is sparse shrinkage will only happen on the active weights.""",
        required=True,
        default=0.0,
    )
    argument_parser.add_argument(
        "--beta",
        type=float,
        help="A float value, representing the beta value from the paper.",
        required=True,
        default=0.0,
    )
    argument_parser.add_argument(
        "--kwargs",
        type=loads,
        help="""Keyword arguments. Allowed to be one of
    `"clipnorm"` or `"clipvalue"`.
    `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
    gradients by value.""",
    )
    return argument_parser


def NadamConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Optimizer that implements the NAdam algorithm.

Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
Nesterov momentum.


Reference:
  - [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf)."""
    argument_parser.add_argument(
        "--learning_rate",
        type=float,
        help="A Tensor or a floating point value.  The learning rate.",
        required=True,
        default=0.001,
    )
    argument_parser.add_argument(
        "--beta_1",
        type=float,
        help="""A float value or a constant float tensor. The exponential decay
    rate for the 1st moment estimates.""",
        required=True,
        default=0.9,
    )
    argument_parser.add_argument(
        "--beta_2",
        type=float,
        help="""A float value or a constant float tensor. The exponential decay
    rate for the exponentially weighted infinity norm.""",
        required=True,
        default=0.999,
    )
    argument_parser.add_argument(
        "--epsilon",
        type=float,
        help="A small constant for numerical stability.",
        required=True,
        default=1e-07,
    )
    argument_parser.add_argument(
        "--name",
        type=str,
        help="""Optional name for the operations created when applying gradients.
   """,
        default="Nadam",
    )
    argument_parser.add_argument(
        "--kwargs",
        type=loads,
        help="""Keyword arguments. Allowed to be one of
    `"clipnorm"` or `"clipvalue"`.
    `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
    gradients by value.""",
    )
    argument_parser.add_argument(
        "--_HAS_AGGREGATE_GRAD", type=bool, required=True, default=True
    )
    return argument_parser


def RMSpropConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Optimizer that implements the RMSprop algorithm.

The gist of RMSprop is to:

- Maintain a moving (discounted) average of the square of gradients
- Divide the gradient by the root of this average

This implementation of RMSprop uses plain momentum, not Nesterov momentum.

The centered version additionally maintains a moving average of the
gradients, and uses that average to estimate the variance.


Note that in the dense implementation of this algorithm, variables and their
corresponding accumulators (momentum, gradient moving average, square
gradient moving average) will be updated even if the gradient is zero
(i.e. accumulators will decay, momentum will be applied). The sparse
implementation (used when the gradient is an `IndexedSlices` object,
typically because of `tf.gather` or an embedding lookup in the forward pass)
will not update variable slices or their accumulators unless those slices
were used in the forward pass (nor is there an "eventual" correction to
account for these omitted updates). This leads to more efficient updates for
large embedding lookup tables (where most of the slices are not accessed in
a particular graph execution), but differs from the published algorithm.

Usage:

>>> opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
>>> var1 = tf.Variable(10.0)
>>> loss = lambda: (var1 ** 2) / 2.0    # d(loss) / d(var1) = var1
>>> step_count = opt.minimize(loss, [var1]).numpy()
>>> var1.numpy()
9.683772

Reference:
  - [Hinton, 2012](
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)"""
    argument_parser.add_argument(
        "--learning_rate",
        type=float,
        help="""A `Tensor`, floating point value, or a schedule that is a
    `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
    that takes no arguments and returns the actual value to use. The
    learning rate.""",
        required=True,
        default=0.001,
    )
    argument_parser.add_argument(
        "--rho",
        type=float,
        help="Discounting factor for the history/coming gradient.",
        required=True,
        default=0.9,
    )
    argument_parser.add_argument(
        "--momentum",
        type=float,
        help="A scalar or a scalar `Tensor`.",
        required=True,
        default=0.0,
    )
    argument_parser.add_argument(
        "--epsilon",
        type=float,
        help="""A small constant for numerical stability. This epsilon is
    "epsilon hat" in the Kingma and Ba paper (in the formula just before
    Section 2.1), not the epsilon in Algorithm 1 of the paper.""",
        required=True,
        default=1e-07,
    )
    argument_parser.add_argument(
        "--centered",
        type=bool,
        help="""Boolean. If `True`, gradients are normalized by the estimated
    variance of the gradient; if False, by the uncentered second moment.
    Setting this to `True` may help with training, but is slightly more
    expensive in terms of computation and memory.""",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--name",
        type=str,
        help="""Optional name prefix for the operations created when applying
    gradients.""",
        default="RMSprop",
    )
    argument_parser.add_argument(
        "--kwargs",
        type=loads,
        help="""Keyword arguments. Allowed to be one of
    `"clipnorm"` or `"clipvalue"`.
    `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
    gradients by value.""",
    )
    argument_parser.add_argument(
        "--_HAS_AGGREGATE_GRAD", type=bool, required=True, default=True
    )
    return argument_parser


__all__ = [
    "AdadeltaConfig",
    "AdagradConfig",
    "AdamConfig",
    "AdamaxConfig",
    "FtrlConfig",
    "NadamConfig",
    "RMSpropConfig",
]
