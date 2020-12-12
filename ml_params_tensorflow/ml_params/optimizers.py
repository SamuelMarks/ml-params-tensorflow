""" Generated Callback config classes """
from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

NoneType = type(None)


@dataclass
class AdadeltaConfig(object):
    """
    Optimizer that implements the Adadelta algorithm.

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
      - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)

    :cvar learning_rate: A `Tensor`, floating point value, or a schedule that is a
    `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate.
    To match the exact form in the original paper use 1.0. Defaults to 0.001
    :cvar rho: A `Tensor` or a floating point value. The decay rate. Defaults to 0.95
    :cvar epsilon: A `Tensor` or a floating point value.  A constant epsilon used
           to better conditioning the grad update. Defaults to 1e-07
    :cvar name: Optional name prefix for the operations created when applying
    gradients. . Defaults to "Adadelta"
    :cvar kwargs: Keyword arguments. Allowed to be one of
    `"clipnorm"` or `"clipvalue"`.
    `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
    gradients by value.
    :cvar _HAS_AGGREGATE_GRAD: None"""

    learning_rate: float = 0.001
    rho: float = 0.95
    epsilon: float = 1e-07
    name: str = "Adadelta"
    kwargs: Optional[dict] = None
    _HAS_AGGREGATE_GRAD = True


@dataclass
class AdagradConfig(object):
    """
    Optimizer that implements the Adagrad algorithm.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.


    Reference:
      - [Duchi et al., 2011](
        http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).

    :cvar learning_rate: A `Tensor`, floating point value, or a schedule that is a
    `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate. Defaults to 0.001
    :cvar initial_accumulator_value: A floating point value.
    Starting value for the accumulators, must be non-negative. Defaults to 0.1
    :cvar epsilon: A small floating point value to avoid zero denominator. Defaults to 1e-07
    :cvar name: Optional name prefix for the operations created when applying
    gradients. . Defaults to "Adagrad"
    :cvar kwargs: Keyword arguments. Allowed to be one of
    `"clipnorm"` or `"clipvalue"`.
    `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
    gradients by value.
    :cvar _HAS_AGGREGATE_GRAD: None"""

    learning_rate: float = 0.001
    initial_accumulator_value: float = 0.1
    epsilon: float = 1e-07
    name: str = "Adagrad"
    kwargs: Optional[dict] = None
    _HAS_AGGREGATE_GRAD = True


@dataclass
class AdamConfig(object):
    """
     Optimizer that implements the Adam algorithm.

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
     unless a variable slice was actually used).

     :cvar learning_rate: A `Tensor`, floating point value, or a schedule that is a
     `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
     that takes no arguments and returns the actual value to use, The
     learning rate. Defaults to 0.001
     :cvar beta_1: A float value or a constant float tensor, or a callable
     that takes no arguments and returns the actual value to use. The
     exponential decay rate for the 1st moment estimates. Defaults to 0.9
     :cvar beta_2: A float value or a constant float tensor, or a callable
     that takes no arguments and returns the actual value to use, The
     exponential decay rate for the 2nd moment estimates. Defaults to 0.999
     :cvar epsilon: A small constant for numerical stability. This epsilon is
     "epsilon hat" in the Kingma and Ba paper (in the formula just before
     Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to 1e-07
     :cvar amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
     the paper "On the Convergence of Adam and beyond". Defaults to False
     :cvar name: Optional name for the operations created when applying gradients.
    . Defaults to "Adam"
     :cvar kwargs: Keyword arguments. Allowed to be one of
     `"clipnorm"` or `"clipvalue"`.
     `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
     gradients by value.
     :cvar _HAS_AGGREGATE_GRAD: None"""

    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False
    name: str = "Adam"
    kwargs: Optional[dict] = None
    _HAS_AGGREGATE_GRAD = True


@dataclass
class AdamaxConfig(object):
    """
     Optimizer that implements the Adamax algorithm.

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
       - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)

     :cvar learning_rate: A `Tensor`, floating point value, or a schedule that is a
     `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate. Defaults to 0.001
     :cvar beta_1: A float value or a constant float tensor. The exponential decay
     rate for the 1st moment estimates. Defaults to 0.9
     :cvar beta_2: A float value or a constant float tensor. The exponential decay
     rate for the exponentially weighted infinity norm. Defaults to 0.999
     :cvar epsilon: A small constant for numerical stability. Defaults to 1e-07
     :cvar name: Optional name for the operations created when applying gradients.
    . Defaults to "Adamax"
     :cvar kwargs: Keyword arguments. Allowed to be one of
     `"clipnorm"` or `"clipvalue"`.
     `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
     gradients by value.
     :cvar _HAS_AGGREGATE_GRAD: None"""

    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    name: str = "Adamax"
    kwargs: Optional[dict] = None
    _HAS_AGGREGATE_GRAD = True


@dataclass
class FtrlConfig(object):
    """
    Optimizer that implements the FTRL algorithm.

    See Algorithm 1 of this [paper](
    https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).
    This version has support for both online L2 (the L2 penalty given in the paper
    above) and shrinkage-type L2 (which is the addition of an L2 penalty to the
    loss function).


    Reference:
      - [paper](
        https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)

    :cvar learning_rate: A `Tensor`, floating point value, or a schedule that is a
    `tf.keras.optimizers.schedules.LearningRateSchedule`. The learning rate. Defaults to 0.001
    :cvar learning_rate_power: A float value, must be less or equal to zero.
    Controls how the learning rate decreases during training. Use zero for
    a fixed learning rate. Defaults to -0.5
    :cvar initial_accumulator_value: The starting value for accumulators.
    Only zero or positive values are allowed. Defaults to 0.1
    :cvar l1_regularization_strength: A float value, must be greater than or
    equal to zero. Defaults to 0.0
    :cvar l2_regularization_strength: A float value, must be greater than or
    equal to zero. Defaults to 0.0
    :cvar name: Optional name prefix for the operations created when applying
    gradients. . Defaults to "Ftrl"
    :cvar l2_shrinkage_regularization_strength: A float value, must be greater than
    or equal to zero. This differs from L2 above in that the L2 above is a
    stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
    When input is sparse shrinkage will only happen on the active weights. Defaults to 0.0
    :cvar kwargs: Keyword arguments. Allowed to be one of
    `"clipnorm"` or `"clipvalue"`.
    `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
    gradients by value."""

    learning_rate: float = 0.001
    learning_rate_power: float = -0.5
    initial_accumulator_value: float = 0.1
    l1_regularization_strength: float = 0.0
    l2_regularization_strength: float = 0.0
    name: str = "Ftrl"
    l2_shrinkage_regularization_strength: float = 0.0
    kwargs: Optional[dict] = None


@dataclass
class NadamConfig(object):
    """
     Optimizer that implements the NAdam algorithm.

     Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
     Nesterov momentum.


     Reference:
       - [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).

     :cvar learning_rate: A Tensor or a floating point value.  The learning rate. Defaults to 0.001
     :cvar beta_1: A float value or a constant float tensor. The exponential decay
     rate for the 1st moment estimates. Defaults to 0.9
     :cvar beta_2: A float value or a constant float tensor. The exponential decay
     rate for the exponentially weighted infinity norm. Defaults to 0.999
     :cvar epsilon: A small constant for numerical stability. Defaults to 1e-07
     :cvar name: Optional name for the operations created when applying gradients.
    . Defaults to "Nadam"
     :cvar kwargs: Keyword arguments. Allowed to be one of
     `"clipnorm"` or `"clipvalue"`.
     `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
     gradients by value.
     :cvar _HAS_AGGREGATE_GRAD: None"""

    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    name: str = "Nadam"
    kwargs: Optional[dict] = None
    _HAS_AGGREGATE_GRAD = True


@dataclass
class RMSpropConfig(object):
    """
    Optimizer that implements the RMSprop algorithm.

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
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    :cvar learning_rate: A `Tensor`, floating point value, or a schedule that is a
    `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
    that takes no arguments and returns the actual value to use. The
    learning rate. Defeaults to 0.001. Defaults to 0.001
    :cvar rho: Discounting factor for the history/coming gradient. Defaults to 0.9
    :cvar momentum: A scalar or a scalar `Tensor`. Defaults to 0.0
    :cvar epsilon: A small constant for numerical stability. This epsilon is
    "epsilon hat" in the Kingma and Ba paper (in the formula just before
    Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to 1e-07
    :cvar centered: Boolean. If `True`, gradients are normalized by the estimated
    variance of the gradient; if False, by the uncentered second moment.
    Setting this to `True` may help with training, but is slightly more
    expensive in terms of computation and memory. Defaults to False
    :cvar name: Optional name prefix for the operations created when applying
    gradients. Defaults to "RMSprop"
    :cvar kwargs: Keyword arguments. Allowed to be one of
    `"clipnorm"` or `"clipvalue"`.
    `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
    gradients by value.
    :cvar _HAS_AGGREGATE_GRAD: None"""

    learning_rate: float = 0.001
    rho: float = 0.9
    momentum: float = 0.0
    epsilon: float = 1e-07
    centered: bool = False
    name: str = "RMSprop"
    kwargs: Optional[dict] = None
    _HAS_AGGREGATE_GRAD = True


__all__ = [
    "AdadeltaConfig",
    "AdagradConfig",
    "AdamConfig",
    "AdamaxConfig",
    "FtrlConfig",
    "NadamConfig",
    "RMSpropConfig",
]
