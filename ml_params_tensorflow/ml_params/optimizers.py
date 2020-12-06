""" Generated Optimizer config classes """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables


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
    kwargs: dict = {}
    _HAS_AGGREGATE_GRAD = True

    def __call__(self):
        self._HAS_AGGREGATE_GRAD = True

        def __init__(
            self,
            learning_rate=0.001,
            rho=0.95,
            epsilon=1e-07,
            name="Adadelta",
            **kwargs
        ):
            super(Adadelta, self).__init__(self.name, **self.kwargs)
            self._set_hyper("learning_rate", self.kwargs.get("lr", self.learning_rate))
            self._set_hyper("decay", self._initial_decay)
            self._set_hyper("rho", self.rho)
            self.epsilon = self.epsilon or backend_config.epsilon()

        def _create_slots(self, var_list):
            for v in var_list:
                self.add_slot(v, "accum_grad")
            for v in var_list:
                self.add_slot(v, "accum_var")

        def _prepare_local(self, var_device, var_dtype, apply_state):
            super(Adadelta, self)._prepare_local(var_device, var_dtype, apply_state)
            apply_state[var_device, var_dtype].update(
                dict(
                    epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                    rho=array_ops.identity(self._get_hyper("rho", var_dtype)),
                )
            )

        def set_weights(self, weights):
            params = self.weights
            if len(params) == len(weights) + 1:
                weights = [np.array(0)] + weights
            super(Adadelta, self).set_weights(weights)

        def _resource_apply_dense(self, grad, var, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            accum_grad = self.get_slot(var, "accum_grad")
            accum_var = self.get_slot(var, "accum_var")
            return training_ops.resource_apply_adadelta(
                var.handle,
                accum_grad.handle,
                accum_var.handle,
                coefficients["lr_t"],
                coefficients["rho"],
                coefficients["epsilon"],
                grad,
                use_locking=self._use_locking,
            )

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            accum_grad = self.get_slot(var, "accum_grad")
            accum_var = self.get_slot(var, "accum_var")
            return training_ops.resource_sparse_apply_adadelta(
                var.handle,
                accum_grad.handle,
                accum_var.handle,
                coefficients["lr_t"],
                coefficients["rho"],
                coefficients["epsilon"],
                grad,
                indices,
                use_locking=self._use_locking,
            )

        def get_config(self):
            config = super(Adadelta, self).get_config()
            config.update(
                {
                    "learning_rate": self._serialize_hyperparameter("learning_rate"),
                    "decay": self._serialize_hyperparameter("decay"),
                    "rho": self._serialize_hyperparameter("rho"),
                    "epsilon": self.epsilon,
                }
            )
            return config


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
    kwargs: dict = {}
    _HAS_AGGREGATE_GRAD = True

    def __call__(self):
        self._HAS_AGGREGATE_GRAD = True

        def __init__(
            self,
            learning_rate=0.001,
            initial_accumulator_value=0.1,
            epsilon=1e-07,
            name="Adagrad",
            **kwargs
        ):
            if self.initial_accumulator_value < 0.0:
                raise ValueError(
                    "initial_accumulator_value must be non-negative: %s"
                    % self.initial_accumulator_value
                )
            if self.epsilon is None:
                self.epsilon = backend_config.epsilon()
            super(Adagrad, self).__init__(self.name, **self.kwargs)
            self._set_hyper("learning_rate", self.kwargs.get("lr", self.learning_rate))
            self._set_hyper("decay", self._initial_decay)
            self._initial_accumulator_value = self.initial_accumulator_value
            self.epsilon = self.epsilon or backend_config.epsilon()

        def _create_slots(self, var_list):
            for var in var_list:
                dtype = var.dtype.base_dtype
                init = init_ops.constant_initializer(
                    self._initial_accumulator_value, dtype=dtype
                )
                self.add_slot(var, "accumulator", init)

        def _prepare_local(self, var_device, var_dtype, apply_state):
            super(Adagrad, self)._prepare_local(var_device, var_dtype, apply_state)
            apply_state[var_device, var_dtype].update(
                dict(
                    epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                    neg_lr_t=-apply_state[var_device, var_dtype]["lr_t"],
                    zero=array_ops.zeros((), dtype=dtypes.int64),
                )
            )

        def set_weights(self, weights):
            params = self.weights
            if len(params) == len(weights) + 1:
                weights = [np.array(0)] + weights
            super(Adagrad, self).set_weights(weights)

        @classmethod
        def from_config(cls, config, custom_objects=None):
            """Creates an optimizer from its config.

            This method is the reverse of `get_config`,
            capable of instantiating the same optimizer from the config
            dictionary.

            Args:
                config: A Python dictionary, typically the output of get_config.
                custom_objects: A Python dictionary mapping names to additional Python
                  objects used to create this optimizer, such as a function used for a
                  hyperparameter.

            Returns:
                An optimizer instance.
            """
            if "initial_accumulator_value" not in config:
                config["initial_accumulator_value"] = 0.1
            if "lr" in config:
                config["learning_rate"] = config.pop("lr")
            return cls(**config)

        def _resource_apply_dense(self, grad, var, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            acc = self.get_slot(var, "accumulator")
            return training_ops.resource_apply_adagrad_v2(
                var.handle,
                acc.handle,
                coefficients["lr_t"],
                coefficients["epsilon"],
                grad,
                use_locking=self._use_locking,
            )

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            acc = self.get_slot(var, "accumulator")
            return training_ops.resource_sparse_apply_adagrad_v2(
                var.handle,
                acc.handle,
                coefficients["lr_t"],
                coefficients["epsilon"],
                grad,
                indices,
                use_locking=self._use_locking,
            )

        def get_config(self):
            config = super(Adagrad, self).get_config()
            config.update(
                {
                    "learning_rate": self._serialize_hyperparameter("learning_rate"),
                    "decay": self._serialize_hyperparameter("decay"),
                    "initial_accumulator_value": self._initial_accumulator_value,
                    "epsilon": self.epsilon,
                }
            )
            return config


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
    kwargs: dict = {}
    _HAS_AGGREGATE_GRAD = True

    def __call__(self):
        self._HAS_AGGREGATE_GRAD = True

        def __init__(
            self,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
            **kwargs
        ):
            super(Adam, self).__init__(self.name, **self.kwargs)
            self._set_hyper("learning_rate", self.kwargs.get("lr", self.learning_rate))
            self._set_hyper("decay", self._initial_decay)
            self._set_hyper("beta_1", self.beta_1)
            self._set_hyper("beta_2", self.beta_2)
            self.epsilon = self.epsilon or backend_config.epsilon()
            self.amsgrad = self.amsgrad

        def _create_slots(self, var_list):
            for var in var_list:
                self.add_slot(var, "m")
            for var in var_list:
                self.add_slot(var, "v")
            if self.amsgrad:
                for var in var_list:
                    self.add_slot(var, "vhat")

        def _prepare_local(self, var_device, var_dtype, apply_state):
            super(Adam, self)._prepare_local(var_device, var_dtype, apply_state)
            local_step = math_ops.cast(self.iterations + 1, var_dtype)
            beta_1_t = array_ops.identity(self._get_hyper("beta_1", var_dtype))
            beta_2_t = array_ops.identity(self._get_hyper("beta_2", var_dtype))
            beta_1_power = math_ops.pow(beta_1_t, local_step)
            beta_2_power = math_ops.pow(beta_2_t, local_step)
            lr = apply_state[var_device, var_dtype]["lr_t"] * (
                math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
            )
            apply_state[var_device, var_dtype].update(
                dict(
                    lr=lr,
                    epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                    beta_1_t=beta_1_t,
                    beta_1_power=beta_1_power,
                    one_minus_beta_1_t=1 - beta_1_t,
                    beta_2_t=beta_2_t,
                    beta_2_power=beta_2_power,
                    one_minus_beta_2_t=1 - beta_2_t,
                )
            )

        def set_weights(self, weights):
            params = self.weights
            num_vars = int((len(params) - 1) / 2)
            if len(weights) == 3 * num_vars + 1:
                weights = weights[: len(params)]
            super(Adam, self).set_weights(weights)

        def _resource_apply_dense(self, grad, var, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            m = self.get_slot(var, "m")
            v = self.get_slot(var, "v")
            if not self.amsgrad:
                return training_ops.resource_apply_adam(
                    var.handle,
                    m.handle,
                    v.handle,
                    coefficients["beta_1_power"],
                    coefficients["beta_2_power"],
                    coefficients["lr_t"],
                    coefficients["beta_1_t"],
                    coefficients["beta_2_t"],
                    coefficients["epsilon"],
                    grad,
                    use_locking=self._use_locking,
                )
            else:
                vhat = self.get_slot(var, "vhat")
                return training_ops.resource_apply_adam_with_amsgrad(
                    var.handle,
                    m.handle,
                    v.handle,
                    vhat.handle,
                    coefficients["beta_1_power"],
                    coefficients["beta_2_power"],
                    coefficients["lr_t"],
                    coefficients["beta_1_t"],
                    coefficients["beta_2_t"],
                    coefficients["epsilon"],
                    grad,
                    use_locking=self._use_locking,
                )

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            m = self.get_slot(var, "m")
            m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
            m_t = state_ops.assign(
                m, m * coefficients["beta_1_t"], use_locking=self._use_locking
            )
            with ops.control_dependencies([m_t]):
                m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
            v = self.get_slot(var, "v")
            v_scaled_g_values = grad * grad * coefficients["one_minus_beta_2_t"]
            v_t = state_ops.assign(
                v, v * coefficients["beta_2_t"], use_locking=self._use_locking
            )
            with ops.control_dependencies([v_t]):
                v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
            if not self.amsgrad:
                v_sqrt = math_ops.sqrt(v_t)
                var_update = state_ops.assign_sub(
                    var,
                    coefficients["lr"] * m_t / (v_sqrt + coefficients["epsilon"]),
                    use_locking=self._use_locking,
                )
                return control_flow_ops.group(*[var_update, m_t, v_t])
            else:
                v_hat = self.get_slot(var, "vhat")
                v_hat_t = math_ops.maximum(v_hat, v_t)
                with ops.control_dependencies([v_hat_t]):
                    v_hat_t = state_ops.assign(
                        v_hat, v_hat_t, use_locking=self._use_locking
                    )
                v_hat_sqrt = math_ops.sqrt(v_hat_t)
                var_update = state_ops.assign_sub(
                    var,
                    coefficients["lr"] * m_t / (v_hat_sqrt + coefficients["epsilon"]),
                    use_locking=self._use_locking,
                )
                return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

        def get_config(self):
            config = super(Adam, self).get_config()
            config.update(
                {
                    "learning_rate": self._serialize_hyperparameter("learning_rate"),
                    "decay": self._serialize_hyperparameter("decay"),
                    "beta_1": self._serialize_hyperparameter("beta_1"),
                    "beta_2": self._serialize_hyperparameter("beta_2"),
                    "epsilon": self.epsilon,
                    "amsgrad": self.amsgrad,
                }
            )
            return config


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
    kwargs: dict = {}
    _HAS_AGGREGATE_GRAD = True

    def __call__(self):
        self._HAS_AGGREGATE_GRAD = True

        def __init__(
            self,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            name="Adamax",
            **kwargs
        ):
            super(Adamax, self).__init__(self.name, **self.kwargs)
            self._set_hyper("learning_rate", self.kwargs.get("lr", self.learning_rate))
            self._set_hyper("decay", self._initial_decay)
            self._set_hyper("beta_1", self.beta_1)
            self._set_hyper("beta_2", self.beta_2)
            self.epsilon = self.epsilon or backend_config.epsilon()

        def _create_slots(self, var_list):
            for var in var_list:
                self.add_slot(var, "m")
            for var in var_list:
                self.add_slot(var, "v")

        def _prepare_local(self, var_device, var_dtype, apply_state):
            super(Adamax, self)._prepare_local(var_device, var_dtype, apply_state)
            local_step = math_ops.cast(self.iterations + 1, var_dtype)
            beta_1_t = array_ops.identity(self._get_hyper("beta_1", var_dtype))
            beta_2_t = array_ops.identity(self._get_hyper("beta_2", var_dtype))
            beta_1_power = math_ops.pow(beta_1_t, local_step)
            lr_t = apply_state[var_device, var_dtype]["lr_t"]
            apply_state[var_device, var_dtype].update(
                dict(
                    neg_scaled_lr=-lr_t / (1 - beta_1_power),
                    epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                    beta_1_t=beta_1_t,
                    beta_1_power=beta_1_power,
                    one_minus_beta_1_t=1 - beta_1_t,
                    beta_2_t=beta_2_t,
                    zero=array_ops.zeros((), dtype=dtypes.int64),
                )
            )

        def _resource_apply_dense(self, grad, var, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            m = self.get_slot(var, "m")
            v = self.get_slot(var, "v")
            return training_ops.resource_apply_ada_max(
                var.handle,
                m.handle,
                v.handle,
                coefficients["beta_1_power"],
                coefficients["lr_t"],
                coefficients["beta_1_t"],
                coefficients["beta_2_t"],
                coefficients["epsilon"],
                grad,
                use_locking=self._use_locking,
            )

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            m = self.get_slot(var, "m")
            m_slice = array_ops.gather(m, indices, axis=coefficients["zero"])
            m_t_slice = (
                m_slice * coefficients["beta_1_t"]
                + grad * coefficients["one_minus_beta_1_t"]
            )
            with ops.control_dependencies([m_t_slice]):
                m_t = self._resource_scatter_update(m, indices, m_t_slice)
            v = self.get_slot(var, "v")
            v_slice = array_ops.gather(v, indices, axis=coefficients["zero"])
            v_t_slice = math_ops.maximum(
                v_slice * coefficients["beta_2_t"], math_ops.abs(grad)
            )
            with ops.control_dependencies([v_t_slice]):
                v_t = self._resource_scatter_update(v, indices, v_t_slice)
            var_slice = coefficients["neg_scaled_lr"] * (
                m_t_slice / (v_t_slice + coefficients["epsilon"])
            )
            with ops.control_dependencies([var_slice]):
                var_update = self._resource_scatter_add(var, indices, var_slice)
            return control_flow_ops.group(*[var_update, m_t, v_t])

        def get_config(self):
            config = super(Adamax, self).get_config()
            config.update(
                {
                    "learning_rate": self._serialize_hyperparameter("learning_rate"),
                    "decay": self._serialize_hyperparameter("decay"),
                    "beta_1": self._serialize_hyperparameter("beta_1"),
                    "beta_2": self._serialize_hyperparameter("beta_2"),
                    "epsilon": self.epsilon,
                }
            )
            return config


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
    kwargs: dict = {}

    def __call__(self):
        def __init__(
            self,
            learning_rate=0.001,
            learning_rate_power=-0.5,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0,
            name="Ftrl",
            l2_shrinkage_regularization_strength=0.0,
            **kwargs
        ):
            super(Ftrl, self).__init__(self.name, **self.kwargs)
            if self.initial_accumulator_value < 0.0:
                raise ValueError(
                    "initial_accumulator_value %f needs to be positive or zero"
                    % self.initial_accumulator_value
                )
            if self.learning_rate_power > 0.0:
                raise ValueError(
                    "learning_rate_power %f needs to be negative or zero"
                    % self.learning_rate_power
                )
            if self.l1_regularization_strength < 0.0:
                raise ValueError(
                    "l1_regularization_strength %f needs to be positive or zero"
                    % self.l1_regularization_strength
                )
            if self.l2_regularization_strength < 0.0:
                raise ValueError(
                    "l2_regularization_strength %f needs to be positive or zero"
                    % self.l2_regularization_strength
                )
            if self.l2_shrinkage_regularization_strength < 0.0:
                raise ValueError(
                    "l2_shrinkage_regularization_strength %f needs to be positive or zero"
                    % self.l2_shrinkage_regularization_strength
                )
            self._set_hyper("learning_rate", self.learning_rate)
            self._set_hyper("decay", self._initial_decay)
            self._set_hyper("learning_rate_power", self.learning_rate_power)
            self._set_hyper(
                "l1_regularization_strength", self.l1_regularization_strength
            )
            self._set_hyper(
                "l2_regularization_strength", self.l2_regularization_strength
            )
            self._initial_accumulator_value = self.initial_accumulator_value
            self._l2_shrinkage_regularization_strength = (
                self.l2_shrinkage_regularization_strength
            )

        def _create_slots(self, var_list):
            for var in var_list:
                dtype = var.dtype.base_dtype
                init = init_ops.constant_initializer(
                    self._initial_accumulator_value, dtype=dtype
                )
                self.add_slot(var, "accumulator", init)
                self.add_slot(var, "linear")

        def _prepare_local(self, var_device, var_dtype, apply_state):
            super(Ftrl, self)._prepare_local(var_device, var_dtype, apply_state)
            apply_state[var_device, var_dtype].update(
                dict(
                    learning_rate_power=array_ops.identity(
                        self._get_hyper("learning_rate_power", var_dtype)
                    ),
                    l1_regularization_strength=array_ops.identity(
                        self._get_hyper("l1_regularization_strength", var_dtype)
                    ),
                    l2_regularization_strength=array_ops.identity(
                        self._get_hyper("l2_regularization_strength", var_dtype)
                    ),
                    l2_shrinkage_regularization_strength=math_ops.cast(
                        self._l2_shrinkage_regularization_strength, var_dtype
                    ),
                )
            )

        def _resource_apply_dense(self, grad, var, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            accum = self.get_slot(var, "accumulator")
            linear = self.get_slot(var, "linear")
            if self._l2_shrinkage_regularization_strength <= 0.0:
                return training_ops.resource_apply_ftrl(
                    var.handle,
                    accum.handle,
                    linear.handle,
                    grad,
                    coefficients["lr_t"],
                    coefficients["l1_regularization_strength"],
                    coefficients["l2_regularization_strength"],
                    coefficients["learning_rate_power"],
                    use_locking=self._use_locking,
                )
            else:
                return training_ops.resource_apply_ftrl_v2(
                    var.handle,
                    accum.handle,
                    linear.handle,
                    grad,
                    coefficients["lr_t"],
                    coefficients["l1_regularization_strength"],
                    coefficients["l2_regularization_strength"],
                    coefficients["l2_shrinkage_regularization_strength"],
                    coefficients["learning_rate_power"],
                    use_locking=self._use_locking,
                )

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            accum = self.get_slot(var, "accumulator")
            linear = self.get_slot(var, "linear")
            if self._l2_shrinkage_regularization_strength <= 0.0:
                return training_ops.resource_sparse_apply_ftrl(
                    var.handle,
                    accum.handle,
                    linear.handle,
                    grad,
                    indices,
                    coefficients["lr_t"],
                    coefficients["l1_regularization_strength"],
                    coefficients["l2_regularization_strength"],
                    coefficients["learning_rate_power"],
                    use_locking=self._use_locking,
                )
            else:
                return training_ops.resource_sparse_apply_ftrl_v2(
                    var.handle,
                    accum.handle,
                    linear.handle,
                    grad,
                    indices,
                    coefficients["lr_t"],
                    coefficients["l1_regularization_strength"],
                    coefficients["l2_regularization_strength"],
                    coefficients["l2_shrinkage_regularization_strength"],
                    coefficients["learning_rate_power"],
                    use_locking=self._use_locking,
                )

        def get_config(self):
            config = super(Ftrl, self).get_config()
            config.update(
                {
                    "learning_rate": self._serialize_hyperparameter("learning_rate"),
                    "decay": self._serialize_hyperparameter("decay"),
                    "initial_accumulator_value": self._initial_accumulator_value,
                    "learning_rate_power": self._serialize_hyperparameter(
                        "learning_rate_power"
                    ),
                    "l1_regularization_strength": self._serialize_hyperparameter(
                        "l1_regularization_strength"
                    ),
                    "l2_regularization_strength": self._serialize_hyperparameter(
                        "l2_regularization_strength"
                    ),
                    "l2_shrinkage_regularization_strength": self._l2_shrinkage_regularization_strength,
                }
            )
            return config


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
    kwargs: dict = {}
    _HAS_AGGREGATE_GRAD = True

    def __call__(self):
        self._HAS_AGGREGATE_GRAD = True

        def __init__(
            self,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            name="Nadam",
            **kwargs
        ):
            self.kwargs["decay"] = self.kwargs.pop("schedule_decay", 0.004)
            self.learning_rate = self.kwargs.get("lr", self.learning_rate)
            if isinstance(
                self.learning_rate, learning_rate_schedule.LearningRateSchedule
            ):
                raise ValueError(
                    "The Nadam optimizer does not support tf.keras.optimizers.LearningRateSchedules as the learning rate."
                )
            super(Nadam, self).__init__(self.name, **self.kwargs)
            self._set_hyper("learning_rate", self.kwargs.get("lr", self.learning_rate))
            self._set_hyper("decay", self._initial_decay)
            self._set_hyper("beta_1", self.beta_1)
            self._set_hyper("beta_2", self.beta_2)
            self.epsilon = self.epsilon or backend_config.epsilon()
            self._m_cache = None

        def _create_slots(self, var_list):
            var_dtype = var_list[0].dtype.base_dtype
            if self._m_cache is None:
                self._m_cache = self.add_weight(
                    "momentum_cache",
                    shape=[],
                    dtype=var_dtype,
                    initializer="ones",
                    trainable=False,
                    aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA,
                )
                self._weights.append(self._m_cache)
            for var in var_list:
                self.add_slot(var, "m")
            for var in var_list:
                self.add_slot(var, "v")

        def _prepare_local(self, var_device, var_dtype, apply_state):
            lr_t = array_ops.identity(self._get_hyper("learning_rate", var_dtype))
            beta_1_t = array_ops.identity(self._get_hyper("beta_1", var_dtype))
            beta_2_t = array_ops.identity(self._get_hyper("beta_2", var_dtype))
            local_step = math_ops.cast(self.iterations + 1, var_dtype)
            next_step = math_ops.cast(self.iterations + 2, var_dtype)
            decay_base = math_ops.cast(0.96, var_dtype)
            m_t = beta_1_t * (
                1.0 - 0.5 * math_ops.pow(decay_base, self._initial_decay * local_step)
            )
            m_t_1 = beta_1_t * (
                1.0 - 0.5 * math_ops.pow(decay_base, self._initial_decay * next_step)
            )
            m_schedule_new = math_ops.cast(self._m_cache_read, var_dtype) * m_t
            if var_dtype is self._m_cache.dtype:
                m_schedule_new = array_ops.identity(
                    state_ops.assign(
                        self._m_cache, m_schedule_new, use_locking=self._use_locking
                    )
                )
            m_schedule_next = m_schedule_new * m_t_1
            apply_state[var_device, var_dtype] = dict(
                lr_t=lr_t,
                neg_lr_t=-lr_t,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_2_t=beta_2_t,
                m_t=m_t,
                m_t_1=m_t_1,
                one_minus_beta_1_t=1 - beta_1_t,
                one_minus_beta_2_t=1 - beta_2_t,
                one_minus_m_t=1.0 - m_t,
                one_minus_m_schedule_new=1.0 - m_schedule_new,
                one_minus_m_schedule_next=1.0 - m_schedule_next,
                v_t_prime_denominator=1.0 - math_ops.pow(beta_2_t, local_step),
            )

        def _prepare(self, var_list):
            self._m_cache_read = array_ops.identity(self._m_cache)
            return super(Nadam, self)._prepare(var_list)

        def _resource_apply_dense(self, grad, var, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            m = self.get_slot(var, "m")
            v = self.get_slot(var, "v")
            g_prime = grad / coefficients["one_minus_m_schedule_new"]
            m_t = (
                coefficients["beta_1_t"] * m + coefficients["one_minus_beta_1_t"] * grad
            )
            m_t = state_ops.assign(m, m_t, use_locking=self._use_locking)
            m_t_prime = m_t / coefficients["one_minus_m_schedule_next"]
            v_t = coefficients["beta_2_t"] * v + coefficients[
                "one_minus_beta_2_t"
            ] * math_ops.square(grad)
            v_t = state_ops.assign(v, v_t, use_locking=self._use_locking)
            v_t_prime = v_t / coefficients["v_t_prime_denominator"]
            m_t_bar = (
                coefficients["one_minus_m_t"] * g_prime
                + coefficients["m_t_1"] * m_t_prime
            )
            var_t = var - coefficients["lr_t"] * m_t_bar / (
                math_ops.sqrt(v_t_prime) + coefficients["epsilon"]
            )
            return state_ops.assign(var, var_t, use_locking=self._use_locking).op

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            m = self.get_slot(var, "m")
            v = self.get_slot(var, "v")
            g_prime = grad / coefficients["one_minus_m_schedule_new"]
            m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
            m_t = state_ops.assign(
                m, m * coefficients["beta_1_t"], use_locking=self._use_locking
            )
            with ops.control_dependencies([m_t]):
                m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
                m_t_slice = array_ops.gather(m_t, indices)
            m_t_prime = m_t_slice / coefficients["one_minus_m_schedule_next"]
            m_t_bar = (
                coefficients["one_minus_m_t"] * g_prime
                + coefficients["m_t_1"] * m_t_prime
            )
            v_scaled_g_values = grad * grad * coefficients["one_minus_beta_2_t"]
            v_t = state_ops.assign(
                v, v * coefficients["beta_2_t"], use_locking=self._use_locking
            )
            with ops.control_dependencies([v_t]):
                v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
                v_t_slice = array_ops.gather(v_t, indices)
            v_t_prime = v_t_slice / coefficients["v_t_prime_denominator"]
            v_prime_sqrt_plus_eps = math_ops.sqrt(v_t_prime) + coefficients["epsilon"]
            var_update = self._resource_scatter_add(
                var, indices, coefficients["neg_lr_t"] * m_t_bar / v_prime_sqrt_plus_eps
            )
            return control_flow_ops.group(*[var_update, m_t_bar, v_t])

        def get_config(self):
            config = super(Nadam, self).get_config()
            config.update(
                {
                    "learning_rate": self._serialize_hyperparameter("learning_rate"),
                    "decay": self._serialize_hyperparameter("decay"),
                    "beta_1": self._serialize_hyperparameter("beta_1"),
                    "beta_2": self._serialize_hyperparameter("beta_2"),
                    "epsilon": self.epsilon,
                }
            )
            return config


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
    kwargs: dict = {}
    _HAS_AGGREGATE_GRAD = True

    def __call__(self):
        self._HAS_AGGREGATE_GRAD = True

        def __init__(
            self,
            learning_rate=0.001,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False,
            name="RMSprop",
            **kwargs
        ):
            """Construct a new RMSprop optimizer.

            Args:
              learning_rate: A `Tensor`, floating point value, or a schedule that is a
                `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
                that takes no arguments and returns the actual value to use. The
                learning rate. Defeaults to 0.001.
              rho: Discounting factor for the history/coming gradient. Defaults to 0.9.
              momentum: A scalar or a scalar `Tensor`. Defaults to 0.0.
              epsilon: A small constant for numerical stability. This epsilon is
                "epsilon hat" in the Kingma and Ba paper (in the formula just before
                Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
                1e-7.
              centered: Boolean. If `True`, gradients are normalized by the estimated
                variance of the gradient; if False, by the uncentered second moment.
                Setting this to `True` may help with training, but is slightly more
                expensive in terms of computation and memory. Defaults to `False`.
              name: Optional name prefix for the operations created when applying
                gradients. Defaults to "RMSprop".
              **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
                `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
                gradients by value, `decay` is included for backward compatibility to
                allow time inverse decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.

            @compatibility(eager)
            When eager execution is enabled, `learning_rate`, `decay`, `momentum`, and
            `epsilon` can each be a callable that takes no arguments and returns the
            actual value to use. This can be useful for changing these values across
            different invocations of optimizer functions.
            @end_compatibility
            """
            super(RMSprop, self).__init__(self.name, **self.kwargs)
            self._set_hyper("learning_rate", self.kwargs.get("lr", self.learning_rate))
            self._set_hyper("decay", self._initial_decay)
            self._set_hyper("rho", self.rho)
            self._momentum = False
            if (
                isinstance(self.momentum, ops.Tensor)
                or callable(self.momentum)
                or self.momentum > 0
            ):
                self._momentum = True
            if isinstance(self.momentum, (int, float)) and (
                self.momentum < 0 or self.momentum > 1
            ):
                raise ValueError("`momentum` must be between [0, 1].")
            self._set_hyper("momentum", self.momentum)
            self.epsilon = self.epsilon or backend_config.epsilon()
            self.centered = self.centered

        def _create_slots(self, var_list):
            for var in var_list:
                self.add_slot(var, "rms")
            if self._momentum:
                for var in var_list:
                    self.add_slot(var, "momentum")
            if self.centered:
                for var in var_list:
                    self.add_slot(var, "mg")

        def _prepare_local(self, var_device, var_dtype, apply_state):
            super(RMSprop, self)._prepare_local(var_device, var_dtype, apply_state)
            self.rho = array_ops.identity(self._get_hyper("rho", var_dtype))
            apply_state[var_device, var_dtype].update(
                dict(
                    neg_lr_t=-apply_state[var_device, var_dtype]["lr_t"],
                    epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                    rho=self.rho,
                    momentum=array_ops.identity(self._get_hyper("momentum", var_dtype)),
                    one_minus_rho=1.0 - self.rho,
                )
            )

        def _resource_apply_dense(self, grad, var, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            rms = self.get_slot(var, "rms")
            if self._momentum:
                mom = self.get_slot(var, "momentum")
                if self.centered:
                    mg = self.get_slot(var, "mg")
                    return training_ops.resource_apply_centered_rms_prop(
                        var.handle,
                        mg.handle,
                        rms.handle,
                        mom.handle,
                        coefficients["lr_t"],
                        coefficients["rho"],
                        coefficients["momentum"],
                        coefficients["epsilon"],
                        grad,
                        use_locking=self._use_locking,
                    )
                else:
                    return training_ops.resource_apply_rms_prop(
                        var.handle,
                        rms.handle,
                        mom.handle,
                        coefficients["lr_t"],
                        coefficients["rho"],
                        coefficients["momentum"],
                        coefficients["epsilon"],
                        grad,
                        use_locking=self._use_locking,
                    )
            else:
                rms_t = coefficients["rho"] * rms + coefficients[
                    "one_minus_rho"
                ] * math_ops.square(grad)
                rms_t = state_ops.assign(rms, rms_t, use_locking=self._use_locking)
                denom_t = rms_t
                if self.centered:
                    mg = self.get_slot(var, "mg")
                    mg_t = (
                        coefficients["rho"] * mg + coefficients["one_minus_rho"] * grad
                    )
                    mg_t = state_ops.assign(mg, mg_t, use_locking=self._use_locking)
                    denom_t = rms_t - math_ops.square(mg_t)
                var_t = var - coefficients["lr_t"] * grad / (
                    math_ops.sqrt(denom_t) + coefficients["epsilon"]
                )
                return state_ops.assign(var, var_t, use_locking=self._use_locking).op

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
            ) or self._fallback_apply_state(var_device, var_dtype)
            rms = self.get_slot(var, "rms")
            if self._momentum:
                mom = self.get_slot(var, "momentum")
                if self.centered:
                    mg = self.get_slot(var, "mg")
                    return training_ops.resource_sparse_apply_centered_rms_prop(
                        var.handle,
                        mg.handle,
                        rms.handle,
                        mom.handle,
                        coefficients["lr_t"],
                        coefficients["rho"],
                        coefficients["momentum"],
                        coefficients["epsilon"],
                        grad,
                        indices,
                        use_locking=self._use_locking,
                    )
                else:
                    return training_ops.resource_sparse_apply_rms_prop(
                        var.handle,
                        rms.handle,
                        mom.handle,
                        coefficients["lr_t"],
                        coefficients["rho"],
                        coefficients["momentum"],
                        coefficients["epsilon"],
                        grad,
                        indices,
                        use_locking=self._use_locking,
                    )
            else:
                rms_scaled_g_values = grad * grad * coefficients["one_minus_rho"]
                rms_t = state_ops.assign(
                    rms, rms * coefficients["rho"], use_locking=self._use_locking
                )
                with ops.control_dependencies([rms_t]):
                    rms_t = self._resource_scatter_add(
                        rms, indices, rms_scaled_g_values
                    )
                    rms_slice = array_ops.gather(rms_t, indices)
                denom_slice = rms_slice
                if self.centered:
                    mg = self.get_slot(var, "mg")
                    mg_scaled_g_values = grad * coefficients["one_minus_rho"]
                    mg_t = state_ops.assign(
                        mg, mg * coefficients["rho"], use_locking=self._use_locking
                    )
                    with ops.control_dependencies([mg_t]):
                        mg_t = self._resource_scatter_add(
                            mg, indices, mg_scaled_g_values
                        )
                        mg_slice = array_ops.gather(mg_t, indices)
                        denom_slice = rms_slice - math_ops.square(mg_slice)
                var_update = self._resource_scatter_add(
                    var,
                    indices,
                    coefficients["neg_lr_t"]
                    * grad
                    / (math_ops.sqrt(denom_slice) + coefficients["epsilon"]),
                )
                if self.centered:
                    return control_flow_ops.group(*[var_update, rms_t, mg_t])
                return control_flow_ops.group(*[var_update, rms_t])

        def set_weights(self, weights):
            params = self.weights
            if len(params) == len(weights) + 1:
                weights = [np.array(0)] + weights
            super(RMSprop, self).set_weights(weights)

        def get_config(self):
            config = super(RMSprop, self).get_config()
            config.update(
                {
                    "learning_rate": self._serialize_hyperparameter("learning_rate"),
                    "decay": self._serialize_hyperparameter("decay"),
                    "rho": self._serialize_hyperparameter("rho"),
                    "momentum": self._serialize_hyperparameter("momentum"),
                    "epsilon": self.epsilon,
                    "centered": self.centered,
                }
            )
            return config
