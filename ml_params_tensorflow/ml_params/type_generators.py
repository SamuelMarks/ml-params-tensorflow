"""
Type generators. Use these to generate the type annotations throughout this ml-params implementation.

Install doctrans then run, for example:

    python -m doctrans sync_properties \
                       --input-file 'ml_params_tensorflow/ml_params/type_generators.py' \
                       --input-eval \
                       --output-file 'ml_params_tensorflow/ml_params/trainer.py' \
                       --input-param 'exposed_loss_keys' \
                       --output-param 'TensorFlowTrainer.train.loss' \
                       --input-param 'exposed_optimizer_keys' \
                       --output-param 'TensorFlowTrainer.train.optimizer' \
                       --input-param 'exposed_datasets_keys' \
                       --output-param 'TensorFlowTrainer.load_data.dataset_name'

    python -m doctrans sync_properties \
                       --input-file 'ml_params_tensorflow/ml_params/type_generators.py' \
                       --input-eval \
                       --output-file 'ml_params_tensorflow/ml_params/trainer.py' \
                       --output-param-wrap 'Optional[List[{output_param}]]' \
                       --input-param 'exposed_callbacks_keys' \
                       --output-param 'TensorFlowTrainer.train.callbacks' \
                       --input-param 'exposed_metrics_keys' \
                       --output-param 'TensorFlowTrainer.train.metrics'

    python -m doctrans sync_properties \
                       --input-file 'ml_params_tensorflow/ml_params/type_generators.py' \
                       --input-eval \
                       --output-file 'ml_params_tensorflow/ml_params/trainer.py' \
                       --input-param 'exposed_loss_keys' \
                       --output-param 'TensorFlowTrainer.train.loss' \
                       --input-param 'exposed_optimizer_keys' \
                       --output-param 'TensorFlowTrainer.train.optimizer' \
                       --input-param 'exposed_datasets_keys' \
                       --output-param 'TensorFlowTrainer.load_data.dataset_name'

    python -m doctrans sync_properties \
                       --input-file 'ml_params_tensorflow/ml_params/type_generators.py' \
                       --input-eval \
                       --output-file 'ml_params_tensorflow/ml_params/trainer.py' \
                       --input-param 'exposed_applications_keys' \
                       --output-param-wrap 'Union[{output_param}, AnyStr]' \
                       --output-param 'TensorFlowTrainer.load_model.model'

    python -m doctrans sync_properties \
                       --input-file 'ml_params_tensorflow/ml_params/type_generators.py' \
                       --input-eval \
                       --output-file 'ml_params_tensorflow/ml_params/datasets.py' \
                       --output-param-wrap 'Union[{output_param}, AnyStr]' \
                       --input-param 'exposed_datasets_keys' \
                       --output-param 'load_data_from_tfds_or_ml_prepare.dataset_name'
"""
from typing import Any, AnyStr, Callable, Dict, Optional, Tuple

import tensorflow as tf


def _is_main_name(param: str, extra_filter=None) -> bool:
    """
    TensorFlow follows a rough naming convention, this checks if the param follows it

    :param param: The attribute name

    :param extra_filter: Any additional filter to run
    :type extra_filter: Optional[Callable[[AnyStr], bool]]

    :return: Whether it is a 'main' name
    """
    return (
        not param.startswith("_")
        and "_" not in param
        and not param.islower()
        and not param.isupper()
        and (extra_filter is None or extra_filter(param))
    )


def _expose_module(
    mod: Any,
    exclusions: frozenset = frozenset(),
    extra_filter: Optional[Callable[[AnyStr], bool]] = None,
) -> Dict[AnyStr, Any]:
    """
    :param mod: Any module

    :param exclusions: Any attributes to exclude

    :param extra_filter: Any additional filter to run

    :return: Mapping from name to object at identified by that name
    """
    return {
        name: getattr(mod, name)
        for name in dir(mod)
        if name not in exclusions and _is_main_name(name, extra_filter)
    }


_global_exclude: frozenset = frozenset(
    ("deserialize", "serialize", "get", "experimental", "schedules")
)

exposed_activations: Dict[str, Any] = {
    name: getattr(tf.keras.activations, name)
    for name in dir(tf.keras.activations)
    if name not in _global_exclude and not name.startswith("_")
}
exposed_activations_keys: Tuple[str, ...] = tuple(sorted(exposed_activations.keys()))

exposed_applications: Dict[str, Any] = _expose_module(tf.keras.applications)
exposed_applications_keys: Tuple[str, ...] = tuple(sorted(exposed_applications.keys()))

exposed_callbacks: Dict[str, Any] = _expose_module(tf.keras.callbacks)
exposed_callbacks_keys: Tuple[str, ...] = tuple(sorted(exposed_callbacks.keys()))

exposed_constraints: Dict[str, Any] = _expose_module(tf.keras.constraints)
exposed_constraints_keys: Tuple[str, ...] = tuple(sorted(exposed_constraints.keys()))

exposed_datasets: Dict[str, Any] = {
    name: getattr(tf.keras.datasets, name)
    for name in dir(tf.keras.datasets)
    if not name.startswith("_")
}
exposed_datasets_keys: Tuple[str, ...] = tuple(sorted(exposed_datasets.keys()))

exposed_initializers: Dict[str, Any] = _expose_module(tf.keras.initializers)
exposed_initializers_keys: Tuple[str, ...] = tuple(sorted(exposed_initializers.keys()))

exposed_layers: Dict[str, Any] = {
    name: getattr(tf.keras.layers, name)
    for name in dir(tf.keras.layers)
    if name not in _global_exclude and not name.startswith("_")
}
exposed_layers_keys: Tuple[str, ...] = tuple(sorted(exposed_layers.keys()))

exposed_losses: Dict[str, Any] = {
    name: getattr(tf.keras.losses, name)
    for name in dir(tf.keras.losses)
    if not name.startswith("_")
    and name not in frozenset(("Loss",)) | _global_exclude
    and "_" in name
}
exposed_loss_keys: Tuple[str, ...] = tuple(sorted(exposed_losses.keys()))

exposed_metrics: Dict[str, Any] = {
    metric: getattr(tf.keras.metrics, metric)
    for metric in dir(tf.keras.metrics)
    if not metric.startswith("_")
    and metric.islower()
    and metric not in frozenset(("Metric",)) | _global_exclude
}
exposed_metrics_keys: Tuple[str, ...] = tuple(sorted(exposed_metrics.keys()))

exposed_optimizers: Dict[str, Any] = _expose_module(
    tf.keras.optimizers, frozenset(("Optimizer",))
)
exposed_optimizer_keys: Tuple[str, ...] = tuple(sorted(exposed_optimizers.keys()))

exposed_regularizers: Dict[str, Any] = _expose_module(tf.keras.regularizers)
exposed_regularizers_keys: Tuple[str, ...] = tuple(sorted(exposed_regularizers.keys()))
