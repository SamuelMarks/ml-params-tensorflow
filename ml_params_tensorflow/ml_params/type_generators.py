"""
Type generators. Use these to generate the type annotations throughout this ml-params implementation.

Install cdd then run, for example:

    python -m cdd sync_properties \
                       --input-file 'ml_params_tensorflow/ml_params/type_generators.py' \
                       --input-eval \
                       --output-file 'ml_params_tensorflow/ml_params/trainer.py' \
                       --input-param 'exposed_loss_keys' \
                       --output-param 'TensorFlowTrainer.train.loss' \
                       --input-param 'exposed_optimizer_keys' \
                       --output-param 'TensorFlowTrainer.train.optimizer' \
                       --input-param 'exposed_datasets_keys' \
                       --output-param 'TensorFlowTrainer.load_data.dataset_name'

    python -m cdd sync_properties \
                       --input-file 'ml_params_tensorflow/ml_params/type_generators.py' \
                       --input-eval \
                       --output-file 'ml_params_tensorflow/ml_params/trainer.py' \
                       --output-param-wrap 'Optional[List[{output_param}]]' \
                       --input-param 'exposed_callbacks_keys' \
                       --output-param 'TensorFlowTrainer.train.callbacks' \
                       --input-param 'exposed_metrics_keys' \
                       --output-param 'TensorFlowTrainer.train.metrics'

    python -m cdd sync_properties \
                       --input-file 'ml_params_tensorflow/ml_params/type_generators.py' \
                       --input-eval \
                       --output-file 'ml_params_tensorflow/ml_params/trainer.py' \
                       --input-param 'exposed_loss_keys' \
                       --output-param 'TensorFlowTrainer.train.loss' \
                       --input-param 'exposed_optimizer_keys' \
                       --output-param 'TensorFlowTrainer.train.optimizer' \
                       --input-param 'exposed_datasets_keys' \
                       --output-param 'TensorFlowTrainer.load_data.dataset_name'

    python -m cdd sync_properties \
                       --input-file 'ml_params_tensorflow/ml_params/type_generators.py' \
                       --input-eval \
                       --output-file 'ml_params_tensorflow/ml_params/trainer.py' \
                       --input-param 'exposed_applications_keys' \
                       --output-param-wrap 'Union[{output_param}, AnyStr]' \
                       --output-param 'TensorFlowTrainer.load_model.model'

    python -m cdd sync_properties \
                       --input-file 'ml_params_tensorflow/ml_params/type_generators.py' \
                       --input-eval \
                       --output-file 'ml_params_tensorflow/ml_params/datasets.py' \
                       --output-param-wrap 'Union[{output_param}, AnyStr]' \
                       --input-param 'exposed_datasets_keys' \
                       --output-param 'load_data_from_tfds_or_ml_prepare.dataset_name'
"""
from collections import deque
from functools import partial
from itertools import filterfalse, groupby
from operator import contains, itemgetter
from types import ModuleType
from typing import Any, AnyStr, Callable, Dict, Iterator, Optional, Tuple, Union

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


def _unique_members(mod: Union[ModuleType, Any]) -> Iterator[Tuple[Any, ...]]:
    """
    In TensorFlow there are duplicates, like `binary_accuracy` and `BinaryAccuracy`.

    To avoid over-generating so that the CLI help text is too long, this function returns only one of these.

    (preference towards lowercase underscore separated form)

    :param mod: The module to run `dir` or `getmembers` against

    :return: Deduplicated members
    """

    def cmp(s: str) -> str:
        """
        Used as a comparator, returns a lowercase variant without underscores

        :param s: The input string

        :return: lowercase variant without underscores
        """
        return s.replace("_", "").casefold()

    return map(
        itemgetter(1),
        (
            (k, deque(g, maxlen=1)[0])
            for k, g in groupby(
                sorted(
                    (attr for attr in dir(mod) if not attr.startswith("_")),
                    key=cmp,
                ),
                key=cmp,
            )
        ),
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

exposed_activations_keys: Tuple[str, ...] = tuple(
    filterfalse(
        partial(contains, _global_exclude),
        _unique_members(tf.keras.activations),
    )
)
exposed_activations: Dict[str, Any] = dict(
    map(
        lambda attr: (attr, getattr(tf.keras.activations, attr)),
        exposed_activations_keys,
    )
)

exposed_applications: Dict[str, Any] = _expose_module(tf.keras.applications)
exposed_applications_keys: Tuple[str, ...] = tuple(sorted(exposed_applications.keys()))

exposed_callbacks: Dict[str, Any] = _expose_module(tf.keras.callbacks)
exposed_callbacks_keys: Tuple[str, ...] = tuple(sorted(exposed_callbacks.keys()))

exposed_constraints: Dict[str, Any] = _expose_module(tf.keras.constraints)
exposed_constraints_keys: Tuple[str, ...] = tuple(sorted(exposed_constraints.keys()))

exposed_datasets: Dict[str, Any] = {
    attr: getattr(tf.keras.datasets, attr)
    for attr in dir(tf.keras.datasets)
    if not attr.startswith("_")
}
exposed_datasets_keys: Tuple[str, ...] = tuple(sorted(exposed_datasets.keys()))

exposed_initializers: Dict[str, Any] = _expose_module(tf.keras.initializers)
exposed_initializers_keys: Tuple[str, ...] = tuple(sorted(exposed_initializers.keys()))

exposed_layers_keys: Tuple[str, ...] = tuple(
    filterfalse(
        partial(contains, _global_exclude),
        _unique_members(tf.keras.layers),
    )
)
exposed_layers: Dict[str, Any] = dict(
    map(
        lambda attr: (attr, getattr(tf.keras.layers, attr)),
        exposed_layers_keys,
    )
)

exposed_loss_keys: Tuple[str, ...] = tuple(
    filterfalse(
        partial(contains, frozenset(("Loss",)) | _global_exclude),
        _unique_members(tf.keras.losses),
    )
)
exposed_losses: Dict[str, Any] = dict(
    map(
        lambda attr: (attr, getattr(tf.keras.losses, attr)),
        exposed_loss_keys,
    )
)

exposed_metrics_keys: Tuple[str, ...] = tuple(
    filterfalse(
        partial(contains, frozenset(("Accuracy", "Metric")) | _global_exclude),
        _unique_members(tf.keras.metrics),
    )
)
exposed_metrics: Dict[str, Any] = dict(
    map(
        lambda attr: (attr, getattr(tf.keras.metrics, attr)),
        exposed_metrics_keys,
    )
)

exposed_optimizers: Dict[str, Any] = _expose_module(
    tf.keras.optimizers, frozenset(("Optimizer",))
)
exposed_optimizer_keys: Tuple[str, ...] = tuple(sorted(exposed_optimizers.keys()))

exposed_regularizers: Dict[str, Any] = _expose_module(tf.keras.regularizers)
exposed_regularizers_keys: Tuple[str, ...] = tuple(sorted(exposed_regularizers.keys()))

__all__ = [
    "exposed_activations",
    "exposed_activations_keys",
    "exposed_applications",
    "exposed_applications_keys",
    "exposed_callbacks",
    "exposed_callbacks_keys",
    "exposed_constraints",
    "exposed_constraints_keys",
    "exposed_datasets",
    "exposed_datasets_keys",
    "exposed_initializers",
    "exposed_initializers_keys",
    "exposed_layers",
    "exposed_layers_keys",
    "exposed_loss_keys",
    "exposed_losses",
    "exposed_metrics",
    "exposed_metrics_keys",
    "exposed_optimizer_keys",
    "exposed_optimizers",
    "exposed_regularizers",
    "exposed_regularizers_keys",
]
