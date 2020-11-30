"""
Config interface to ml-params-tensorflow. Expected to be bootstrapped by ml-params, as well as internally.
"""

from dataclasses import dataclass
from json import loads
from typing import Any, Optional, List, Callable, AnyStr, Union, Tuple, Dict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from enforce import runtime_validation
from typing_extensions import Literal


# @dataclass()
class self(object):
    model: Any = None
    data: Any = None


def from_string(cls, s):
    return cls(**loads(s))


def run_typed(f):
    f.from_string = classmethod(from_string)
    f.__argparse__ = dict(from_string=f.from_string)
    return runtime_validation(dataclass(f))


# @s(auto_attribs=True)
# @run_typed
@runtime_validation
@dataclass
class TrainConfig(object):
    """
    Run the training loop for your ML pipeline.

    :cvar epochs: number of epochs (must be greater than 0)
    :cvar loss: Loss function, can be a string (depending on the framework) or an instance of a class
    :cvar optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
    :cvar callbacks: Collection of callables that are run inside the training loop. Defaults to None
    :cvar metrics: Collection of metrics to monitor, e.g., accuracy, f1. Defaults to None
    :cvar metric_emit_freq: `None` for every epoch. E.g., `eq(mod(epochs, 10), 0)` for every 10. Defaults to None
    :cvar save_directory: Directory to save output in, e.g., weights in h5 files. If None, don't save. Defaults to None
    :cvar output_type: `if save_directory is not None` then save in this format, e.g., 'h5'. Defaults to infer
    :cvar validation_split: Optional float between 0 and 1, fraction of data to reserve for validation. Defaults to 0.1
    :cvar batch_size: batch size at each iteration. Defaults to 128
    :cvar kwargs: additional keyword arguments
    :cvar return_type: the model. Defaults to self.model"""

    epochs: int = 0
    loss: Literal[
        "BinaryCrossentropy",
        "CategoricalCrossentropy",
        "CategoricalHinge",
        "CosineSimilarity",
        "Hinge",
        "Huber",
        "KLDivergence",
        "LogCosh",
        "MeanAbsoluteError",
        "MeanAbsolutePercentageError",
        "MeanSquaredError",
        "MeanSquaredLogarithmicError",
        "Poisson",
        "Reduction",
        "SparseCategoricalCrossentropy",
        "SquaredHinge",
    ] = None
    optimizer: Literal[
        "Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop"
    ] = None
    callbacks: Optional[
        List[
            Literal[
                "BaseLogger",
                "CSVLogger",
                "Callback",
                "CallbackList",
                "EarlyStopping",
                "History",
                "LambdaCallback",
                "LearningRateScheduler",
                "ModelCheckpoint",
                "ProgbarLogger",
                "ReduceLROnPlateau",
                "RemoteMonitor",
                "TensorBoard",
                "TerminateOnNaN",
            ]
        ]
    ] = None
    metrics: Optional[
        List[
            Literal[
                "binary_accuracy",
                "binary_crossentropy",
                "categorical_accuracy",
                "categorical_crossentropy",
                "hinge",
                "kl_divergence",
                "kld",
                "kullback_leibler_divergence",
                "mae",
                "mape",
                "mean_absolute_error",
                "mean_absolute_percentage_error",
                "mean_squared_error",
                "mean_squared_logarithmic_error",
                "mse",
                "msle",
                "poisson",
                "sparse_categorical_accuracy",
                "sparse_categorical_crossentropy",
                "sparse_top_k_categorical_accuracy",
                "squared_hinge",
                "top_k_categorical_accuracy",
            ]
        ]
    ] = None
    metric_emit_freq: Optional[Callable[[int], bool]] = None
    save_directory: Optional[str] = None
    output_type: str = "infer"
    validation_split: float = 0.1
    batch_size: int = 128
    kwargs: Optional[dict] = None
    return_type: Any = self.model


@run_typed
class LoadDataConfig(object):
    """
    Load the data for your ML pipeline. Will be fed into `train`.

    :cvar dataset_name: name of dataset
    :cvar data_loader: function that returns the expected data type. Defaults to None
    :cvar data_type: incoming data type. Defaults to infer
    :cvar output_type: outgoing data_type. Defaults to None
    :cvar K: backend engine, e.g., `np` or `tf`. Defaults to None
    :cvar data_loader_kwargs: pass this as arguments to data_loader function
    :cvar return_type: Dataset splits (by default, your train and test). Defaults to self.data"""

    dataset_name: Literal[
        "boston_housing",
        "cifar10",
        "cifar100",
        "fashion_mnist",
        "imdb",
        "mnist",
        "reuters",
    ] = None
    data_loader: Optional[
        Callable[
            [AnyStr, AnyStr, Literal["np", "tf"], bool, Dict],
            Union[
                Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo],
                Tuple[np.ndarray, np.ndarray, Any],
                Tuple[Any, Any, Any],
            ],
        ]
    ] = None
    data_type: str = "infer"
    output_type: Optional[Literal["np"]] = None
    K: Optional[Literal["np", "tf"]] = None
    data_loader_kwargs: Optional[dict] = None
    return_type: Union[
        Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]
    ] = self.data


@run_typed
class LoadModelConfig(object):
    """
        Load the model.
    Takes a model object, or a pipeline that downloads & configures before returning a model object.

        :cvar model: model object, e.g., a tf.keras.Sequential, tl.Serial,  nn.Module instance
        :cvar call: whether to call `model()` even if `len(model_kwargs) == 0`. Defaults to False
        :cvar model_kwargs: to be passed into the model. If empty, doesn't call, unless call=True.
        :cvar return_type: self.model, e.g., the result of applying `model_kwargs` on model. Defaults to self.model"""

    model: Union[
        Literal[
            "DenseNet121",
            "DenseNet169",
            "DenseNet201",
            "EfficientNetB0",
            "EfficientNetB1",
            "EfficientNetB2",
            "EfficientNetB3",
            "EfficientNetB4",
            "EfficientNetB5",
            "EfficientNetB6",
            "EfficientNetB7",
            "InceptionResNetV2",
            "InceptionV3",
            "MobileNet",
            "MobileNetV2",
            "NASNetLarge",
            "NASNetMobile",
            "ResNet101",
            "ResNet101V2",
            "ResNet152",
            "ResNet152V2",
            "ResNet50",
            "ResNet50V2",
            "Xception",
        ],
        AnyStr,
    ] = None
    call: bool = False
    model_kwargs: Optional[dict] = None
    return_type: tf.keras.Model = self.model


__all__ = ["LoadDataConfig", "LoadModelConfig", "TrainConfig"]
