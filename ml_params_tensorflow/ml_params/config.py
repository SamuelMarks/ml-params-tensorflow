"""
Config interface to ml-params-tensorflow. Expected to be bootstrapped by ml-params, as well as internally.
"""
from json import loads
from typing import Any, AnyStr, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing_extensions import Literal


class self(object):
    """
    Simple class to proxy object expected by code generated from `train` function

    :cvar model: The model (probably a `tf.keras.models.Sequential`)
    :cvar data: The data (probably a `tf.data.Dataset`)
    """

    model: Any = None
    data: Any = None


model = self.model


def from_string(cls, s):
    """
    Generate a new object of the class using a loaded s

    :param cls: The class to create
    :type cls: ```Callable[[Any, ...], Any]```

    :param s: The arguments string to be parsed
    :type s: ```str```

    :return: Constructed class
    :rtype: ```Any```
    """
    return cls(**loads(s))


# def run_typed(f):
#     """
#     Decorate to validate the input class properties
#
#     :param f: Function or class
#     :type f: ```Any```
#
#     :return: Object that will now validate its input
#     :rtype: ```Any```
#     """
#     f.from_string = classmethod(from_string)
#     f.__argparse__ = dict(from_string=f.from_string)
#     return runtime_validation(dataclass(f))


class TrainConfig(object):
    """
    Run the training loop for your ML pipeline.

    :cvar epochs: number of epochs (must be greater than 0)
    :cvar loss: Loss function, can be a string (depending on the framework) or an instance of a class
    :cvar optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
    :cvar callbacks: Collection of callables that are run inside the training loop
    :cvar metrics: Collection of metrics to monitor, e.g., accuracy, f1
    :cvar metric_emit_freq: `None` for every epoch. E.g., `eq(mod(epochs, 10), 0)` for every 10.
    :cvar output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.
    :cvar validation_split: Optional float between 0 and 1, fraction of data to reserve for validation.
    :cvar batch_size: batch size at each iteration.
    :cvar tpu_address: Address of TPU cluster. If None, don't connect & run within TPU context.
    :cvar kwargs: additional keyword arguments
    :cvar return_type: the model"""

    epochs: int = 0
    loss: Literal[
        "binary_crossentropy",
        "categorical_crossentropy",
        "categorical_hinge",
        "cosine_similarity",
        "hinge",
        "huber",
        "kld",
        "kl_divergence",
        "kullback_leibler_divergence",
        "logcosh",
        "Loss",
        "mae",
        "mape",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "mean_squared_logarithmic_error",
        "mse",
        "msle",
        "poisson",
        "Reduction",
        "sparse_categorical_crossentropy",
        "squared_hinge",
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
                "AUC",
                "binary_accuracy",
                "binary_crossentropy",
                "categorical_accuracy",
                "categorical_crossentropy",
                "CategoricalHinge",
                "CosineSimilarity",
                "FalseNegatives",
                "FalsePositives",
                "hinge",
                "kld",
                "kl_divergence",
                "kullback_leibler_divergence",
                "logcosh",
                "LogCoshError",
                "mae",
                "mape",
                "Mean",
                "mean_absolute_error",
                "mean_absolute_percentage_error",
                "MeanIoU",
                "MeanRelativeError",
                "mean_squared_error",
                "mean_squared_logarithmic_error",
                "MeanTensor",
                "mse",
                "msle",
                "poisson",
                "Precision",
                "PrecisionAtRecall",
                "Recall",
                "RecallAtPrecision",
                "RootMeanSquaredError",
                "SensitivityAtSpecificity",
                "sparse_categorical_accuracy",
                "sparse_categorical_crossentropy",
                "sparse_top_k_categorical_accuracy",
                "SpecificityAtSensitivity",
                "squared_hinge",
                "Sum",
                "top_k_categorical_accuracy",
                "TrueNegatives",
                "TruePositives",
            ]
        ]
    ] = None
    metric_emit_freq: Optional[Callable[[int], bool]] = "``````None``````"
    output_type: str = "infer"
    validation_split: float = 0.1
    batch_size: int = 128
    tpu_address: Optional[str] = None
    kwargs: Optional[dict] = None
    return_type: Any = model


class LoadDataConfig(object):
    """
    Load the data for your ML pipeline. Will be fed into `train`.

    :cvar dataset_name: name of dataset
    :cvar data_loader: function that returns the expected data type.
    :cvar data_type: incoming data type
    :cvar output_type: outgoing data_type,
    :cvar K: backend engine, e.g., `np` or `tf`
    :cvar data_loader_kwargs: pass this as arguments to data_loader function
    :cvar return_type: Dataset splits (by default, your train and test)"""

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
    ] = "``````self.data``````"


class LoadModelConfig(object):
    """
    Load the model.
    Takes a model object, or a pipeline that downloads & configures before returning a model object.

    :cvar model: model object, e.g., a tf.keras.Sequential, tl.Serial,  nn.Module instance
    :cvar call: whether to call `model()` even if `len(model_kwargs) == 0`
    :cvar model_kwargs: to be passed into the model. If empty, doesn't call, unless call=True.
    :cvar return_type: self.model, e.g., the result of applying `model_kwargs` on model"""

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
            "MobileNetV3Large",
            "MobileNetV3Small",
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
    return_type: Callable[[], tf.keras.Model] = "``````self.get_model``````"


__all__ = ["LoadDataConfig", "LoadModelConfig", "TrainConfig"]
