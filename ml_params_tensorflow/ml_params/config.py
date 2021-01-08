"""
Config interface to ml-params-tensorflow. Expected to be bootstrapped by ml-params, as well as internally.
"""

from typing import Any, AnyStr, Callable, Dict, Optional, Tuple, Union, Iterator, List

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing_extensions import Literal


class ConfigClass(object):
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
    loss: Any = None
    optimizer: Any = None
    callbacks: Any = None
    metrics: Any = None
    metric_emit_freq: Optional[Callable[[int], bool]] = None
    output_type: str = "infer"
    validation_split: Optional[float] = 0.1
    batch_size: int = 128
    tpu_address: Optional[str] = None
    kwargs: Optional[dict] = None
    return_type: str = "model"


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
            [AnyStr, Literal["np", "tf"], bool, Dict],
            Tuple[
                Union[
                    Tuple[tf.data.Dataset, tf.data.Dataset],
                    Tuple[
                        Iterator[
                            Union[
                                tf.RaggedTensor,
                                np.ndarray,
                                np.generic,
                                bytes,
                                Iterable[
                                    Union[
                                        tf.RaggedTensor, np.ndarray, np.generic, bytes
                                    ]
                                ],
                            ]
                        ],
                        Iterator[
                            Union[
                                tf.RaggedTensor,
                                np.ndarray,
                                np.generic,
                                bytes,
                                Iterable[
                                    Union[
                                        tf.RaggedTensor, np.ndarray, np.generic, bytes
                                    ]
                                ],
                            ]
                        ],
                    ],
                ],
                Union[
                    Tuple[tf.data.Dataset, tf.data.Dataset],
                    Tuple[
                        Iterator[
                            Union[
                                tf.RaggedTensor,
                                np.ndarray,
                                np.generic,
                                bytes,
                                Iterable[
                                    Union[
                                        tf.RaggedTensor, np.ndarray, np.generic, bytes
                                    ]
                                ],
                            ]
                        ],
                        Iterator[
                            Union[
                                tf.RaggedTensor,
                                np.ndarray,
                                np.generic,
                                bytes,
                                Iterable[
                                    Union[
                                        tf.RaggedTensor, np.ndarray, np.generic, bytes
                                    ]
                                ],
                            ]
                        ],
                    ],
                ],
                tfds.core.DatasetInfo,
            ],
        ]
    ] = None
    data_type: str = "infer"
    output_type: Optional[Literal["np"]] = "no conversion"
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
