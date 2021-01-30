"""
Implementation of ml_params BaseTrainer API
"""
from functools import partial
from os import path
from types import FunctionType
from typing import (
    AnyStr,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ml_params.base import BaseTrainer
from typing_extensions import Literal

from ml_params_tensorflow import get_logger
from ml_params_tensorflow.ml_params.datasets import load_data_from_tfds_or_ml_prepare
from ml_params_tensorflow.ml_params.type_generators import exposed_datasets

logger = get_logger(
    ".".join(
        (
            path.basename(path.dirname(__file__)),
            path.basename(__file__).rpartition(".")[0],
        )
    )
)


class TensorFlowTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for TensorFlow """

    data = None
    ds_info: Optional[tfds.core.DatasetInfo] = None
    get_model: Optional[Callable[[], tf.keras.Model]] = None

    def load_data(
        self,
        *,
        dataset_name: Literal[
            "boston_housing",
            "cifar10",
            "cifar100",
            "fashion_mnist",
            "imdb",
            "mnist",
            "reuters",
        ],
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
                                            tf.RaggedTensor,
                                            np.ndarray,
                                            np.generic,
                                            bytes,
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
                                            tf.RaggedTensor,
                                            np.ndarray,
                                            np.generic,
                                            bytes,
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
                                            tf.RaggedTensor,
                                            np.ndarray,
                                            np.generic,
                                            bytes,
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
                                            tf.RaggedTensor,
                                            np.ndarray,
                                            np.generic,
                                            bytes,
                                        ]
                                    ],
                                ]
                            ],
                        ],
                    ],
                    tfds.core.DatasetInfo,
                ],
            ]
        ] = None,
        data_type: str = "infer",
        output_type: Optional[Literal["np"]] = None,
        K: Optional[Literal["np", "tf"]] = None,
        **data_loader_kwargs
    ) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]:
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param *: syntactic note indicating everything after is a keyword-only argument

        :param dataset_name: name of dataset

        :param data_loader: function that returns the expected data type.

        :param data_type: incoming data type

        :param output_type: outgoing data_type, defaults to no conversion

        :param K: backend engine, e.g., `np` or `tf`

        :param data_loader_kwargs: pass this as arguments to data_loader function

        :return: Dataset splits (by default, your train and test)
        """
        self.data = super(TensorFlowTrainer, self).load_data(
            dataset_name=dataset_name,
            data_loader=data_loader or load_data_from_tfds_or_ml_prepare,
            data_type=data_type,
            output_type=output_type,
            as_numpy=output_type == "np",
            K=K,
            **data_loader_kwargs
        )
        if len(self.data) > 2:
            self.ds_info: tfds.core.DatasetInfo = self.data[2]
        return self.data

    def load_model(
        self,
        *,
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
        ],
        call: bool = False,
        **model_kwargs
    ) -> Callable[[], tf.keras.Model]:
        """
        Load the model.
        Takes a model object, or a pipeline that downloads & configures before returning a model object.

        :param *: syntactic note indicating everything after is a keyword-only argument

        :param model: model object, e.g., a tf.keras.Sequential, tl.Serial,  nn.Module instance

        :param call: whether to call `model()` even if `len(model_kwargs) == 0`

        :param **model_kwargs: to be passed into the model. If empty, doesn't call, unless call=True.

        :return: self.model, e.g., the result of applying `model_kwargs` on model
        """
        assert (
            self.data or self.ds_info
        ), "Run `load_data` before `load_model` so that `ds_info` is available"

        def get_model():
            """
            Call this to get the model.
            Distributed strategies need models to be constructed within its scope,
            so that's why this function

            :return: model, e.g., the result of applying `model_kwargs` on model
            :rtype: ```Any```
            """
            if get_model.run_before is True:
                return
            else:
                get_model.run_before = True
            super(TensorFlowTrainer, self).load_model(
                model=model, call=callable(model) or call, **model_kwargs
            )
            assert self.get_model is not None
            self.get_model()
            assert self.model is not None
            if not isinstance(self.model, (tf.keras.Model, FunctionType)):
                if isinstance(self.model, str):
                    if self.model.startswith(
                        "tf.keras.applications."
                    ) or self.model in dir(tf.keras.applications):
                        self.model = getattr(
                            tf.keras.applications, self.model.rpartition(".")[2]
                        )
                    elif self.model in exposed_datasets:
                        self.model = exposed_datasets[self.model]
                    else:
                        raise NotImplementedError(
                            "`tf.keras.Model` from {!r}".format(self.model)
                        )
                    extra_model_kwargs = (
                        next(
                            (
                                {"input_shape": v.shape}
                                for k, v in self.ds_info.features.items()
                                if hasattr(v, "shape") and v.shape
                            ),
                            {},
                        )
                        if self.ds_info is not None and self.ds_info.features
                        else {}
                    )
                    if (
                        extra_model_kwargs.get("input_shape", (None,) * 3)[:-1]
                        == (None,) * 2
                    ):
                        if self.ds_info.features["image"].shape[:-1] != (None,) * 2:
                            extra_model_kwargs["input_shape"] = self.ds_info.features[
                                "image"
                            ].shape
                        else:
                            for image, label in self.data[0].take(1):
                                extra_model_kwargs["input_shape"] = tuple(
                                    tf.shape(image).numpy()[1:]
                                )
                    self.model = self.model(
                        include_top=model_kwargs.get("include_top", False),
                        **extra_model_kwargs,
                        **{k: v for k, v in model_kwargs.items() if k != "include_top"}
                    )
                    self.model.trainable = False
                assert isinstance(
                    self.model, tf.keras.Model
                ), "Expected `tf.keras.Model` got {!r}".format(type(self.model))
                assert self.ds_info is not None
                self.model = tf.keras.Sequential(
                    [
                        self.model,
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(
                            *(
                                (1, "sigmoid")
                                if self.ds_info.features["label"].num_classes
                                in frozenset((1, 2))
                                else (self.ds_info.features["label"].num_classes,)
                            )
                        ),
                    ]
                )
            return self.model

        get_model.run_before = False
        self.get_model = get_model
        return self.get_model

    def train(
        self,
        *,
        epochs: int,
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
        ],
        optimizer: Literal[
            "Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop"
        ],
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
        ] = None,
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
        ] = None,
        metric_emit_freq: Optional[Callable[[int], bool]] = None,
        output_type: str = "infer",
        validation_split: float = 0.1,
        batch_size: int = 128,
        tpu_address: Optional[str] = None,
        **kwargs
    ):
        """
        Run the training loop for your ML pipeline.

        :param *: syntactic note indicating everything after is a keyword-only argument

        :param epochs: number of epochs (must be greater than 0)

        :param loss: Loss function, can be a string (depending on the framework) or an instance of a class

        :param optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class

        :param callbacks: Collection of callables that are run inside the training loop

        :param metrics: Collection of metrics to monitor, e.g., accuracy, f1

        :param metric_emit_freq: `None` for every epoch. E.g., `eq(mod(epochs, 10), 0)` for every 10. Defaults to None

        :param output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.

        :param validation_split: Optional float between 0 and 1, fraction of data to reserve for validation.

        :param batch_size: batch size at each iteration.

        :param tpu_address: Address of TPU cluster. If None, don't connect & run within TPU context.

        :param kwargs: additional keyword arguments

        :return: the model
        :rtype: ```Any```
        """
        from ml_params_tensorflow.ml_params.type_generators import (
            exposed_callbacks,
            exposed_losses,
            exposed_metrics,
            exposed_optimizers,
        )

        super(TensorFlowTrainer, self).train(epochs=epochs)
        assert self.data is not None
        assert self.get_model is not None
        if tpu_address is None:
            strategy = tf.distribute.MirroredStrategy()
        else:
            print("Connecting to:", tpu_address, ";")
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu="grpc://{tpu_address}".format(tpu_address=tpu_address)
            )
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            print("TPU devices:", tf.config.list_logical_devices("TPU"), ";")
            strategy = tf.distribute.TPUStrategy(resolver)
        with strategy.scope():
            model = self.get_model()
            optimizer = acquire_symbols_from(
                name=optimizer, name2sym=exposed_optimizers, never_str=False
            )
            metrics = (
                list(
                    map(
                        partial(acquire_symbols_from, name2sym=exposed_metrics), metrics
                    )
                )
                if metrics
                else None
            )
            if isinstance(loss, tuple):
                loss_kwargs = {
                    k: v
                    for k, v in vars(loss[1]).items()
                    if k not in frozenset(("y_pred", "y_true"))
                }
                loss = acquire_symbols_from(
                    name2sym=exposed_losses,
                    name="".join(map(str.title, loss[0].rpartition(".")[2].split("_"))),
                    never_str=True,
                )(**loss_kwargs)
            print("optimizer:", optimizer, ";")
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        callbacks = (
            list(
                map(
                    partial(acquire_symbols_from, name2sym=exposed_callbacks), callbacks
                )
            )
            if callbacks
            else None
        )
        print("callbacks:", callbacks, ";")
        model.fit(
            self.data[0],
            validation_data=self.data[1],
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split
            if tf.is_tensor(self.data[1])
            or type(self.data[1]).__module__ == np.__name__
            else None,
            batch_size=batch_size,
        )
        return model


def acquire_symbols_from(name, name2sym, never_str=False):
    """
    Acquire the symbol(s) from the iterable

    :param name: Name of symbol. All namespace is removed.
    :type name: ```Union[Any, str]```

    :param name2sym: Dict from symbol name to symbol
    :type name2sym: ```Dict[Str, Any]```

    :param never_str: If True, ensure that `getattr` on the module is always called
    :type never_str: ```bool```

    :return: The list of symbols acquired from the module
    :rtype: ```Callable[[...], Any]```
    """
    if isinstance(name, str):
        name = name.rpartition(".")[2] if name.count(".") > 0 else name
        if name in name2sym and never_str:
            return name2sym[name]
    elif isinstance(name, tuple) and len(name) == 2:
        name, namespace = name
        name = name.rpartition(".")[2] if name.count(".") > 0 else name
        return name2sym[name](**vars(namespace))
    if never_str and isinstance(name, str):
        raise KeyError("{!r} not found in {!r}".format(name, ""))
    return name


del get_logger
__all__ = ["TensorFlowTrainer"]
