"""
Implementation of ml_params BaseTrainer API
"""
from collections import namedtuple
from functools import partial
from itertools import filterfalse
from operator import eq
from os import path
from types import FunctionType
from typing import Any, AnyStr, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ml_params.base import BaseTrainer
from typing_extensions import Literal

from ml_params_tensorflow import get_logger
from ml_params_tensorflow.ml_params.datasets import load_data_from_tfds_or_ml_prepare

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
                [AnyStr, AnyStr, Literal["np", "tf"], bool, Dict],
                Union[
                    Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo],
                    Tuple[np.ndarray, np.ndarray, Any],
                    Tuple[Any, Any, Any],
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
                    self.model = self.model(
                        include_top=model_kwargs.get("include_top", False),
                        **extra_model_kwargs,
                        **{k: v for k, v in model_kwargs.items() if k != "include_top"}
                    )
                    self.model.trainable = False
                assert isinstance(
                    self.model, tf.keras.Model
                ), "Expected `tf.keras.Model` got {!r}".format(type(self.model))
                # elif isinstance(self.model, tf.keras.Layer):
                assert self.ds_info is not None
                self.model = tf.keras.Sequential(
                    [
                        self.model,
                        tf.keras.layers.Dense(
                            self.ds_info.features["label"].num_classes
                        ),
                    ]
                )
            return self.model

        self.get_model = get_model
        return self.get_model

    def train(
        self,
        *,
        epochs: int,
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
        super(TensorFlowTrainer, self).train(epochs=epochs)

        assert self.data is not None
        assert self.get_model is not None

        if tpu_address is None:
            strategy = namedtuple("FreeScope", ("scope",))(
                lambda: memoryview(b"")  # nop
            )
        else:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu="grpc://{tpu_address}".format(tpu_address=tpu_address)
            )
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)

            print("TPU devices:", tf.config.list_logical_devices("TPU"), ";")

            strategy = tf.distribute.TPUStrategy(resolver)

        with strategy.scope():
            model = self.get_model()
            model.compile(
                loss=loss,
                optimizer=set_from((optimizer,), tf.keras.optimizers)[0](),
                metrics=set_from(metrics, tf.keras.metrics),
            )

        callbacks = set_from(callbacks, tf.keras.callbacks) if callbacks else None
        model.fit(
            self.data[0],
            validation_data=self.data[1],
            epochs=epochs,
            # loss=loss,
            # optimizer=optimizer,
            callbacks=callbacks,
            # metrics=metrics,
            validation_split=validation_split
            if tf.is_tensor(self.data[1])
            or type(self.data[1]).__module__ == np.__name__
            else None,
            batch_size=batch_size,
        )

        return model


def set_from(iterable, o):
    """
    Helper function for generating
    """
    return (
        None
        if iterable is None
        else list(
            getattr(o, obj.rpartition(".")[2])
            if isinstance(obj, str)
            else getattr(o, obj.__class__.__name__)(
                **dict(filterfalse(partial(eq, ("kwargs", None)), vars(obj).items()))
            )
            for obj in iterable
        )
    )


del get_logger

__all__ = ["TensorFlowTrainer"]
