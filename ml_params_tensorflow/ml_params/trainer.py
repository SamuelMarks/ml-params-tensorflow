"""
Implementation of ml_params BaseTrainer API
"""
from os import path
from typing import Tuple, Optional, List, Callable, Union, Any, Dict, AnyStr

import numpy as np
import tensorflow as tf
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
    model = None

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
                    Tuple[tf.data.Dataset, tf.data.Dataset],
                    Tuple[np.ndarray, np.ndarray],
                    Tuple[Any, Any],
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

        :param data_loader_kwargs: pass this as arguments to data_loader function

        :param data_type: incoming data type

        :param output_type: outgoing data_type, defaults to no conversion

        :param K: backend engine, e.g., `np` or `tf`

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
        return self.data

    def train(
        self,
        *,
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
        ],
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
        ],
        optimizer: Literal[
            "Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop"
        ],
        metric_emit_freq: Optional[Callable[[int], bool]] = None,
        save_directory: Optional[str] = None,
        output_type: str = "infer",
        validation_split: float = 0.1,
        batch_size: int = 128,
        **kwargs
    ):
        """
        Run the training loop for your ML pipeline.

        :param *: syntactic note indicating everything after is a keyword-only argument

        :param callbacks: Collection of callables that are run inside the training loop

        :param epochs: number of epochs (must be greater than 0)

        :param loss: Loss function, can be a string (depending on the framework) or an instance of a class

        :param metrics: Collection of metrics to monitor, e.g., accuracy, f1

        :param optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class

        :param metric_emit_freq: Frequency of metric emission, e.g., `lambda: epochs % 10 == 0`, None means every epoch

        :param save_directory: Directory to save output in, e.g., weights in h5 files. If None, don't save.

        :param output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.

        :param validation_split: Optional float between 0 and 1, fraction of data to reserve for validation.

        :param batch_size: batch size at each iteration.

        :param kwargs: additional keyword arguments

        :return: the model
        :rtype: ```Any```
        """
        super(TensorFlowTrainer, self).train(epochs=epochs)
        assert self.data is not None
        assert self.model is not None
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.model.fit(self.data[0], epochs=epochs, validation_data=self.data[1])
        return self.model


del get_logger
__all__ = ["TensorFlowTrainer"]
