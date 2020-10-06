"""
CLI interface to ml-params-tensorflow. Expected to be bootstrapped by ml-params.
"""

from collections import namedtuple
from json import loads

# TODO
self = namedtuple("State", ("model",))


def train_parser(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, the model
    :rtype: ```Tuple[ArgumentParser, Any]```
    """
    argument_parser.description = "Run the training loop for your ML pipeline."
    argument_parser.add_argument(
        "--callbacks",
        type=str,
        choices=(
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
        ),
        help="Collection of callables that are run inside the training loop",
    )
    argument_parser.add_argument(
        "--epochs",
        type=int,
        help="number of epochs (must be greater than 0)",
        required=True,
    )
    argument_parser.add_argument(
        "--loss",
        type=str,
        choices=(
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
        ),
        help="Loss function, can be a string (depending on the framework) or an instance of a class",
        required=True,
    )
    argument_parser.add_argument(
        "--metrics",
        type=str,
        choices=(
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
        ),
        help="Collection of metrics to monitor, e.g., accuracy, f1",
    )
    argument_parser.add_argument(
        "--optimizer",
        type=str,
        choices=("Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop"),
        help="Optimizer, can be a string (depending on the framework) or an instance of a class",
        required=True,
    )
    argument_parser.add_argument(
        "--metric_emit_freq",
        type=int,
        help="`None` for every epoch. E.g., `eq(mod(epochs, 10), 0)` for every 10. Defaults to None",
    )
    argument_parser.add_argument(
        "--save_directory",
        type=str,
        help="Directory to save output in, e.g., weights in h5 files. If None, don't save.",
    )
    argument_parser.add_argument(
        "--output_type",
        type=str,
        help="`if save_directory is not None` then save in this format, e.g., 'h5'.",
        required=True,
        default="infer",
    )
    argument_parser.add_argument(
        "--validation_split",
        type=str,
        help="Optional float between 0 and 1, fraction of data to reserve for validation.",
        required=True,
        default=0.1,
    )
    argument_parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size at each iteration.",
        required=True,
        default=128,
    )
    argument_parser.add_argument(
        "--kwargs", type=loads, help="additional keyword arguments"
    )
    return argument_parser, self.model
