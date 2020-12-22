"""
CLI interface to ml-params-tensorflow. Expected to be bootstrapped by ml-params.
"""

from dataclasses import dataclass
from itertools import chain
from json import loads

try:
    from ml_prepare.datasets import datasets2classes
except ImportError:
    datasets2classes = {}


@dataclass
class self(object):
    """
    Simple class to proxy object expected by code generated from `train` function
    :cvar model: The model (probably a `tf.keras.models.Sequential`)
    :cvar data: The data (probably a `tf.data.Dataset`)
    """

    model: object = None
    data: object = None


model = self.model


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
        "--epochs",
        type=int,
        help="number of epochs (must be greater than 0)",
        required=True,
    )
    argument_parser.add_argument(
        "--loss",
        type=str,
        choices=(
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
        ),
        help="Loss function, can be a string (depending on the framework) or an instance of a class",
        required=True,
    )
    argument_parser.add_argument(
        "--optimizer",
        type=str,
        choices=("Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop"),
        help="Optimizer, can be a string (depending on the framework) or an instance of a class",
        required=True,
    )
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
        action="append",
        help="Collection of callables that are run inside the training loop",
    )
    argument_parser.add_argument(
        "--metrics",
        type=str,
        choices=(
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
        ),
        action="append",
        help="Collection of metrics to monitor, e.g., accuracy, f1",
    )
    argument_parser.add_argument(
        "--metric_emit_freq",
        type=int,
        help="`None` for every epoch. E.g., `eq(mod(epochs, 10), 0)` for every 10.",
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
        type=float,
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
        "--tpu_address",
        type=str,
        help="Address of TPU cluster. If None, don't connect & run within TPU context.",
    )
    argument_parser.add_argument(
        "--kwargs", type=loads, help="additional keyword arguments"
    )
    return argument_parser, model


def load_data_parser(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Dataset splits (by default, your train and test)
    :rtype: ```Tuple[ArgumentParser, Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]]```
    """
    argument_parser.description = (
        "Load the data for your ML pipeline. Will be fed into `train`."
    )
    argument_parser.add_argument(
        "--dataset_name",
        type=str,
        choices=tuple(
            sorted(
                chain.from_iterable(
                    (
                        (
                            "boston_housing",
                            "cifar10",
                            "cifar100",
                            "fashion_mnist",
                            "imdb",
                            "mnist",
                            "reuters",
                        ),
                        datasets2classes.keys(),
                    )
                )
            )
        ),
        help="name of dataset",
        required=True,
    )
    argument_parser.add_argument(
        "--data_loader",
        type=str,
        choices=("np", "tf"),
        help="function that returns the expected data type.",
    )
    argument_parser.add_argument(
        "--data_type",
        type=str,
        help="incoming data type",
        required=True,
        default="infer",
    )
    argument_parser.add_argument("--output_type", type=str, help="outgoing data_type,")
    argument_parser.add_argument(
        "--K", type=str, choices=("np", "tf"), help="backend engine, e.g., `np` or `tf`"
    )
    argument_parser.add_argument(
        "--data_loader_kwargs",
        type=loads,
        help="pass this as arguments to data_loader function",
    )
    return argument_parser, "```self.data```"


def load_model_parser(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, self.model, e.g., the result of applying `model_kwargs` on model
    :rtype: ```Tuple[ArgumentParser, Callable[[], tf.keras.Model]]```
    """
    argument_parser.description = """Load the model.
Takes a model object, or a pipeline that downloads & configures before returning a model object."""
    argument_parser.add_argument(
        "--model",
        type=str,
        choices=(
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
        ),
        help="model object, e.g., a tf.keras.Sequential, tl.Serial,  nn.Module instance",
        required=True,
    )
    argument_parser.add_argument(
        "--call",
        type=bool,
        help="whether to call `model()` even if `len(model_kwargs) == 0`",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--model_kwargs",
        type=loads,
        help="to be passed into the model. If empty, doesn't call, unless call=True.",
    )
    return argument_parser, "```self.get_model```"


__all__ = ["load_data_parser", "load_model_parser", "train_parser"]
