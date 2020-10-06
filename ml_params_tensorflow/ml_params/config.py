from typing import Any, Literal, Optional, List, Callable


class ConfigClass(object):
    """
    Run the training loop for your ML pipeline.

    :cvar callbacks: Collection of callables that are run inside the training loop
    :cvar epochs: number of epochs (must be greater than 0)
    :cvar loss: Loss function, can be a string (depending on the framework) or an instance of a class
    :cvar metrics: Collection of metrics to monitor, e.g., accuracy, f1
    :cvar optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
    :cvar metric_emit_freq: Frequency of metric emission, e.g., `lambda: epochs % 10 == 0`, None means every epoch. Defaults to None
    :cvar save_directory: Directory to save output in, e.g., weights in h5 files. If None, don't save. Defaults to None
    :cvar output_type: `if save_directory is not None` then save in this format, e.g., 'h5'. Defaults to infer
    :cvar validation_split: Optional float between 0 and 1, fraction of data to reserve for validation. Defaults to 0.1
    :cvar batch_size: batch size at each iteration. Defaults to 128
    :cvar kwargs: additional keyword arguments
    :cvar return_type: the model. Defaults to self.model"""

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
    optimizer: Literal[
        "Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop"
    ] = None
    metric_emit_freq: Optional[Callable[[int], bool]] = None
    save_directory: Optional[str] = None
    output_type: str = "infer"
    validation_split: float = 0.1
    batch_size: int = 128
    kwargs: dict = {}
    return_type: Any = self.model
