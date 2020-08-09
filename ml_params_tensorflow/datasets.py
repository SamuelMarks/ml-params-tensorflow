"""
Implementation of datasets following vendor recommendations.
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from ml_params.datasets import load_data_from_ml_prepare
from ml_prepare.datasets import datasets2classes


def load_data_from_tfds_or_ml_prepare(
    dataset_name,
    tfds_dir="~/tensorflow_datasets",
    K="tf",
    as_numpy=True,
    **data_loader_kwargs
):
    """
    Acquire from the official tfds model zoo, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param tfds_dir: directory to look for models in.
    :type tfds_dir: ```str```

    :param K: backend engine, e.g., `np` or `tf`
    :type K: ```Literal['np', 'tf']```

    :param as_numpy: Convert to numpy ndarrays
    :type as_numpy: ```bool```

    :param data_loader_kwargs: pass this as arguments to data_loader function
    :type data_loader_kwargs: ```**data_loader_kwargs```

    :return: Train and tests dataset splits
    :rtype: ```Union[Tuple[tf.data.Dataset, tf.data.Dataset], Tuple[np.ndarray, np.ndarray]]```
    """
    if dataset_name in datasets2classes:
        return load_data_from_ml_prepare(
            dataset_name=dataset_name, tfds_dir=tfds_dir, **data_loader_kwargs
        )

    data_loader_kwargs.update(
        {"dataset_name": dataset_name, "tfds_dir": tfds_dir,}
    )
    if "scale" not in data_loader_kwargs:
        data_loader_kwargs["scale"] = 255

    (ds_train, ds_test), ds_info = tfds.load(
        dataset_name,
        split=[tfds.core.ReadInstruction("train"), tfds.core.ReadInstruction("test"),],
        data_dir=tfds_dir,
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    if "batch_size" not in data_loader_kwargs:
        data_loader_kwargs["batch_size"] = 128

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / data_loader_kwargs["scale"], label

    num_parallel_calls = tf.data.experimental.AUTOTUNE if "tf" in globals() else 10

    ds_train = ds_train.map(normalize_img, num_parallel_calls=num_parallel_calls)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(data_loader_kwargs["batch_size"])
    ds_train = ds_train.prefetch(num_parallel_calls)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=num_parallel_calls)
    ds_test = ds_test.batch(data_loader_kwargs["batch_size"])
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(num_parallel_calls)

    return ds_train, ds_test
