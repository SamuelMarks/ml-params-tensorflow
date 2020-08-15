"""
Implementation of datasets following vendor recommendations.
"""
import os
from functools import partial
from typing import (
    Union,
    Tuple,
    Literal,
    AnyStr,
    Callable,
    Optional,
    List,
    Any,
    Iterator,
    Iterable,
)

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ml_params.datasets import load_data_from_ml_prepare
from ml_prepare.datasets import datasets2classes

# Two lines from https://github.com/tensorflow/datasets/blob/d2b7dd0/tensorflow_datasets/core/dataset_utils.py#L33-L34

NumpyValue = Union[tf.RaggedTensor, np.ndarray, np.generic, bytes]
NumpyElem = Union[NumpyValue, Iterable[NumpyValue]]


def normalize_img(image: Any, label: str, scale: Union[float, int]) -> Tuple[Any, str]:
    """
    Normalizes images: `uint8` -> `float32`.

    :param image: The image (well, a matrix anyway)
    :type image: ```Union[tf.uint8, tf.uint16, tf.float16, tf.float32]```

    :param label: The label (not used, just passed through as last element of returned tuple)
    :type label: ```str```

    :param scale: The scale (i.e., the denominator)
    :type scale: ```Union[float, str]```

    :returns: The image in the right datatype and scale, the label
    :rtype: ```Tuple[tf.float32, str]```
    """
    return tf.cast(image, dtype=tf.float32) / scale, label


def load_data_from_tfds_or_ml_prepare(
    dataset_name: Union[
        Literal[
            "boston_housing",
            "cifar10",
            "cifar100",
            "fashion_mnist",
            "imdb",
            "mnist",
            "reuters",
        ],
        AnyStr,
    ],
    tfds_dir: str = "~/tensorflow_datasets",
    K: Literal["np", "tf"] = "tf",
    as_numpy: bool = True,
    **data_loader_kwargs
) -> Tuple[
    Union[
        Tuple[tf.data.Dataset, tf.data.Dataset],
        Tuple[Iterator[NumpyElem], Iterator[NumpyElem]],
    ],
    Union[
        Tuple[tf.data.Dataset, tf.data.Dataset],
        Tuple[Iterator[NumpyElem], Iterator[NumpyElem]],
    ],
]:
    """
    Acquire from the official tfds model zoo, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset

    :param tfds_dir: directory to look for models in.

    :param K: backend engine, e.g., `np` or `tf`

    :param as_numpy: Convert to numpy ndarrays. If `True`, then a structure
      matching `dataset` where `tf.data.Dataset`s are converted to generators
      of NumPy arrays and `tf.Tensor`s are converted to NumPy arrays.

    :param data_loader_kwargs: pass this as arguments to data_loader function
    :type data_loader_kwargs: ```**data_loader_kwargs```

    :return: Train and tests dataset splits.
    """
    if dataset_name in datasets2classes:
        return load_data_from_ml_prepare(
            dataset_name=dataset_name, tfds_dir=tfds_dir, **data_loader_kwargs
        )

    data_loader_kwargs.update({"dataset_name": dataset_name, "tfds_dir": tfds_dir})
    data_loader_kwargs.setdefault("scale", 255)
    data_loader_kwargs.setdefault("batch_size", 128)
    data_loader_kwargs.setdefault(
        "download_and_prepare_kwargs",
        {
            "download_dir": os.path.join(tfds_dir, "downloads"),
            "download_config": tfds.download.DownloadConfig(
                extract_dir=os.path.join(tfds_dir, "extracted"),
                manual_dir=os.path.join(tfds_dir, "downloads", "manual"),
                download_mode=tfds.GenerateMode.REUSE_DATASET_IF_EXISTS,
                compute_stats=tfds.download.ComputeStatsMode.AUTO,
                max_examples_per_split=None,
                register_checksums=False,
                force_checksums_validation=False,
                beam_runner=None,
                beam_options=None,
                try_download_gcs=True,
            ),
        },
    )

    tfds_load: Callable[
        [
            Tuple[
                str,
                Optional[List[str]],
                Optional[str],
                Optional[int],
                Optional[bool],
                Optional[bool],
                Optional[bool],
                Optional[dict],
                Optional[tfds.ReadConfig],
                Optional[bool],
                Optional[dict],
                Optional[dict],
                Optional[bool],
            ]
        ],
        Tuple[Tuple[tf.data.Dataset, tf.data.Dataset], tfds.core.DatasetInfo,],
    ] = tfds.load
    (_ds_train, _ds_test), _ds_info = tfds_load(
        dataset_name,
        split=[tfds.core.ReadInstruction("train"), tfds.core.ReadInstruction("test")],
        data_dir=tfds_dir,
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        download_and_prepare_kwargs=data_loader_kwargs["download_and_prepare_kwargs"]
        # builder_kwargs=**data_loader_kwargs
    )
    ds_train: tf.data.Dataset = _ds_train
    ds_test: tf.data.Dataset = _ds_test
    ds_info: tfds.core.DatasetInfo = _ds_info

    assert (
        "num_classes" not in data_loader_kwargs
        or data_loader_kwargs["num_classes"] == ds_info.features["label"].num_classes
    ), "Expected {!r} got {!r}".format(
        data_loader_kwargs["num_classes"], ds_info.features["label"].num_classes
    )

    num_parallel_calls = tf.data.experimental.AUTOTUNE if "tf" in globals() else 10
    normalize_img_at_scale = partial(normalize_img, scale=data_loader_kwargs["scale"])

    ds_train = ds_train.map(
        normalize_img_at_scale, num_parallel_calls=num_parallel_calls
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(data_loader_kwargs["batch_size"])
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(normalize_img_at_scale, num_parallel_calls=num_parallel_calls)
    ds_test = ds_test.batch(data_loader_kwargs["batch_size"])
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    splits = ds_train, ds_test
    if as_numpy:

        _ds_train, _ds_test = tuple(
            map(tfds.as_numpy, splits)
        )  # type: Tuple[Iterator[NumpyElem], Iterator[NumpyElem]]
    else:
        _ds_train, _ds_test = tuple(
            splits
        )  # type: Tuple[tf.data.Dataset, tf.data.Dataset]

    ds_train: Union[
        Tuple[tf.data.Dataset, tf.data.Dataset],
        Tuple[Iterator[NumpyElem], Iterator[NumpyElem]],
    ] = _ds_train
    ds_test: Union[
        Tuple[tf.data.Dataset, tf.data.Dataset],
        Tuple[Iterator[NumpyElem], Iterator[NumpyElem]],
    ] = _ds_test

    return ds_train, ds_test
