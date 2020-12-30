"""
Implementation of datasets following vendor recommendations.
"""
import os
from functools import partial
from importlib import import_module
from pkgutil import find_loader
from typing import (
    Any,
    AnyStr,
    Callable,
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
from ml_params.datasets import load_data_from_ml_prepare
from typing_extensions import Literal

datasets2classes = (
    {}
    if find_loader("ml_prepare") is None
    else getattr(import_module("ml_prepare.datasets"), "datasets2classes")
)

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

    :return: The image in the right datatype and scale, the label
    :rtype: ```Tuple[tf.float32, str]```
    """
    return tf.cast(image, dtype=tf.float32) / scale, label


def load_data_from_tfds_or_ml_prepare(
    *,
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
    tfds_dir: str = os.environ.get(
        "TFDS_DATA_DIR", os.path.join(os.path.expanduser("~"), "tensorflow_datasets")
    ),
    K: Literal["np", "tf"] = "tf",
    acquire_and_concat_validation_to_train: bool = True,
    as_numpy: bool = True,
    **data_loader_kwargs
) -> Tuple[
    Union[
        Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo],
        Tuple[
            Iterator[NumpyElem], Iterator[NumpyElem], Union[tfds.core.DatasetInfo, Any]
        ],
    ],
    Union[
        Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo],
        Tuple[
            Iterator[NumpyElem], Iterator[NumpyElem], Union[tfds.core.DatasetInfo, Any]
        ],
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

    :param acquire_and_concat_validation_to_train: Whether to acquire the validation split
      and then concatenate it to train

    :param data_loader_kwargs: pass this as arguments to data_loader function

    :return: Train and tests dataset splits.
    """
    tfds_dir = os.path.expanduser(tfds_dir)
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
        Tuple[Tuple[tf.data.Dataset, tf.data.Dataset], tfds.core.DatasetInfo],
    ] = tfds.load
    (_ds_train, _ds_test), _ds_info = tfds_load(
        dataset_name,
        split=[tfds.core.ReadInstruction("train"), tfds.core.ReadInstruction("test")],
        data_dir=tfds_dir,
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        download_and_prepare_kwargs=data_loader_kwargs["download_and_prepare_kwargs"],
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
    ds_train = (
        ds_train.map(normalize_img_at_scale, num_parallel_calls=num_parallel_calls)
        .cache()
        .shuffle(ds_info.splits["train"].num_examples)
        .batch(data_loader_kwargs["batch_size"])
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_test = (
        ds_test.map(normalize_img_at_scale, num_parallel_calls=num_parallel_calls)
        .batch(data_loader_kwargs["batch_size"])
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    splits = ds_train, ds_test
    _ds_train, _ds_test = map(tfds.as_numpy, splits) if as_numpy else splits
    ds_train: Union[
        Tuple[tf.data.Dataset, tf.data.Dataset],
        Tuple[Iterator[NumpyElem], Iterator[NumpyElem]],
    ] = _ds_train
    ds_test: Union[
        Tuple[tf.data.Dataset, tf.data.Dataset],
        Tuple[Iterator[NumpyElem], Iterator[NumpyElem]],
    ] = _ds_test
    return ds_train, ds_test, ds_info
