""" Implementation of ml_params API """

# Mostly based off https://github.com/keras-team/keras-io/blob/8320a6c/examples/vision/mnist_convnet.py

from os import path, environ
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ml_params.base import BaseTrainer
from ml_prepare.datasets import datasets2classes
from ml_prepare.exectors import build_tfds_dataset

from ml_params_tensorflow import get_logger

if environ.get('TF_KERAS', True):
    from tensorflow import keras
else:
    import keras

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


class TensorFlowTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for TensorFlow """

    data = None  # type: (None or Tuple[tf.data.Dataset, tf.data.Dataset] )
    model = None  # contains the model, e.g., a `tl.Serial`

    def __init__(self, model, **model_kwargs):
        super(TensorFlowTrainer, self).__init__()
        self.model = model(**model_kwargs)

    def load_data(self, dataset_name, data_loader=None,
                  data_loader_kwargs=None, data_type='infer',
                  output_type=None, K=None):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        self.data = super(TensorFlowTrainer, self).load_data(
            dataset_name=dataset_name,
            data_loader=data_loader or self.load_data_from_tfds_or_ml_prepare,
            data_loader_kwargs=data_loader_kwargs,
            data_type=data_type,
            output_type=output_type
        )

    @staticmethod
    def load_data_from_tfds_or_ml_prepare(dataset_name, tensorflow_datasets_dir=None, data_loader_kwargs=None):
        """
        Acquire from the official keras model zoo, or the ophthalmology focussed ml-prepare library

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param tensorflow_datasets_dir: directory to look for models in. Default is ~/tensorflow_datasets.
        :type tensorflow_datasets_dir: ```None or str```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :return: Train and tests dataset splits
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        data_loader_kwargs.update({
            'dataset_name': dataset_name,
            'tfds_dir': tensorflow_datasets_dir,

        })
        if 'scale' not in data_loader_kwargs:
            data_loader_kwargs['scale'] = 255

        if dataset_name in datasets2classes:
            ds_builder = build_tfds_dataset(**data_loader_kwargs)

            if hasattr(ds_builder, 'download_and_prepare_kwargs'):
                download_and_prepare_kwargs = getattr(ds_builder, 'download_and_prepare_kwargs')
                delattr(ds_builder, 'download_and_prepare_kwargs')
            else:
                download_and_prepare_kwargs = None

            return BaseTrainer.common_dataset_handler(
                ds_builder=ds_builder,
                download_and_prepare_kwargs=download_and_prepare_kwargs,
                scale=None, K=None, as_numpy=False
            )
        else:
            (ds_train, ds_test), ds_info = tfds.load(
                'mnist',
                split=['train', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
            )

            if 'batch_size' not in data_loader_kwargs:
                data_loader_kwargs['batch_size'] = 128

            def normalize_img(image, label):
                """Normalizes images: `uint8` -> `float32`."""
                return tf.cast(image, tf.float32) / data_loader_kwargs['scale'], label

            num_parallel_calls = tf.data.experimental.AUTOTUNE if 'tf' in globals() else 10

            ds_train = ds_train.map(
                normalize_img, num_parallel_calls=num_parallel_calls)
            ds_train = ds_train.cache()
            ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
            ds_train = ds_train.batch(data_loader_kwargs['batch_size'])
            ds_train = ds_train.prefetch(num_parallel_calls)

            ds_test = ds_test.map(
                normalize_img, num_parallel_calls=num_parallel_calls)
            ds_test = ds_test.batch(data_loader_kwargs['batch_size'])
            ds_test = ds_test.cache()
            ds_test = ds_test.prefetch(num_parallel_calls)

            return ds_train, ds_test

    def train(self, epochs, validation_split=0.1, batch_size=128, *args, **kwargs):
        super(TensorFlowTrainer, self).train(epochs=epochs, *args, **kwargs)
        assert self.data is not None
        assert self.model is not None

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=keras.optimizers.Adam(0.001),
            metrics=['accuracy'],
        )
        self.model.fit(
            self.data[0],
            epochs=epochs,
            validation_data=self.data[1],
        )

        return self.model


del Tuple, build_tfds_dataset, get_logger

__all__ = ['TensorFlowTrainer']
