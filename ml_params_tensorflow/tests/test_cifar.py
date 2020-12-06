"""
CIFAR classification test(s)
"""

from os import path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Optional
from unittest import TestCase

import tensorflow as tf
from ml_params_tensorflow.ml_params.trainer import TensorFlowTrainer
from ml_params_tensorflow.tests.utils_for_tests import unittest_main


class TestCifar(TestCase):
    """
    Tests classification on the CIFAR dataset using this ml-params implementation
    """

    tfds_dir = None  # type: Optional[str]
    model_dir = None  # type: Optional[str]

    @classmethod
    def setUpClass(cls) -> None:
        """
        Creates the temporary directory and sets tfds_dir before running the suite
        """
        TestCifar.tfds_dir = path.join(path.expanduser("~"), "tensorflow_datasets")
        TestCifar.model_dir = mkdtemp("_model_dir")

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Removes the temporary directory after running the suite
        """
        # rmtree(TestCifar.tfds_dir)
        rmtree(TestCifar.model_dir)

    def test_cifar_with_transfer_learning(self) -> None:
        """
        Tests classification roundtrip, using a builtin transfer-learning model
        """
        epochs = 3

        trainer = TensorFlowTrainer()
        trainer.load_data(dataset_name="cifar10", tfds_dir=TestCifar.tfds_dir)
        trainer.load_model(model="MobileNet")
        self.assertIsInstance(
            trainer.train(
                epochs=epochs,
                model_dir=TestCifar.model_dir,
                loss="SparseCategoricalCrossentropy",
                optimizer="Adam",
                metrics=["categorical_accuracy"],
                callbacks=None,
                save_directory=None,
                metric_emit_freq=None,
            ),
            tf.keras.Sequential,
        )


unittest_main()
