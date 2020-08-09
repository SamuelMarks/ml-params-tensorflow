"""
MNIST classification test(s)
"""

from os import path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Optional
from unittest import TestCase

import tensorflow as tf

from ml_params_tensorflow.example_model import get_model
from ml_params_tensorflow.ml_params_impl import TensorFlowTrainer
from ml_params_tensorflow.tests.utils_for_tests import unittest_main


class TestMnist(TestCase):
    """
    Tests classification on the MNIST dataset using this ml-params implementation
    """

    tfds_dir = None  # type: Optional[str]
    model_dir = None  # type: Optional[str]

    @classmethod
    def setUpClass(cls) -> None:
        """
        Creates the temporary directory and sets tfds_dir before running the suite
        """
        TestMnist.tfds_dir = path.join(path.expanduser("~"), "tensorflow_datasets")
        TestMnist.model_dir = mkdtemp("_model_dir")

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Removes the temporary directory after running the suite
        """
        # rmtree(TestMnist.tfds_dir)
        rmtree(TestMnist.model_dir)

    def test_mnist(self) -> None:
        """
        Tests classification roundtrip
        """
        num_classes = 10
        epochs = 3

        trainer = TensorFlowTrainer()
        trainer.load_data("mnist", tfds_dir=TestMnist.tfds_dir, num_classes=num_classes)
        trainer.load_model(get_model, num_classes=num_classes)
        trainer.train(
            epochs=epochs,
            model_dir=TestMnist.model_dir,
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=["accuracy"],
            callbacks=None,
            save_directory=None,
            metric_emit_freq=None,
        )


unittest_main()
