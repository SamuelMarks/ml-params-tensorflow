from os import path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase, main as unittest_main

from tensorflow import keras

from ml_params_tensorflow.example_model import get_model
from ml_params_tensorflow.ml_params_impl import TensorFlowTrainer


class TestMnist(TestCase):
    tfds_dir = None  # type: str or None
    model_dir = None  # type: str or None

    @classmethod
    def setUpClass(cls) -> None:
        TestMnist.tfds_dir = path.join(path.expanduser('~'), 'tensorflow_datasets')
        TestMnist.model_dir = mkdtemp('_model_dir')

    @classmethod
    def tearDownClass(cls) -> None:
        # rmtree(TestMnist.tfds_dir)
        rmtree(TestMnist.model_dir)

    def test_mnist(self) -> None:
        num_classes = 10
        epochs = 3

        trainer = TensorFlowTrainer()
        trainer.load_data(
            'mnist',
            tfds_dir=TestMnist.tfds_dir,
            num_classes=num_classes
        )
        trainer.load_model(get_model, num_classes=num_classes)
        trainer.train(epochs=epochs,
                      model_dir=TestMnist.model_dir,
                      loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(0.001),
                      metrics=['accuracy'],
                      callbacks=None,
                      save_directory=None,
                      metric_emit_freq=None)


if __name__ == '__main__':
    unittest_main()
