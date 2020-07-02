from os import path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase, main as unittest_main

from ml_params_tensorflow.example_model import get_model
from ml_params_tensorflow.ml_params_impl import TensorFlowTrainer


class TestMnist(TestCase):
    tensorflow_datasets_dir = None  # type: str or None
    model_dir = None  # type: str or None

    @classmethod
    def setUpClass(cls) -> None:
        TestMnist.tensorflow_datasets_dir = path.join(path.expanduser('~'), 'tensorflow_datasets')
        TestMnist.model_dir = mkdtemp('_model_dir')

    @classmethod
    def tearDownClass(cls) -> None:
        # rmtree(TestMnist.tensorflow_datasets_dir)
        rmtree(TestMnist.model_dir)

    def test_mnist(self) -> None:
        num_classes = 10
        trainer = TensorFlowTrainer(get_model, num_classes=num_classes)
        trainer.load_data('mnist', data_loader_kwargs={
            'tensorflow_datasets_dir': TestMnist.tensorflow_datasets_dir,
            'data_loader_kwargs': {'num_classes': num_classes}
        })

        epochs = 3
        trainer.train(epochs=epochs, model_dir=TestMnist.model_dir)


if __name__ == '__main__':
    unittest_main()
