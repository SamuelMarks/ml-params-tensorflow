"""
Datasets test(s)
"""

from unittest import TestCase
from unittest.mock import patch, MagicMock

import tensorflow as tf

from ml_params_tensorflow.datasets import (
    normalize_img,
)
from ml_params_tensorflow.tests.utils_for_tests import unittest_main


class TestDatasetsUtils(TestCase):
    """
    Tests utility functions from the datasets module
    """

    def test_normalize_img(self) -> None:
        """
        Confirms correct result from normalize_img
        """
        label = "anything"
        res = normalize_img(
            tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16), label, 255
        )

        expect = tf.constant(
            [
                [0.00392157, 0.00784314],
                [0.01176471, 0.01568628],
                [0.01960784, 0.02352941],
            ],
            shape=(3, 2),
            dtype=tf.float32,
        )

        self.assertTrue(
            tf.size(tf.reduce_any(tf.reduce_all(tf.math.equal(res[0], expect))))
        )
        self.assertEqual(res[1], label)

    def test_load_data_from_tfds_or_ml_prepare_in_datasets2classes(self):
        """
        Tests `load_data_from_tfds_or_ml_prepare` when `dataset_name in datasets2classes`
        """
        d = {"a": 5}
        magic = MagicMock()
        with patch("ml_params_tensorflow.datasets.datasets2classes", d), patch(
            "ml_params_tensorflow.datasets.load_data_from_ml_prepare", magic
        ):
            import ml_params_tensorflow.datasets

            self.assertIsInstance(
                ml_params_tensorflow.datasets.load_data_from_tfds_or_ml_prepare("a"),
                MagicMock,
            )


unittest_main()