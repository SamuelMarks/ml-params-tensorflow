"""
Tests for the doctrans_cli_gen script
"""

from argparse import ArgumentError
from io import StringIO
from unittest import TestCase
from unittest.mock import MagicMock, patch

import ml_params_tensorflow.ml_params.callbacks
import ml_params_tensorflow.ml_params.doctrans_cli_gen
import ml_params_tensorflow.ml_params.losses
import ml_params_tensorflow.ml_params.metrics
import ml_params_tensorflow.ml_params.optimizers
from ml_params_tensorflow.tests.utils_for_tests import unittest_main


class TestCliGen(TestCase):
    """
    Tests whether doctrans_cli_gen are exposed
    """

    def test_run_main(self) -> None:
        """ Tests that main will be called """

        with patch(
            "ml_params_tensorflow.ml_params.doctrans_cli_gen.main",
            new_callable=MagicMock,
        ) as f, patch(
            "ml_params_tensorflow.ml_params.doctrans_cli_gen.__name__", "__main__"
        ):
            ml_params_tensorflow.ml_params.doctrans_cli_gen.run_main()
            self.assertEqual(f.call_count, 1)

    def test_main(self) -> None:
        """ Tests that main will be called """

        self.assertRaises(
            ArgumentError,
            lambda: ml_params_tensorflow.ml_params.doctrans_cli_gen.main(
                ["python", "bar", "foo"]
            ),
        )
        with self.assertRaises(SystemExit), patch("sys.stdout", new_callable=StringIO):
            self.assertIsNone(
                ml_params_tensorflow.ml_params.doctrans_cli_gen.main(["bar", "-h"])
            )

        with patch("sys.stdout", new_callable=StringIO):
            self.assertIsNone(
                ml_params_tensorflow.ml_params.doctrans_cli_gen.main(["bar", "howzat"])
            )


unittest_main()
