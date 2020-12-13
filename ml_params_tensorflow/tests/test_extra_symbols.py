"""
Tests for extra_symbols
"""
from unittest import TestCase

from ml_params_tensorflow.ml_params.extra_symbols import extra_symbols
from ml_params_tensorflow.tests.utils_for_tests import unittest_main


class TestExtraSymbols(TestCase):
    """
    Tests for extra_symbols
    """

    def test_extra_symbols_keys(self) -> None:
        """
        Tests whether `extra_symbols` has the right keys
        """
        self.assertListEqual(
            sorted(extra_symbols.keys()), ["callbacks", "metrics", "optimizers"]
        )


unittest_main()
