"""
Tests for the config
"""

from unittest import TestCase

from ml_params_tensorflow.tests.utils_for_tests import unittest_main
from ml_params_tensorflow.ml_params.config import from_string
from ml_params_tensorflow.ml_params.cli import self as Self


class TestConfig(TestCase):
    """
    Tests whether config members that aren't tested elsewhere work as advertised
    """

    def test_from_string(self) -> None:
        """ Tests that `from_string` will be called """

        self.assertEqual(
            from_string(Self, '{"model": null, "data": [5]}'),
            Self(model=None, data=[5]),
        )


unittest_main()
