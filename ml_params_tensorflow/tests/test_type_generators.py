"""
Tests for utils for type_generators
"""
from unittest import TestCase

from ml_params_tensorflow.ml_params.type_generators import _is_main_name
from ml_params_tensorflow.tests.utils_for_tests import unittest_main


class TestTypeGenerators(TestCase):
    """
    Tests whether TypeGenerators work
    """

    def test__is_main_name(self) -> None:
        """
        Tests whether `_is_main_name` works
        """
        self.assertTrue(_is_main_name("FooBar"))
        self.assertTrue(_is_main_name("Foo"))
        self.assertFalse(_is_main_name("Foo_Bar"))
        self.assertFalse(_is_main_name("foo_bar"))


unittest_main()
