"""
Tests for CLI parsers
"""
from argparse import ArgumentParser
from unittest import TestCase

from ml_params_tensorflow.ml_params.cli import (
    train_parser,
    load_data_parser,
    load_model_parser,
)
from ml_params_tensorflow.tests.utils_for_tests import unittest_main


class TestCli(TestCase):
    """
    Tests whether cli parsers return the right thing
    """

    def test_parsers(self) -> None:
        """
        Tests whether the CLI parsers return a collection with first element being `ArgumentParser`
        """
        for parser in train_parser, load_data_parser, load_model_parser:
            self.assertIsInstance(
                parser(argument_parser=ArgumentParser())[0], ArgumentParser
            )


unittest_main()
