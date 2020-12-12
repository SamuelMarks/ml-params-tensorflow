"""
Tests for the generated files
"""

from functools import partial
from inspect import getmembers
from itertools import filterfalse
from operator import contains, itemgetter
from unittest import TestCase

import ml_params_tensorflow.ml_params.callbacks
import ml_params_tensorflow.ml_params.losses
import ml_params_tensorflow.ml_params.metrics
import ml_params_tensorflow.ml_params.optimizers
from ml_params_tensorflow.tests.utils_for_tests import rpartial, unittest_main


class TestGen(TestCase):
    """
    Tests whether symbols are exposed
    """

    @staticmethod
    def gen_all(module):
        """
        Generate the `__all__`

        :param module: Name of module
        :type module: ```str```

        :return: `__all__` contents
        :rtype: ```List[str]```
        """
        return sorted(
            filterfalse(
                partial(
                    contains,
                    frozenset(
                        (
                            "NoneType",
                            "absolute_import",
                            "division",
                            "print_function",
                        )
                    ),
                ),
                filterfalse(
                    rpartial(str.startswith, "_"),
                    map(
                        itemgetter(0),
                        getmembers(module),
                    ),
                ),
            )
        )

    def test_callbacks(self) -> None:
        """
        Tests whether `callbacks` has correct `__all__`
        """
        self.assertListEqual(
            ml_params_tensorflow.ml_params.callbacks.__all__,
            self.gen_all(ml_params_tensorflow.ml_params.callbacks),
        )

    def test_losses(self) -> None:
        """
        Tests whether `losses` has correct `__all__`
        """
        self.assertListEqual(
            ml_params_tensorflow.ml_params.losses.__all__,
            self.gen_all(ml_params_tensorflow.ml_params.losses),
        )

    def test_metrics(self) -> None:
        """
        Tests whether `metrics` has correct `__all__`
        """
        self.assertListEqual(
            ml_params_tensorflow.ml_params.metrics.__all__,
            self.gen_all(ml_params_tensorflow.ml_params.metrics),
        )

    def test_optimizers(self) -> None:
        """
        Tests whether `optimizers` has correct `__all__`
        """
        self.assertListEqual(
            ml_params_tensorflow.ml_params.optimizers.__all__,
            self.gen_all(ml_params_tensorflow.ml_params.optimizers),
        )


unittest_main()
