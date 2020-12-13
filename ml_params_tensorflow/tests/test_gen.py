"""
Tests for the generated files
"""

from argparse import ArgumentParser
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
                            "Optional",
                            "absolute_import",
                            "dataclass",
                            "division",
                            "print_function",
                            "loads",
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

    def _call_all(self, module):
        """
        :param module: The module
        :type module: ```object```

        :return: Whether all the symbols in `__all__` resolve to a callable returning an instance of Argparse
        :rtype: ```bool```
        """
        self.assertIn("__all__", dir(module))
        return all(
            filterfalse(
                lambda name: isinstance(
                    getattr(module, name)(ArgumentParser()), ArgumentParser
                ),
                module.__all__,
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
        self.assertTrue(self._call_all(ml_params_tensorflow.ml_params.callbacks))

    def test_losses(self) -> None:
        """
        Tests whether `losses` has correct `__all__`
        """
        self.assertListEqual(
            ml_params_tensorflow.ml_params.losses.__all__,
            self.gen_all(ml_params_tensorflow.ml_params.losses),
        )
        self.assertTrue(self._call_all(ml_params_tensorflow.ml_params.losses))

    def test_metrics(self) -> None:
        """
        Tests whether `metrics` has correct `__all__`
        """
        self.assertListEqual(
            ml_params_tensorflow.ml_params.metrics.__all__,
            self.gen_all(ml_params_tensorflow.ml_params.metrics),
        )
        self.assertTrue(self._call_all(ml_params_tensorflow.ml_params.metrics))

    def test_optimizers(self) -> None:
        """
        Tests whether `optimizers` has correct `__all__`
        """
        self.assertListEqual(
            ml_params_tensorflow.ml_params.optimizers.__all__,
            self.gen_all(ml_params_tensorflow.ml_params.optimizers),
        )
        self.assertTrue(self._call_all(ml_params_tensorflow.ml_params.optimizers))


unittest_main()
