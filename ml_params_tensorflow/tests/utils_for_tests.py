"""
Shared utility functions used by tests
"""

from sys import version_info
from unittest import main


def unittest_main():
    """ Runs unittest.main if __main__ """
    if __name__ == "__main__":
        main()


# From https://github.com/Suor/funcy/blob/0ee7ae8/funcy/funcs.py#L34-L36
def rpartial(func, *args):
    """Partially applies last arguments."""
    return lambda *a: func(*(a + args))


TF_SUPPORTED = (3, 8) >= version_info[:2] > (3, 5)

__all__ = ["rpartial", "TF_SUPPORTED", "unittest_main"]
