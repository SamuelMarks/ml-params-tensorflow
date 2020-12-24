"""
The symbols to expose. This is the public API used by ml_params.
"""

from importlib import import_module

extra_symbols = {
    "loss"
    if mod == "losses"
    else mod: import_module("ml_params_tensorflow.ml_params.{mod}".format(mod=mod))
    for mod in ("callbacks", "losses", "metrics", "optimizers")
}

del import_module

__all__ = ["extra_symbols"]
