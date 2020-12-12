from importlib import import_module

extra_symbols = {
    mod: import_module("ml_params_tensorflow.ml_params.{mod}".format(mod=mod))
    for mod in ("callbacks", "metrics", "optimizers")
}

del import_module

__all__ = ["extra_symbols"]
