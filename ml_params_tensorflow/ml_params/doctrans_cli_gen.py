#!/usr/bin/env python
"""
Basic helper to generate CLI arguments for `doctrans` (see README.md)
"""

import sys
from argparse import ArgumentError
from collections import namedtuple
from importlib import import_module
from importlib.util import find_spec
from os import path

from ml_params_tensorflow import get_logger

p = (
    get_logger("doctrans_cli_gen").warning(
        "For `doctrans_cli_gen` to work well, you should `pip install inflect`"
    )
    or namedtuple("NotInflect", ("singular_noun",))(
        lambda s: s[:-2] if s.endswith("es") else s[:-1]
    )
    if find_spec("inflect") is None
    else getattr(import_module("inflect"), "engine")()
)


def main(argv=None):
    """
    CLI main function for doctrans_cli_gen

    :param argv: argv, defaults to ```sys.argv```
    :type argv: ```Optional[List[str]]```
    """
    argv = argv or sys.argv
    usage = "Usage: {executable} {script} <module_name>".format(
        executable=sys.executable, script=argv[0]
    )
    if len(argv) != 2:
        raise ArgumentError(None, usage)
    elif len(argv) > 1 and argv[1] in frozenset(("-h", "--help")):
        print(usage)
        exit()

    mod_pl = argv[1]
    mod = (lambda s: mod_pl if s is False else s)(p.singular_noun(mod_pl))
    mod_cap = mod.capitalize()
    tab = " " * 4

    print(
        " ".join(
            "{tab}--{opt} '{val!s}'".format(tab=tab, opt=opt.replace("_", "-"), val=val)
            for opt, val in dict(
                name_tpl="{name}Config",
                input_mapping="ml_params_tensorflow.ml_params.type_generators.exposed_{mod_pl}".format(
                    mod_pl=mod_pl
                ),
                prepend='""" Generated {mod_cap} CLI parsers """\\n'
                "import tensorflow as tf\\n"
                "from typing import Optional, Union\\n"
                "from {typing} import Literal\\n\\n"
                "from dataclasses import dataclass\\n\\n"
                "from yaml import safe_load as loads\\n\\n"
                "NoneType = type(None)\\n".format(
                    mod_cap=mod_cap,
                    typing="typing"
                    if sys.version_info[:2] < (3, 8)
                    else "typing_extensions",
                ),
                imports_from_file="tf.keras.{mod_pl}.{mod_cap}".format(
                    mod_pl=mod_pl, mod_cap=mod_cap
                ),
                type="argparse",
                output_filename=path.join(
                    "ml_params_tensorflow",
                    "ml_params",
                    "{mod_pl}.py".format(mod_pl=mod_pl),
                ),
                decorator="dataclass",
            ).items()
        )
    )


def run_main():
    """" Run the `main` function if `__name__ == "__main__"` """
    if __name__ == "__main__":
        main()


run_main()
