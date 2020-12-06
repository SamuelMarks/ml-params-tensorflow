#!/usr/bin/env python
"""
Basic helper to generate CLI arguments for `doctrans` (see README.md)
"""

from argparse import ArgumentError
from os import path
from sys import argv, executable

import inflect

p = inflect.engine()

if __name__ == "__main__":
    usage = "Usage: {executable} {script} <module_name>".format(
        executable=executable, script=argv[0]
    )
    if len(argv) != 2:
        raise ArgumentError(None, usage)
    elif len(argv) > 1 and argv[1] in frozenset(("-h", "--help")):
        print(usage)
        exit()

    mod_pl = argv[1]
    mod = p.singular_noun(mod_pl)
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
                prepend='""" Generated Callback config classes """\\nimport tensorflow as tf\\n',
                imports_from_file="tf.keras.{mod_pl}.{mod_cap}".format(
                    mod_pl=mod_pl, mod_cap=mod_cap
                ),
                type="class",
                output_filename=path.join(
                    "ml_params_tensorflow",
                    "ml_params",
                    "{mod_pl}.py".format(mod_pl=mod_pl),
                ),
            ).items()
        )
    )
