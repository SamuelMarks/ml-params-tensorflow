ml_params_tensorflow
===============
![Python version range](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Linting, testing, and coverage](https://github.com/SamuelMarks/ml-params-tensorflow/workflows/Linting,%20testing,%20and%20coverage/badge.svg)](https://github.com/SamuelMarks/ml-params-tensorflow/actions)
![Tested OSs, others may work](https://img.shields.io/badge/Tested%20on-Linux%20|%20macOS-green)
![Documentation coverage](.github/doccoverage.svg)
[![codecov](https://codecov.io/gh/SamuelMarks/ml-params-tensorflow/branch/master/graph/badge.svg)](https://codecov.io/gh/SamuelMarks/ml-params-tensorflow)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

[TensorFlow](https://tensorflow.org) implementation of the [ml-params](https://github.com/SamuelMarks/ml-params) API and CLI.

The purpose of ml-params is to expose the hooks and levers of ML experiments for external usage, e.g., in GUIs, CLIs, REST & RPC APIs, and parameter and hyperparameter optimisers.

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

## Usage

After installing as above, follow usage from [ml-params](https://github.com/SamuelMarks/ml-params)

## Sibling projects

| Google | Other vendors |
| -------| ------------- |
| _[tensorflow](https://github.com/SamuelMarks/ml-params-tensorflow)_  | [pytorch](https://github.com/SamuelMarks/ml-params-pytorch) |
| [keras](https://github.com/SamuelMarks/ml-params-keras)  | [skorch](https://github.com/SamuelMarks/ml-params-skorch) |
| [flax](https://github.com/SamuelMarks/ml-params-flax) | [sklearn](https://github.com/SamuelMarks/ml-params-sklearn) |
| [trax](https://github.com/SamuelMarks/ml-params-trax) | [xgboost](https://github.com/SamuelMarks/ml-params-xgboost) |
| [jax](https://github.com/SamuelMarks/ml-params-jax) | [cntk](https://github.com/SamuelMarks/ml-params-cntk) |

## Related official projects

  - [ml-prepare](https://github.com/SamuelMarks/ml-prepare)

## Development guide

To make the development of _ml-params-tensorflow_ type safer and maintain consistency with the other ml-params implementing projects, the [cdd](https://github.com/offscale/cdd-python) was created.

When TensorFlow itself changes—i.e., a new major version of TensorFlow is releases—then run the `sync_properties`, as shown in the module-level docstring here [`ml_params_tensorflow/ml_params/type_generators.py`](ml_params_tensorflow/ml_params/type_generators.py);

To synchronise all the various other APIs, edit one and it'll translate to the others, but make sure you select which one is the gold-standard.

### Dependencies

    $ pip install python-cdd autoflake black isort

[`fd`](https://github.com/sharkdp/fd#installation) (or you can just use `find`, rewrite to `find -type f -name '*.py' -exec`)

### Examples

NOTE: The below steps are available in a script within [sync_scripts](sync_scripts).

As an example, using the `class TensorFlowTrainer` methods as truth, this will update the CLI parsers and config classes:

    $ python -m cdd sync --class 'ml_params_tensorflow/ml_params/config.py' \
                         --class-name 'TrainConfig' \
                         --function 'ml_params_tensorflow/ml_params/trainer.py' \
                         --function-name 'TensorFlowTrainer.train' \
                         --argparse-function 'ml_params_tensorflow/ml_params/cli.py' \
                         --argparse-function-name 'train_parser' \
                         --truth 'function'

    $ python -m cdd sync --class 'ml_params_tensorflow/ml_params/config.py' \
                         --class-name 'LoadDataConfig' \
                         --function 'ml_params_tensorflow/ml_params/trainer.py' \
                         --function-name 'TensorFlowTrainer.load_data' \
                         --argparse-function 'ml_params_tensorflow/ml_params/cli.py' \
                         --argparse-function-name 'load_data_parser' \
                         --truth 'function'

    $ python -m cdd sync --class 'ml_params_tensorflow/ml_params/config.py' \
                         --class-name 'LoadModelConfig' \
                         --function 'ml_params_tensorflow/ml_params/trainer.py' \
                         --function-name 'TensorFlowTrainer.load_model' \
                         --argparse-function 'ml_params_tensorflow/ml_params/cli.py' \
                         --argparse-function-name 'load_model_parser' \
                         --truth 'function'

Another example, that you'd run before ^, to generate custom config CLI parsers for members of `tf.keras.losses`:

    $ python -m cdd gen --name-tpl '{name}Config' \
                        --input-mapping 'ml_params_tensorflow.ml_params.type_generators.exposed_losses' \
                        --prepend '""" Generated Loss config classes """\nimport tensorflow as tf\n' \
                        --imports-from-file 'tf.keras.losses.Loss' \
                        --type 'argparse' \
                        --output-filename 'ml_params_tensorflow/ml_params/losses.py'

There's a bit of boilerplate here, so let's automate it:

    $ for name in 'callbacks' 'losses' 'metrics' 'optimizers'; do
        rm 'ml_params_tensorflow/ml_params/'"$name"'.py';        
        python -m ml_params_tensorflow.ml_params.cdd_cli_gen "$name" 2>/dev/null | xargs python -m cdd gen;
      done

Cleanup the code everywhere, removing unused imports and autolinting/autoformatting:

    $ fd -HIepy -x sh -c 'autoflake --remove-all-unused-imports -i "$0" && isort --atomic "$0" && python -m black "$0"' {} \;

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
