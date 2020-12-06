ml_params_tensorflow
===============
![Python version range](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Linting, testing, and coverage](https://github.com/SamuelMarks/ml-params-tensorflow/workflows/Linting,%20testing,%20and%20coverage/badge.svg)](https://github.com/SamuelMarks/ml-params-tensorflow/actions)
![Tested OSs, others may work](https://img.shields.io/badge/Tested%20on-Linux%20|%20macOS%20|%20Windows-green)
![Documentation coverage](.github/doccoverage.svg)
[![codecov](https://codecov.io/gh/SamuelMarks/ml-params-tensorflow/branch/master/graph/badge.svg)](https://codecov.io/gh/SamuelMarks/ml-params-tensorflow)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[TensorFlow](https://tensorflow.org) implementation of the [ml-params](https://github.com/SamuelMarks/ml-params) API and CLI.

The purpose of ml-params is to expose the hooks and levers of ML experiments for external usage, e.g., in GUIs, CLIs,
REST & RPC APIs, and parameter and hyperparameter optimisers.

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

To make the development of _ml-params-tensorflow_ type safer and maintain consistency with the other ml-params implementing projects, the [doctrans](https://github.com/SamuelMarks/doctrans) was created.

When TensorFlow itself changes—i.e., a new major version of TensorFlow is releases—then run the `sync_properties`, as shown in the module-level docstring here [`ml_params_tensorflow/ml_params/type_generators.py`](ml_params_tensorflow/ml_params/type_generators.py);

To synchronise all the various other APIs, edit one and it'll translate to the others, but make sure you select which one is the gold-standard.

As an example, using the `class TensorFlowTrainer` methods as truth, this will update the CLI parsers and config classes:

    python -m doctrans sync --class 'ml_params_tensorflow/ml_params/config.py' \
                            --class-name 'TrainConfig' \
                            --function 'ml_params_tensorflow/ml_params/trainer.py' \
                            --function-name 'TensorFlowTrainer.train' \
                            --argparse-function 'ml_params_tensorflow/ml_params/cli.py' \
                            --argparse-function-name 'train_parser' \
                            --truth 'function'

    python -m doctrans sync --class 'ml_params_tensorflow/ml_params/config.py' \
                            --class-name 'LoadDataConfig' \
                            --function 'ml_params_tensorflow/ml_params/trainer.py' \
                            --function-name 'TensorFlowTrainer.load_data' \
                            --argparse-function 'ml_params_tensorflow/ml_params/cli.py' \
                            --argparse-function-name 'load_data_parser' \
                            --truth 'function'

    python -m doctrans sync --class 'ml_params_tensorflow/ml_params/config.py' \
                            --class-name 'LoadModelConfig' \
                            --function 'ml_params_tensorflow/ml_params/trainer.py' \
                            --function-name 'TensorFlowTrainer.load_model' \
                            --argparse-function 'ml_params_tensorflow/ml_params/cli.py' \
                            --argparse-function-name 'load_model_parser' \
                            --truth 'function'

Another example, that you'd run before ^, to generate custom config classes for members of `tf.keras.losses`, `tf.keras.optimizers`, and `tf.keras.callbacks.Callback`:

    $ python -m doctrans gen --name-tpl '{name}Config' \
                             --input-mapping 'ml_params_tensorflow.ml_params.type_generators.exposed_losses' \
                             --prepend '""" Generated Loss config classes """\nimport tensorflow as tf\n' \
                             --imports-from-file 'tf.keras.losses.Loss' \
                             --type 'class' \
                             --output-filename 'ml_params_tensorflow/ml_params/losses.py'

    $ python -m doctrans gen --name-tpl '{name}Config' \
                             --input-mapping 'ml_params_tensorflow.ml_params.type_generators.exposed_optimizers' \
                             --prepend '""" Generated Optimizer config classes """\nimport tensorflow as tf\n' \
                             --imports-from-file 'tf.keras.optimizers.Optimizer' \
                             --type 'class' \
                             --output-filename 'ml_params_tensorflow/ml_params/optimizers.py'

    $ python -m doctrans gen --name-tpl '{name}Config' \
                             --input-mapping 'ml_params_tensorflow.ml_params.type_generators.exposed_callbacks' \
                             --prepend '""" Generated Callback config classes """\nimport tensorflow as tf\n' \
                             --imports-from-file 'tf.keras.callbacks.Callback' \
                             --type 'class' \
                             --output-filename 'ml_params_tensorflow/ml_params/callbacks.py'

    $ # ^ Or in a loop, just cycle through `{$plural, exposed_$plural, $plural.$capital($singular), $plural.py}`

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
