#!/usr/bin/env bash

# TODO: A cross-platform YAML config to replace this script, and pass along to `python -m doctrans`

python -m doctrans --version

declare -r input_file='ml_params_tensorflow/ml_params/type_generators.py'
declare -r output_file='ml_params_tensorflow/ml_params/trainer.py'

printf 'Setting type annotations of `TensorFlowTrainer` class to match those in %s\n' "$input_file"

python -m doctrans sync_properties \
                   --input-file "$input_file" \
                   --input-eval \
                   --output-file "$output_file" \
                   --input-param 'exposed_loss_keys' \
                   --output-param 'TensorFlowTrainer.train.loss' \
                   --input-param 'exposed_optimizer_keys' \
                   --output-param 'TensorFlowTrainer.train.optimizer' \
                   --input-param 'exposed_datasets_keys' \
                   --output-param 'TensorFlowTrainer.load_data.dataset_name'

python -m doctrans sync_properties \
                   --input-file "$input_file" \
                   --input-eval \
                   --output-file "$output_file" \
                   --output-param-wrap 'Optional[List[{output_param}]]' \
                   --input-param 'exposed_callbacks_keys' \
                   --output-param 'TensorFlowTrainer.train.callbacks' \
                   --input-param 'exposed_metrics_keys' \
                   --output-param 'TensorFlowTrainer.train.metrics'

python -m doctrans sync_properties \
                   --input-file "$input_file" \
                   --input-eval \
                   --output-file "$output_file" \
                   --input-param 'exposed_loss_keys' \
                   --output-param 'TensorFlowTrainer.train.loss' \
                   --input-param 'exposed_optimizer_keys' \
                   --output-param 'TensorFlowTrainer.train.optimizer' \
                   --input-param 'exposed_datasets_keys' \
                   --output-param 'TensorFlowTrainer.load_data.dataset_name'

python -m doctrans sync_properties \
                   --input-file "$input_file" \
                   --input-eval \
                   --output-file "$output_file" \
                   --input-param 'exposed_applications_keys' \
                   --output-param-wrap 'Union[{output_param}, AnyStr]' \
                   --output-param 'TensorFlowTrainer.load_model.model'

printf 'Setting type annotations of `load_data_from_tfds_or_ml_prepare` function to match those in %s\n' "$input_file"

python -m doctrans sync_properties \
                   --input-file "$input_file" \
                   --input-eval \
                   --output-file 'ml_params_tensorflow/ml_params/datasets.py' \
                   --output-param-wrap 'Union[{output_param}, AnyStr]' \
                   --input-param 'exposed_datasets_keys' \
                   --output-param 'load_data_from_tfds_or_ml_prepare.dataset_name'

declare -ra generate=('callbacks' 'losses' 'metrics' 'optimizers')
printf 'Using %s as truth to generate CLIs for %s\n' "$input_file" "${generate[*]}"

for name in ${generate[*]}; do
    rm 'ml_params_tensorflow/ml_params/'"$name"'.py';
    python -m ml_params_tensorflow.ml_params.doctrans_cli_gen "$name" 2>/dev/null | xargs python -m doctrans gen;
done

fd -HIepy -x sh -c 'autoflake --remove-all-unused-imports -i "$0" && isort --atomic "$0" && python -m black "$0"' {} \;

printf '\nFIN\n'
