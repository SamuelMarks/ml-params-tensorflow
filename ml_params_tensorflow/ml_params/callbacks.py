""" Generated Callback config classes """
from __future__ import absolute_import, division, print_function

from yaml import safe_load as loads

NoneType = type(None)


def BaseLoggerConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Callback that accumulates epoch averages of metrics.

This callback is automatically applied to every Keras model."""
    argument_parser.add_argument(
        "--stateful_metrics",
        type=str,
        help="""Iterable of string names of metrics that
        should *not* be averaged over an epoch.
        Metrics in this list will be logged as-is in `on_epoch_end`.
        All others will be averaged in `on_epoch_end`.""",
    )
    return argument_parser


def CSVLoggerConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Callback that streams epoch results to a CSV file.

Supports all values that can be represented as a string,
including 1D iterables such as `np.ndarray`.

Example:

```python
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
```"""
    argument_parser.add_argument(
        "--filename",
        type=str,
        help="Filename of the CSV file, e.g. `'run/log.csv'`.",
        required=True,
        default=",",
    )
    argument_parser.add_argument(
        "--separator",
        type=str,
        help="String used to separate elements in the CSV file.",
        required=True,
        default=",",
    )
    argument_parser.add_argument(
        "--append",
        type=bool,
        help="""Boolean. True: append if file exists (useful for continuing
        training). False: overwrite existing file.""",
        required=True,
        default=False,
    )
    return argument_parser


def CallbackConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Abstract base class used to build new callbacks.

Attributes:
    params: Dict. Training parameters
        (eg. verbosity, batch size, number of epochs...).
    model: Instance of `keras.models.Model`.
        Reference of the model being trained.

The `logs` dictionary that callback methods
take as argument will contain keys for quantities relevant to
the current batch or epoch (see method-specific docstrings)."""
    return argument_parser


def CallbackListConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Container abstracting a list of callbacks."
    argument_parser.add_argument(
        "--callbacks", type=str, help="List of `Callback` instances."
    )
    argument_parser.add_argument(
        "--add_history",
        type=bool,
        help="""Whether a `History` callback should be added, if one does not
    already exist in the `callbacks` list.""",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--add_progbar",
        type=bool,
        help="""Whether a `ProgbarLogger` callback should be added, if one
    does not already exist in the `callbacks` list.""",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--model", type=str, help="The `Model` these callbacks are used with."
    )
    argument_parser.add_argument(
        "--params",
        type=str,
        help="""If provided, parameters will be passed to each `Callback` via
    `Callback.set_params`.""",
    )
    return argument_parser


def EarlyStoppingConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Stop training when a monitored metric has stopped improving.

Assuming the goal of a training is to minimize the loss. With this, the
metric to be monitored would be `'loss'`, and mode would be `'min'`. A
`model.fit()` training loop will check at end of every epoch whether
the loss is no longer decreasing, considering the `min_delta` and
`patience` if applicable. Once it's found no longer decreasing,
`model.stop_training` is marked True and the training terminates.

The quantity to be monitored needs to be available in `logs` dict.
To make it so, pass the loss or metrics at `model.compile()`.


Example:

>>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
>>> # This callback will stop the training when there is no improvement in
>>> # the validation loss for three consecutive epochs.
>>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
>>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
>>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
...                     epochs=10, batch_size=1, callbacks=[callback],
...                     verbose=0)
>>> len(history.history['loss'])  # Only 4 epochs are run.
4"""
    argument_parser.add_argument(
        "--monitor",
        type=str,
        help="Quantity to be monitored.",
        required=True,
        default="val_loss",
    )
    argument_parser.add_argument(
        "--min_delta",
        type=int,
        help="""Minimum change in the monitored quantity
      to qualify as an improvement, i.e. an absolute
      change of less than min_delta, will count as no
      improvement.""",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--patience",
        type=int,
        help="""Number of epochs with no improvement
      after which training will be stopped.""",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--verbose", type=int, help="verbosity mode.", required=True, default=0
    )
    argument_parser.add_argument(
        "--mode",
        type=str,
        help="""One of `{"auto", "min", "max"}`. In `min` mode,
      training will stop when the quantity
      monitored has stopped decreasing; in `"max"`
      mode it will stop when the quantity
      monitored has stopped increasing; in `"auto"`
      mode, the direction is automatically inferred
      from the name of the monitored quantity.""",
        required=True,
        default="auto",
    )
    argument_parser.add_argument(
        "--baseline",
        type=str,
        help="""Baseline value for the monitored quantity.
      Training will stop if the model doesn't show improvement over the
      baseline.""",
    )
    argument_parser.add_argument(
        "--restore_best_weights",
        type=bool,
        help="""Whether to restore model weights from
      the epoch with the best value of the monitored quantity.
      If False, the model weights obtained at the last step of
      training are used.""",
        required=True,
        default=False,
    )
    return argument_parser


def HistoryConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Callback that records events into a `History` object.

This callback is automatically applied to
every Keras model. The `History` object
gets returned by the `fit` method of models."""
    return argument_parser


def LambdaCallbackConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Callback for creating simple, custom callbacks on-the-fly.

This callback is constructed with anonymous functions that will be called
at the appropriate time. Note that the callbacks expects positional
arguments, as:

- `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
  `epoch`, `logs`
- `on_batch_begin` and `on_batch_end` expect two positional arguments:
  `batch`, `logs`
- `on_train_begin` and `on_train_end` expect one positional argument:
  `logs`


Example:

```python
# Print the batch number at the beginning of every batch.
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: print(batch))

# Stream the epoch loss to a file in JSON format. The file content
# is not well-formed JSON but rather has a JSON object per line.
import json
json_log = open('loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\\n'),
    on_train_end=lambda logs: json_log.close()
)

# Terminate some processes after having finished model training.
processes = ...
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
        p.terminate() for p in processes if p.is_alive()])

model.fit(...,
          callbacks=[batch_print_callback,
                     json_logging_callback,
                     cleanup_callback])
```"""
    argument_parser.add_argument(
        "--on_epoch_begin", type=str, help="called at the beginning of every epoch."
    )
    argument_parser.add_argument(
        "--on_epoch_end", type=str, help="called at the end of every epoch."
    )
    argument_parser.add_argument(
        "--on_batch_begin", type=str, help="called at the beginning of every batch."
    )
    argument_parser.add_argument(
        "--on_batch_end", type=str, help="called at the end of every batch."
    )
    argument_parser.add_argument(
        "--on_train_begin", type=str, help="called at the beginning of model training."
    )
    argument_parser.add_argument(
        "--on_train_end", type=str, help="called at the end of model training."
    )
    argument_parser.add_argument("--kwargs", type=loads, help="")
    return argument_parser


def LearningRateSchedulerConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Learning rate scheduler.

At the beginning of every epoch, this callback gets the updated learning rate
value from `schedule` function provided at `__init__`, with the current epoch
and current learning rate, and applies the updated learning rate
on the optimizer.


Example:

>>> # This function keeps the initial learning rate for the first ten epochs
>>> # and decreases it exponentially after that.
>>> def scheduler(epoch, lr):
...   if epoch < 10:
...     return lr
...   else:
...     return lr * tf.math.exp(-0.1)
>>>
>>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
>>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
>>> round(model.optimizer.lr.numpy(), 5)
0.01

>>> callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
>>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
...                     epochs=15, callbacks=[callback], verbose=0)
>>> round(model.optimizer.lr.numpy(), 5)
0.00607"""
    argument_parser.add_argument(
        "--schedule",
        type=int,
        help="""a function that takes an epoch index (integer, indexed from 0)
      and current learning rate (float) as inputs and returns a new
      learning rate as output (float).""",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--verbose",
        type=int,
        help="int. 0: quiet, 1: update messages.",
        required=True,
        default=0,
    )
    return argument_parser


def ModelCheckpointConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Callback to save the Keras model or model weights at some frequency.

`ModelCheckpoint` callback is used in conjunction with training using
`model.fit()` to save a model or weights (in a checkpoint file) at some
interval, so the model or weights can be loaded later to continue the training
from the state saved.

A few options this callback provides include:

- Whether to only keep the model that has achieved the "best performance" so
  far, or whether to save the model at the end of every epoch regardless of
  performance.
- Definition of 'best'; which quantity to monitor and whether it should be
  maximized or minimized.
- The frequency it should save at. Currently, the callback supports saving at
  the end of every epoch, or after a fixed number of training batches.
- Whether only weights are saved, or the whole model is saved.

Example:

```python
EPOCHS = 10
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

# Model weights are saved at the end of every epoch, if it's the best seen
# so far.
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)
```"""
    argument_parser.add_argument(
        "--filepath",
        type=str,
        help="""string or `PathLike`, path to save the model file. `filepath`
      can contain named formatting options, which will be filled the value of
      `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
      `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model
      checkpoints will be saved with the epoch number and the validation loss
      in the filename.""",
        required=True,
        default="val_loss",
    )
    argument_parser.add_argument(
        "--monitor",
        type=str,
        help="quantity to monitor.",
        required=True,
        default="val_loss",
    )
    argument_parser.add_argument(
        "--verbose", type=int, help="verbosity mode, 0 or 1.", required=True, default=0
    )
    argument_parser.add_argument(
        "--save_best_only",
        type=bool,
        help="""if `save_best_only=True`, the latest best model according
      to the quantity monitored will not be overwritten.
      If `filepath` doesn't contain formatting options like `{epoch}` then
      `filepath` will be overwritten by each new better model.""",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--mode",
        type=str,
        help="""one of {auto, min, max}. If `save_best_only=True`, the decision to
      overwrite the current save file is made based on either the maximization
      or the minimization of the monitored quantity. For `val_acc`, this
      should be `max`, for `val_loss` this should be `min`, etc. In `auto`
      mode, the direction is automatically inferred from the name of the
      monitored quantity.""",
        required=True,
        default="auto",
    )
    argument_parser.add_argument(
        "--save_weights_only",
        type=bool,
        help="""if True, then only the model's weights will be saved
      (`model.save_weights(filepath)`), else the full model is saved
      (`model.save(filepath)`).""",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--save_freq",
        type=str,
        help="""`'epoch'` or integer. When using `'epoch'`, the callback saves
      the model after each epoch. When using integer, the callback saves the
      model at end of this many batches. If the `Model` is compiled with
      `experimental_steps_per_execution=N`, then the saving criteria will be
      checked every Nth batch. Note that if the saving isn't aligned to
      epochs, the monitored metric may potentially be less reliable (it
      could reflect as little as 1 batch, since the metrics get reset every
      epoch).""",
        required=True,
        default="epoch",
    )
    argument_parser.add_argument(
        "--options",
        type=str,
        help="""Optional `tf.train.CheckpointOptions` object if
      `save_weights_only` is true or optional `tf.saved_model.SavedOptions`
      object if `save_weights_only` is false.""",
        required=True,
    )
    argument_parser.add_argument(
        "--kwargs",
        type=loads,
        help="""Additional arguments for backwards compatibility. Possible key
      is `period`.""",
    )
    return argument_parser


def ProgbarLoggerConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Callback that prints metrics to stdout.


Raises:
    ValueError: In case of invalid `count_mode`."""
    argument_parser.add_argument(
        "--count_mode",
        type=str,
        help="""One of `"steps"` or `"samples"`.
        Whether the progress bar should
        count samples seen or steps (batches) seen.""",
        required=True,
        default="samples",
    )
    argument_parser.add_argument(
        "--stateful_metrics",
        type=str,
        help="""Iterable of string names of metrics that
        should *not* be averaged over an epoch.
        Metrics in this list will be logged as-is.
        All others will be averaged over time (e.g. loss, etc).
        If not provided,""",
        required=True,
        default="the `Model`'s metrics",
    )
    return argument_parser


def ReduceLROnPlateauConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Reduce learning rate when a metric has stopped improving.

Models often benefit from reducing the learning rate by a factor
of 2-10 once learning stagnates. This callback monitors a
quantity and if no improvement is seen for a 'patience' number
of epochs, the learning rate is reduced.

Example:

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```"""
    argument_parser.add_argument(
        "--monitor",
        type=str,
        help="quantity to be monitored.",
        required=True,
        default="val_loss",
    )
    argument_parser.add_argument(
        "--factor",
        type=float,
        help="""factor by which the learning rate will be reduced.
      `new_lr = lr * factor`.""",
        required=True,
        default=0.1,
    )
    argument_parser.add_argument(
        "--patience",
        type=int,
        help="""number of epochs with no improvement after which learning rate
      will be reduced.""",
        required=True,
        default=10,
    )
    argument_parser.add_argument(
        "--verbose",
        type=int,
        help="int. 0: quiet, 1: update messages.",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--mode",
        type=str,
        help="""one of `{'auto', 'min', 'max'}`. In `'min'` mode,
      the learning rate will be reduced when the
      quantity monitored has stopped decreasing; in `'max'` mode it will be
      reduced when the quantity monitored has stopped increasing; in `'auto'`
      mode, the direction is automatically inferred from the name of the
      monitored quantity.""",
        required=True,
        default="auto",
    )
    argument_parser.add_argument(
        "--min_delta",
        type=float,
        help="""threshold for measuring the new optimum, to only focus on
      significant changes.""",
        required=True,
        default=0.0001,
    )
    argument_parser.add_argument(
        "--cooldown",
        type=int,
        help="""number of epochs to wait before resuming normal operation after
      lr has been reduced.""",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--min_lr",
        type=int,
        help="lower bound on the learning rate.",
        required=True,
        default=0,
    )
    argument_parser.add_argument("--kwargs", type=loads, help="")
    return argument_parser


def RemoteMonitorConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Callback used to stream events to a server.

Requires the `requests` library.
Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
HTTP POST, with a `data` argument which is a
JSON-encoded dictionary of event data.
If `send_as_json=True`, the content type of the request will be
`"application/json"`.
Otherwise the serialized JSON will be sent within a form."""
    argument_parser.add_argument(
        "--root",
        type=str,
        help="String; root url of the target server.",
        required=True,
        default="http://localhost:9000",
    )
    argument_parser.add_argument(
        "--path",
        type=str,
        help="String; path relative to `root` to which the events will be sent.",
        required=True,
        default="/publish/epoch/end/",
    )
    argument_parser.add_argument(
        "--field",
        type=str,
        help="""String; JSON field under which the data will be stored.
      The field is used only if the payload is sent within a form
      (i.e. send_as_json is set to False).""",
        required=True,
        default="data",
    )
    argument_parser.add_argument(
        "--headers", type=str, help="Dictionary; optional custom HTTP headers."
    )
    argument_parser.add_argument(
        "--send_as_json",
        type=bool,
        help="""Boolean; whether the request should be
      sent as `"application/json"`.""",
        required=True,
        default=False,
    )
    return argument_parser


def TensorBoardConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Enable visualizations for TensorBoard.

TensorBoard is a visualization tool provided with TensorFlow.

This callback logs events for TensorBoard, including:

* Metrics summary plots
* Training graph visualization
* Activation histograms
* Sampled profiling

If you have installed TensorFlow with pip, you should be able
to launch TensorBoard from the command line:

```
tensorboard --logdir=path_to_your_logs
```

You can find more information about TensorBoard
[here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

Example (Basic):

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
# run the tensorboard command to view the visualizations.
```

Example (Profile):

```python
# profile a single batch, e.g. the 5th batch.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs',
                                                      profile_batch=5)
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
# Now run the tensorboard command to view the visualizations (profile plugin).

# profile a range of batches, e.g. from 10 to 20.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs',
                                                      profile_batch='10,20')
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
# Now run the tensorboard command to view the visualizations (profile plugin).
```


Raises:
    ValueError: If histogram_freq is set and no validation data is provided."""
    argument_parser.add_argument(
        "--log_dir",
        type=str,
        help="""the path of the directory where to save the log files to be
      parsed by TensorBoard.""",
        required=True,
        default="logs",
    )
    argument_parser.add_argument(
        "--histogram_freq",
        type=int,
        help="""frequency (in epochs) at which to compute activation and
      weight histograms for the layers of the model. If set to 0, histograms
      won't be computed. Validation data (or split) must be specified for
      histogram visualizations.""",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--write_graph",
        type=bool,
        help="""whether to visualize the graph in TensorBoard. The log file
      can become quite large when write_graph is set to True.""",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--write_images",
        type=bool,
        help="""whether to write model weights to visualize as image in
      TensorBoard.""",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--update_freq",
        type=str,
        help="""`'batch'` or `'epoch'` or integer. When using `'batch'`,
      writes the losses and metrics to TensorBoard after each batch. The same
      applies for `'epoch'`. If using an integer, let's say `1000`, the
      callback will write the metrics and losses to TensorBoard every 1000
      batches. Note that writing too frequently to TensorBoard can slow down
      your training.""",
        required=True,
        default="epoch",
    )
    argument_parser.add_argument(
        "--profile_batch",
        type=int,
        help="""Profile the batch(es) to sample compute characteristics.
      profile_batch must be a non-negative integer or a tuple of integers.
      A pair of positive integers signify a range of batches to profile.
      By default, it will profile the second batch. Set profile_batch=0
      to disable profiling.""",
        required=True,
        default=2,
    )
    argument_parser.add_argument(
        "--embeddings_freq",
        type=int,
        help="""frequency (in epochs) at which embedding layers will be
      visualized. If set to 0, embeddings won't be visualized.""",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--embeddings_metadata",
        type=str,
        help="""a dictionary which maps layer name to a file name in
      which metadata for this embedding layer is saved. See the
      [details](
        https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
      about metadata files format. In case if the same metadata file is
      used for all embedding layers, string can be passed.""",
    )
    argument_parser.add_argument("--kwargs", type=loads, help="")
    return argument_parser


def TerminateOnNaNConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = (
        "Callback that terminates training when a NaN loss is encountered."
    )
    return argument_parser


__all__ = [
    "BaseLoggerConfig",
    "CSVLoggerConfig",
    "CallbackConfig",
    "CallbackListConfig",
    "EarlyStoppingConfig",
    "HistoryConfig",
    "LambdaCallbackConfig",
    "LearningRateSchedulerConfig",
    "ModelCheckpointConfig",
    "ProgbarLoggerConfig",
    "ReduceLROnPlateauConfig",
    "RemoteMonitorConfig",
    "TensorBoardConfig",
    "TerminateOnNaNConfig",
]
