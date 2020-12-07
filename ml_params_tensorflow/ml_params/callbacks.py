""" Generated Callback config classes """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BaseLoggerConfig(object):
    """
    Callback that accumulates epoch averages of metrics.

    This callback is automatically applied to every Keras model.

    :cvar stateful_metrics: Iterable of string names of metrics that
        should *not* be averaged over an epoch.
        Metrics in this list will be logged as-is in `on_epoch_end`.
        All others will be averaged in `on_epoch_end`. Defaults to None"""

    stateful_metrics: NoneType = None


class CSVLoggerConfig(object):
    """
    Callback that streams epoch results to a CSV file.

    Supports all values that can be represented as a string,
    including 1D iterables such as `np.ndarray`.

    Example:

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    :cvar filename: Filename of the CSV file, e.g. `'run/log.csv'`.
    :cvar separator: String used to separate elements in the CSV file. Defaults to ,
    :cvar append: Boolean. True: append if file exists (useful for continuing
        training). False: overwrite existing file. Defaults to False"""

    filename = None
    separator: str = ","
    append: bool = False


class CallbackConfig(object):
    """
    Abstract base class used to build new callbacks.

    Attributes:
        params: Dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: Instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch (see method-specific docstrings)."""


class CallbackListConfig(object):
    """
    Container abstracting a list of callbacks."""


class EarlyStoppingConfig(object):
    """
    Stop training when a monitored metric has stopped improving.

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
    4

    :cvar monitor: Quantity to be monitored. Defaults to val_loss
    :cvar min_delta: Minimum change in the monitored quantity
      to qualify as an improvement, i.e. an absolute
      change of less than min_delta, will count as no
      improvement. Defaults to 0
    :cvar patience: Number of epochs with no improvement
      after which training will be stopped. Defaults to 0
    :cvar verbose: verbosity mode. Defaults to 0
    :cvar mode: One of `{"auto", "min", "max"}`. In `min` mode,
      training will stop when the quantity
      monitored has stopped decreasing; in `"max"`
      mode it will stop when the quantity
      monitored has stopped increasing; in `"auto"`
      mode, the direction is automatically inferred
      from the name of the monitored quantity. Defaults to auto
    :cvar baseline: Baseline value for the monitored quantity.
      Training will stop if the model doesn't show improvement over the
      baseline. Defaults to None
    :cvar restore_best_weights: Whether to restore model weights from
      the epoch with the best value of the monitored quantity.
      If False, the model weights obtained at the last step of
      training are used. Defaults to False"""

    monitor: str = "val_loss"
    min_delta: int = 0
    patience: int = 0
    verbose: int = 0
    mode: str = "auto"
    baseline: NoneType = None
    restore_best_weights: bool = False


class HistoryConfig(object):
    """
    Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models."""


class LambdaCallbackConfig(object):
    """
    Callback for creating simple, custom callbacks on-the-fly.

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
    ```

    :cvar on_epoch_begin: called at the beginning of every epoch. Defaults to None
    :cvar on_epoch_end: called at the end of every epoch. Defaults to None
    :cvar on_batch_begin: called at the beginning of every batch. Defaults to None
    :cvar on_batch_end: called at the end of every batch. Defaults to None
    :cvar on_train_begin: called at the beginning of model training. Defaults to None
    :cvar on_train_end: called at the end of model training. Defaults to None"""

    on_epoch_begin: NoneType = None
    on_epoch_end: NoneType = None
    on_batch_begin: NoneType = None
    on_batch_end: NoneType = None
    on_train_begin: NoneType = None
    on_train_end: NoneType = None


class LearningRateSchedulerConfig(object):
    """
    Learning rate scheduler.

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
    0.00607

    :cvar schedule: a function that takes an epoch index (integer, indexed from 0)
      and current learning rate (float) as inputs and returns a new
      learning rate as output (float).
    :cvar verbose: int. 0: quiet, 1: update messages. Defaults to 0"""

    schedule = None
    verbose: int = 0


class ModelCheckpointConfig(object):
    """
    Callback to save the Keras model or model weights at some frequency.

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
    ```

    :cvar filepath: string or `PathLike`, path to save the model file. `filepath`
      can contain named formatting options, which will be filled the value of
      `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
      `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model
      checkpoints will be saved with the epoch number and the validation loss
      in the filename.
    :cvar monitor: quantity to monitor. Defaults to val_loss
    :cvar verbose: verbosity mode, 0 or 1. Defaults to 0
    :cvar save_best_only: if `save_best_only=True`, the latest best model according
      to the quantity monitored will not be overwritten.
      If `filepath` doesn't contain formatting options like `{epoch}` then
      `filepath` will be overwritten by each new better model. Defaults to False
    :cvar mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
      overwrite the current save file is made based on either the maximization
      or the minimization of the monitored quantity. For `val_acc`, this
      should be `max`, for `val_loss` this should be `min`, etc. In `auto`
      mode, the direction is automatically inferred from the name of the
      monitored quantity. Defaults to auto
    :cvar save_weights_only: if True, then only the model's weights will be saved
      (`model.save_weights(filepath)`), else the full model is saved
      (`model.save(filepath)`). Defaults to False
    :cvar save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
      the model after each epoch. When using integer, the callback saves the
      model at end of this many batches. If the `Model` is compiled with
      `experimental_steps_per_execution=N`, then the saving criteria will be
      checked every Nth batch. Note that if the saving isn't aligned to
      epochs, the monitored metric may potentially be less reliable (it
      could reflect as little as 1 batch, since the metrics get reset every
      epoch). Defaults to 'epoch'
    :cvar options: Optional `tf.train.CheckpointOptions` object if
      `save_weights_only` is true or optional `tf.saved_model.SavedOptions`
      object if `save_weights_only` is false. Defaults to None
    :cvar kwargs: Additional arguments for backwards compatibility. Possible key
      is `period`."""

    filepath = None
    monitor: str = "val_loss"
    verbose: int = 0
    save_best_only: bool = False
    mode: str = "auto"
    save_weights_only: bool = False
    save_freq: str = "epoch"
    options: NoneType = None
    kwargs: dict = {}


class ProgbarLoggerConfig(object):
    """
    Callback that prints metrics to stdout.


    Raises:
        ValueError: In case of invalid `count_mode`.

    :cvar count_mode: One of `"steps"` or `"samples"`.
        Whether the progress bar should
        count samples seen or steps (batches) seen. Defaults to samples
    :cvar stateful_metrics: Iterable of string names of metrics that
        should *not* be averaged over an epoch.
        Metrics in this list will be logged as-is.
        All others will be averaged over time (e.g. loss, etc).
        If not provided, Defaults to the `Model`'s metrics"""

    count_mode: str = "samples"
    stateful_metrics: NoneType = "```the `Model`'s metrics```"


class ReduceLROnPlateauConfig(object):
    """
    Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Example:

    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```

    :cvar monitor: quantity to be monitored. Defaults to val_loss
    :cvar factor: factor by which the learning rate will be reduced.
      `new_lr = lr * factor`. Defaults to 0.1
    :cvar patience: number of epochs with no improvement after which learning rate
      will be reduced. Defaults to 10
    :cvar verbose: int. 0: quiet, 1: update messages. Defaults to 0
    :cvar mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
      the learning rate will be reduced when the
      quantity monitored has stopped decreasing; in `'max'` mode it will be
      reduced when the quantity monitored has stopped increasing; in `'auto'`
      mode, the direction is automatically inferred from the name of the
      monitored quantity. Defaults to auto
    :cvar min_delta: threshold for measuring the new optimum, to only focus on
      significant changes. Defaults to 0.0001
    :cvar cooldown: number of epochs to wait before resuming normal operation after
      lr has been reduced. Defaults to 0
    :cvar min_lr: lower bound on the learning rate. Defaults to 0"""

    monitor: str = "val_loss"
    factor: float = 0.1
    patience: int = 10
    verbose: int = 0
    mode: str = "auto"
    min_delta: float = 0.0001
    cooldown: int = 0
    min_lr: int = 0


class RemoteMonitorConfig(object):
    """
    Callback used to stream events to a server.

    Requires the `requests` library.
    Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
    HTTP POST, with a `data` argument which is a
    JSON-encoded dictionary of event data.
    If `send_as_json=True`, the content type of the request will be
    `"application/json"`.
    Otherwise the serialized JSON will be sent within a form.

    :cvar root: String; root url of the target server. Defaults to http://localhost:9000
    :cvar path: String; path relative to `root` to which the events will be sent. Defaults to /publish/epoch/end/
    :cvar field: String; JSON field under which the data will be stored.
      The field is used only if the payload is sent within a form
      (i.e. send_as_json is set to False). Defaults to data
    :cvar headers: Dictionary; optional custom HTTP headers. Defaults to None
    :cvar send_as_json: Boolean; whether the request should be
      sent as `"application/json"`. Defaults to False"""

    root: str = "http://localhost:9000"
    path: str = "/publish/epoch/end/"
    field: str = "data"
    headers: NoneType = None
    send_as_json: bool = False


class TensorBoardConfig(object):
    """
    Enable visualizations for TensorBoard.

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
        ValueError: If histogram_freq is set and no validation data is provided.

    :cvar log_dir: the path of the directory where to save the log files to be
      parsed by TensorBoard.
    :cvar histogram_freq: frequency (in epochs) at which to compute activation and
      weight histograms for the layers of the model. If set to 0, histograms
      won't be computed. Validation data (or split) must be specified for
      histogram visualizations.
    :cvar write_graph: whether to visualize the graph in TensorBoard. The log file
      can become quite large when write_graph is set to True.
    :cvar write_images: whether to write model weights to visualize as image in
      TensorBoard.
    :cvar update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
      writes the losses and metrics to TensorBoard after each batch. The same
      applies for `'epoch'`. If using an integer, let's say `1000`, the
      callback will write the metrics and losses to TensorBoard every 1000
      batches. Note that writing too frequently to TensorBoard can slow down
      your training.
    :cvar profile_batch: Profile the batch(es) to sample compute characteristics.
      profile_batch must be a non-negative integer or a tuple of integers.
      A pair of positive integers signify a range of batches to profile.
      By default, it will profile the second batch. Set profile_batch=0
      to disable profiling.
    :cvar embeddings_freq: frequency (in epochs) at which embedding layers will be
      visualized. If set to 0, embeddings won't be visualized.
    :cvar embeddings_metadata: a dictionary which maps layer name to a file name in
      which metadata for this embedding layer is saved. See the
      [details](
        https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
      about metadata files format. In case if the same metadata file is
      used for all embedding layers, string can be passed."""

    log_dir = None
    histogram_freq = None
    write_graph = None
    write_images = None
    update_freq = None
    profile_batch = None
    embeddings_freq = None
    embeddings_metadata = None


class TerminateOnNaNConfig(object):
    """
    Callback that terminates training when a NaN loss is encountered."""


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
