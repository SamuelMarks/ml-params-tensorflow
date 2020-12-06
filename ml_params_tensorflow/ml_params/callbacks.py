""" Generated Callback config classes """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import csv
import io
import json
import os
import re
import time

import numpy as np
import six
import tensorflow as tf
from tensorflow.python.distribute import distributed_file_utils
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.saved_model import save_options as save_options_lib
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.saving import (
    checkpoint_options as checkpoint_options_lib,
)
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.tools.docs import doc_controls


class BaseLoggerConfig(object):
    """
    Callback that accumulates epoch averages of metrics.

    This callback is automatically applied to every Keras model.

    :cvar stateful_metrics: Iterable of string names of metrics that
        should *not* be averaged over an epoch.
        Metrics in this list will be logged as-is in `on_epoch_end`.
        All others will be averaged in `on_epoch_end`. Defaults to None"""

    stateful_metrics: NoneType = None

    def __call__(self):
        def __init__(self, stateful_metrics=None):
            super(BaseLogger, self).__init__()
            self.stateful_metrics = set(self.stateful_metrics or [])

        def on_epoch_begin(self, epoch, logs=None):
            self.seen = 0
            self.totals = {}

        def on_batch_end(self, batch, logs=None):
            logs = logs or {}
            batch_size = logs.get("size", 0)
            num_steps = logs.get("num_steps", 1)
            self.seen += batch_size * num_steps
            for k, v in logs.items():
                if k in self.stateful_metrics:
                    self.totals[k] = v
                elif k in self.totals:
                    self.totals[k] += v * batch_size
                else:
                    self.totals[k] = v * batch_size

        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                for k in self.params["metrics"]:
                    if k in self.totals:
                        if k in self.stateful_metrics:
                            logs[k] = self.totals[k]
                        else:
                            logs[k] = self.totals[k] / self.seen


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

    def __call__(self):
        def __init__(self, filename, separator=",", append=False):
            self.sep = self.separator
            self.filename = path_to_string(self.filename)
            self.append = self.append
            self.writer = None
            self.keys = None
            self.append_header = True
            if six.PY2:
                self.file_flags = "b"
                self._open_args = {}
            else:
                self.file_flags = ""
                self._open_args = {"newline": "\n"}
            super(CSVLogger, self).__init__()

        def on_train_begin(self, logs=None):
            if self.append:
                if file_io.file_exists(self.filename):
                    with open(self.filename, "r" + self.file_flags) as f:
                        self.append_header = not bool(len(f.readline()))
                mode = "a"
            else:
                mode = "w"
            self.csv_file = io.open(
                self.filename, mode + self.file_flags, **self._open_args
            )

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, six.string_types):
                    return k
                elif (
                    isinstance(k, collections_abc.Iterable) and not is_zero_dim_ndarray
                ):
                    return '"[%s]"' % ", ".join(map(str, k))
                else:
                    return k

            if self.keys is None:
                self.keys = sorted(logs.keys())
            if self.model.stop_training:
                logs = dict((k, logs[k]) if k in logs else (k, "NA") for k in self.keys)
            if not self.writer:

                class CustomDialect(csv.excel):
                    delimiter = self.sep

                fieldnames = ["epoch"] + self.keys
                if six.PY2:
                    fieldnames = [unicode(x) for x in fieldnames]
                self.writer = csv.DictWriter(
                    self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
                )
                if self.append_header:
                    self.writer.writeheader()
            row_dict = collections.OrderedDict({"epoch": epoch})
            row_dict.update((key, handle_value(logs[key])) for key in self.keys)
            self.writer.writerow(row_dict)
            self.csv_file.flush()

        def on_train_end(self, logs=None):
            self.csv_file.close()
            self.writer = None


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

    def __call__(self):
        def __init__(self):
            self.validation_data = None
            self.model = None
            self._chief_worker_only = None
            self._supports_tf_logs = False

        def set_params(self, params):
            self.params = params

        def set_model(self, model):
            self.model = model

        @doc_controls.for_subclass_implementers
        @generic_utils.default
        def on_batch_begin(self, batch, logs=None):
            """A backwards compatibility alias for `on_train_batch_begin`."""

        @doc_controls.for_subclass_implementers
        @generic_utils.default
        def on_batch_end(self, batch, logs=None):
            """A backwards compatibility alias for `on_train_batch_end`."""

        @doc_controls.for_subclass_implementers
        def on_epoch_begin(self, epoch, logs=None):
            """Called at the start of an epoch.

            Subclasses should override for any actions to run. This function should only
            be called during TRAIN mode.

            Args:
                epoch: Integer, index of epoch.
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """

        @doc_controls.for_subclass_implementers
        def on_epoch_end(self, epoch, logs=None):
            """Called at the end of an epoch.

            Subclasses should override for any actions to run. This function should only
            be called during TRAIN mode.

            Args:
                epoch: Integer, index of epoch.
                logs: Dict, metric results for this training epoch, and for the
                  validation epoch if validation is performed. Validation result keys
                  are prefixed with `val_`.
            """

        @doc_controls.for_subclass_implementers
        @generic_utils.default
        def on_train_batch_begin(self, batch, logs=None):
            """Called at the beginning of a training batch in `fit` methods.

            Subclasses should override for any actions to run.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict, contains the return value of `model.train_step`. Typically,
                  the values of the `Model`'s metrics are returned.  Example:
                  `{'loss': 0.2, 'accuracy': 0.7}`.
            """
            self.on_batch_begin(batch, logs=logs)

        @doc_controls.for_subclass_implementers
        @generic_utils.default
        def on_train_batch_end(self, batch, logs=None):
            """Called at the end of a training batch in `fit` methods.

            Subclasses should override for any actions to run.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict. Aggregated metric results up until this batch.
            """
            self.on_batch_end(batch, logs=logs)

        @doc_controls.for_subclass_implementers
        @generic_utils.default
        def on_test_batch_begin(self, batch, logs=None):
            """Called at the beginning of a batch in `evaluate` methods.

            Also called at the beginning of a validation batch in the `fit`
            methods, if validation data is provided.

            Subclasses should override for any actions to run.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict, contains the return value of `model.test_step`. Typically,
                  the values of the `Model`'s metrics are returned.  Example:
                  `{'loss': 0.2, 'accuracy': 0.7}`.
            """

        @doc_controls.for_subclass_implementers
        @generic_utils.default
        def on_test_batch_end(self, batch, logs=None):
            """Called at the end of a batch in `evaluate` methods.

            Also called at the end of a validation batch in the `fit`
            methods, if validation data is provided.

            Subclasses should override for any actions to run.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict. Aggregated metric results up until this batch.
            """

        @doc_controls.for_subclass_implementers
        @generic_utils.default
        def on_predict_batch_begin(self, batch, logs=None):
            """Called at the beginning of a batch in `predict` methods.

            Subclasses should override for any actions to run.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict, contains the return value of `model.predict_step`,
                  it typically returns a dict with a key 'outputs' containing
                  the model's outputs.
            """

        @doc_controls.for_subclass_implementers
        @generic_utils.default
        def on_predict_batch_end(self, batch, logs=None):
            """Called at the end of a batch in `predict` methods.

            Subclasses should override for any actions to run.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict. Aggregated metric results up until this batch.
            """

        @doc_controls.for_subclass_implementers
        def on_train_begin(self, logs=None):
            """Called at the beginning of training.

            Subclasses should override for any actions to run.

            Args:
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """

        @doc_controls.for_subclass_implementers
        def on_train_end(self, logs=None):
            """Called at the end of training.

            Subclasses should override for any actions to run.

            Args:
                logs: Dict. Currently the output of the last call to `on_epoch_end()`
                  is passed to this argument for this method but that may change in
                  the future.
            """

        @doc_controls.for_subclass_implementers
        def on_test_begin(self, logs=None):
            """Called at the beginning of evaluation or validation.

            Subclasses should override for any actions to run.

            Args:
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """

        @doc_controls.for_subclass_implementers
        def on_test_end(self, logs=None):
            """Called at the end of evaluation or validation.

            Subclasses should override for any actions to run.

            Args:
                logs: Dict. Currently the output of the last call to
                  `on_test_batch_end()` is passed to this argument for this method
                  but that may change in the future.
            """

        @doc_controls.for_subclass_implementers
        def on_predict_begin(self, logs=None):
            """Called at the beginning of prediction.

            Subclasses should override for any actions to run.

            Args:
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """

        @doc_controls.for_subclass_implementers
        def on_predict_end(self, logs=None):
            """Called at the end of prediction.

            Subclasses should override for any actions to run.

            Args:
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """

        def _implements_train_batch_hooks(self):
            """Determines if this Callback should be called for each train batch."""
            return (
                not generic_utils.is_default(self.on_batch_begin)
                or not generic_utils.is_default(self.on_batch_end)
                or not generic_utils.is_default(self.on_train_batch_begin)
                or not generic_utils.is_default(self.on_train_batch_end)
            )

        def _implements_test_batch_hooks(self):
            """Determines if this Callback should be called for each test batch."""
            return not generic_utils.is_default(
                self.on_test_batch_begin
            ) or not generic_utils.is_default(self.on_test_batch_end)

        def _implements_predict_batch_hooks(self):
            """Determines if this Callback should be called for each predict batch."""
            return not generic_utils.is_default(
                self.on_predict_batch_begin
            ) or not generic_utils.is_default(self.on_predict_batch_end)


class CallbackListConfig(object):
    """
    Container abstracting a list of callbacks."""

    def __call__(self):
        def __init__(
            self,
            callbacks=None,
            add_history=False,
            add_progbar=False,
            model=None,
            **params
        ):
            """Container for `Callback` instances.

            This object wraps a list of `Callback` instances, making it possible
            to call them all at once via a single endpoint
            (e.g. `callback_list.on_epoch_end(...)`).

            Args:
              callbacks: List of `Callback` instances.
              add_history: Whether a `History` callback should be added, if one does not
                already exist in the `callbacks` list.
              add_progbar: Whether a `ProgbarLogger` callback should be added, if one
                does not already exist in the `callbacks` list.
              model: The `Model` these callbacks are used with.
              **params: If provided, parameters will be passed to each `Callback` via
                `Callback.set_params`.
            """
            self.callbacks = nest.flatten(callbacks) if callbacks else []
            self._add_default_callbacks(add_history, add_progbar)
            if model:
                self.set_model(model)
            if params:
                self.set_params(params)
            self._should_call_train_batch_hooks = any(
                cb._implements_train_batch_hooks() for cb in self.callbacks
            )
            self._should_call_test_batch_hooks = any(
                cb._implements_test_batch_hooks() for cb in self.callbacks
            )
            self._should_call_predict_batch_hooks = any(
                cb._implements_predict_batch_hooks() for cb in self.callbacks
            )
            self._timing = {}
            self._check_timing = False
            self._batch_start_time = None

        def _add_default_callbacks(self, add_history, add_progbar):
            """Adds `Callback`s that are always present."""
            self._progbar = None
            self._history = None
            for cb in self.callbacks:
                if isinstance(cb, ProgbarLogger):
                    self._progbar = cb
                elif isinstance(cb, History):
                    self._history = cb
            if self._progbar is None and add_progbar:
                self._progbar = ProgbarLogger(count_mode="steps")
                self.callbacks.append(self._progbar)
            if self._history is None and add_history:
                self._history = History()
                self.callbacks.append(self._history)

        def append(self, callback):
            self.callbacks.append(callback)

        def set_params(self, params):
            self.params = params
            for callback in self.callbacks:
                callback.set_params(params)

        def set_model(self, model):
            self.model = model
            if self._history:
                model.history = self._history
            for callback in self.callbacks:
                callback.set_model(model)

        def _call_batch_hook(self, mode, hook, batch, logs=None):
            """Helper function for all batch_{begin | end} methods."""
            if not self.callbacks:
                return
            if hook == "begin":
                self._call_batch_begin_hook(mode, batch, logs)
            elif hook == "end":
                self._call_batch_end_hook(mode, batch, logs)
            else:
                raise ValueError("Unrecognized hook: {}".format(hook))

        def _call_batch_begin_hook(self, mode, batch, logs):
            """Helper function for `on_*_batch_begin` methods."""
            hook_name = "on_{mode}_batch_begin".format(mode=mode)
            self._check_timing = batch == 1 and hook_name not in self._timing
            self._call_batch_hook_helper(hook_name, batch, logs)
            if self._check_timing:
                self._batch_start_time = time.time()

        def _call_batch_end_hook(self, mode, batch, logs):
            """Helper function for `on_*_batch_end` methods."""
            hook_name = "on_{mode}_batch_end".format(mode=mode)
            if self._check_timing:
                batch_time = time.time() - self._batch_start_time
            self._call_batch_hook_helper(hook_name, batch, logs)
            if self._check_timing:
                end_hook_name = hook_name
                begin_hook_name = "on_{mode}_batch_begin".format(mode=mode)
                threshold_time = 1.5 * batch_time
                warning_msg = "Callbacks method `{hook}` is slow compared to the batch time (batch time: {batch_time:.4f}s vs `{hook}` time: {cbk_time:.4f}s). Check your callbacks."
                if self._timing[begin_hook_name] > threshold_time:
                    logging.warning(
                        warning_msg.format(
                            hook=begin_hook_name,
                            batch_time=batch_time,
                            cbk_time=self._timing[begin_hook_name],
                        )
                    )
                if self._timing[end_hook_name] > threshold_time:
                    logging.warning(
                        warning_msg.format(
                            hook=end_hook_name,
                            batch_time=batch_time,
                            cbk_time=self._timing[end_hook_name],
                        )
                    )
                self._check_timing = False
                self._batch_start_time = None

        def _call_batch_hook_helper(self, hook_name, batch, logs):
            """Helper function for `on_*_batch_*` methods."""
            logs = logs or {}
            numpy_logs = None
            if self._check_timing:
                start_time = time.time()
            for callback in self.callbacks:
                hook = getattr(callback, hook_name)
                if getattr(callback, "_supports_tf_logs", False):
                    hook(batch, logs)
                else:
                    if numpy_logs is None:
                        numpy_logs = tf_utils.to_numpy_or_python_type(logs)
                    hook(batch, numpy_logs)
            if self._check_timing:
                self._timing[hook_name] = time.time() - start_time

        def _call_begin_hook(self, mode):
            """Helper function for on_{train|test|predict}_begin methods."""
            if mode == ModeKeys.TRAIN:
                self.on_train_begin()
            elif mode == ModeKeys.TEST:
                self.on_test_begin()
            else:
                self.on_predict_begin()

        def _call_end_hook(self, mode):
            """Helper function for on_{train|test|predict}_end methods."""
            if mode == ModeKeys.TRAIN:
                self.on_train_end()
            elif mode == ModeKeys.TEST:
                self.on_test_end()
            else:
                self.on_predict_end()

        def on_batch_begin(self, batch, logs=None):
            if self._should_call_train_batch_hooks:
                self._call_batch_hook(ModeKeys.TRAIN, "begin", batch, logs=logs)

        def on_batch_end(self, batch, logs=None):
            if self._should_call_train_batch_hooks:
                self._call_batch_hook(ModeKeys.TRAIN, "end", batch, logs=logs)

        def on_epoch_begin(self, epoch, logs=None):
            """Calls the `on_epoch_begin` methods of its callbacks.

            This function should only be called during TRAIN mode.

            Args:
                epoch: Integer, index of epoch.
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """
            logs = logs or {}
            numpy_logs = None
            for callback in self.callbacks:
                if getattr(callback, "_supports_tf_logs", False):
                    callback.on_epoch_begin(epoch, logs)
                else:
                    if numpy_logs is None:
                        numpy_logs = tf_utils.to_numpy_or_python_type(logs)
                    callback.on_epoch_begin(epoch, numpy_logs)

        def on_epoch_end(self, epoch, logs=None):
            """Calls the `on_epoch_end` methods of its callbacks.

            This function should only be called during TRAIN mode.

            Args:
                epoch: Integer, index of epoch.
                logs: Dict, metric results for this training epoch, and for the
                  validation epoch if validation is performed. Validation result keys
                  are prefixed with `val_`.
            """
            logs = logs or {}
            numpy_logs = None
            for callback in self.callbacks:
                if getattr(callback, "_supports_tf_logs", False):
                    callback.on_epoch_end(epoch, logs)
                else:
                    if numpy_logs is None:
                        numpy_logs = tf_utils.to_numpy_or_python_type(logs)
                    callback.on_epoch_end(epoch, numpy_logs)

        def on_train_batch_begin(self, batch, logs=None):
            """Calls the `on_train_batch_begin` methods of its callbacks.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict, contains the return value of `model.train_step`. Typically,
                  the values of the `Model`'s metrics are returned.  Example:
                  `{'loss': 0.2, 'accuracy': 0.7}`.
            """
            if self._should_call_train_batch_hooks:
                self._call_batch_hook(ModeKeys.TRAIN, "begin", batch, logs=logs)

        def on_train_batch_end(self, batch, logs=None):
            """Calls the `on_train_batch_end` methods of its callbacks.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict. Aggregated metric results up until this batch.
            """
            if self._should_call_train_batch_hooks:
                self._call_batch_hook(ModeKeys.TRAIN, "end", batch, logs=logs)

        def on_test_batch_begin(self, batch, logs=None):
            """Calls the `on_test_batch_begin` methods of its callbacks.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict, contains the return value of `model.test_step`. Typically,
                  the values of the `Model`'s metrics are returned.  Example:
                  `{'loss': 0.2, 'accuracy': 0.7}`.
            """
            if self._should_call_test_batch_hooks:
                self._call_batch_hook(ModeKeys.TEST, "begin", batch, logs=logs)

        def on_test_batch_end(self, batch, logs=None):
            """Calls the `on_test_batch_end` methods of its callbacks.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict. Aggregated metric results up until this batch.
            """
            if self._should_call_test_batch_hooks:
                self._call_batch_hook(ModeKeys.TEST, "end", batch, logs=logs)

        def on_predict_batch_begin(self, batch, logs=None):
            """Calls the `on_predict_batch_begin` methods of its callbacks.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict, contains the return value of `model.predict_step`,
                  it typically returns a dict with a key 'outputs' containing
                  the model's outputs.
            """
            if self._should_call_predict_batch_hooks:
                self._call_batch_hook(ModeKeys.PREDICT, "begin", batch, logs=logs)

        def on_predict_batch_end(self, batch, logs=None):
            """Calls the `on_predict_batch_end` methods of its callbacks.

            Args:
                batch: Integer, index of batch within the current epoch.
                logs: Dict. Aggregated metric results up until this batch.
            """
            if self._should_call_predict_batch_hooks:
                self._call_batch_hook(ModeKeys.PREDICT, "end", batch, logs=logs)

        def on_train_begin(self, logs=None):
            """Calls the `on_train_begin` methods of its callbacks.

            Args:
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """
            logs = logs or {}
            numpy_logs = None
            for callback in self.callbacks:
                if getattr(callback, "_supports_tf_logs", False):
                    callback.on_train_begin(logs)
                else:
                    if numpy_logs is None:
                        numpy_logs = tf_utils.to_numpy_or_python_type(logs)
                    callback.on_train_begin(numpy_logs)

        def on_train_end(self, logs=None):
            """Calls the `on_train_end` methods of its callbacks.

            Args:
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """
            logs = logs or {}
            numpy_logs = None
            for callback in self.callbacks:
                if getattr(callback, "_supports_tf_logs", False):
                    callback.on_train_end(logs)
                else:
                    if numpy_logs is None:
                        numpy_logs = tf_utils.to_numpy_or_python_type(logs)
                    callback.on_train_end(numpy_logs)

        def on_test_begin(self, logs=None):
            """Calls the `on_test_begin` methods of its callbacks.

            Args:
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """
            logs = logs or {}
            numpy_logs = None
            for callback in self.callbacks:
                if getattr(callback, "_supports_tf_logs", False):
                    callback.on_test_begin(logs)
                else:
                    if numpy_logs is None:
                        numpy_logs = tf_utils.to_numpy_or_python_type(logs)
                    callback.on_test_begin(numpy_logs)

        def on_test_end(self, logs=None):
            """Calls the `on_test_end` methods of its callbacks.

            Args:
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """
            logs = logs or {}
            numpy_logs = None
            for callback in self.callbacks:
                if getattr(callback, "_supports_tf_logs", False):
                    callback.on_test_end(logs)
                else:
                    if numpy_logs is None:
                        numpy_logs = tf_utils.to_numpy_or_python_type(logs)
                    callback.on_test_end(numpy_logs)

        def on_predict_begin(self, logs=None):
            """Calls the 'on_predict_begin` methods of its callbacks.

            Args:
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """
            logs = logs or {}
            numpy_logs = None
            for callback in self.callbacks:
                if getattr(callback, "_supports_tf_logs", False):
                    callback.on_predict_begin(logs)
                else:
                    if numpy_logs is None:
                        numpy_logs = tf_utils.to_numpy_or_python_type(logs)
                    callback.on_predict_begin(numpy_logs)

        def on_predict_end(self, logs=None):
            """Calls the `on_predict_end` methods of its callbacks.

            Args:
                logs: Dict. Currently no data is passed to this argument for this method
                  but that may change in the future.
            """
            logs = logs or {}
            numpy_logs = None
            for callback in self.callbacks:
                if getattr(callback, "_supports_tf_logs", False):
                    callback.on_predict_end(logs)
                else:
                    if numpy_logs is None:
                        numpy_logs = tf_utils.to_numpy_or_python_type(logs)
                    callback.on_predict_end(numpy_logs)

        def __iter__(self):
            return iter(self.callbacks)


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

    def __call__(self):
        def __init__(
            self,
            monitor="val_loss",
            min_delta=0,
            patience=0,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        ):
            super(EarlyStopping, self).__init__()
            self.monitor = self.monitor
            self.patience = self.patience
            self.verbose = self.verbose
            self.baseline = self.baseline
            self.min_delta = abs(self.min_delta)
            self.wait = 0
            self.stopped_epoch = 0
            self.restore_best_weights = self.restore_best_weights
            self.best_weights = None
            if self.mode not in ["auto", "min", "max"]:
                logging.warning(
                    "EarlyStopping mode %s is unknown, fallback to auto mode.",
                    self.mode,
                )
                self.mode = "auto"
            if self.mode == "min":
                self.monitor_op = np.less
            elif self.mode == "max":
                self.monitor_op = np.greater
            elif "acc" in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
            if self.monitor_op == np.greater:
                self.min_delta *= 1
            else:
                self.min_delta *= -1

        def on_train_begin(self, logs=None):
            self.wait = 0
            self.stopped_epoch = 0
            if self.baseline is not None:
                self.best = self.baseline
            else:
                self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            current = self.get_monitor_value(logs)
            if current is None:
                return
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.restore_best_weights:
                        if self.verbose > 0:
                            print(
                                "Restoring model weights from the end of the best epoch."
                            )
                        self.model.set_weights(self.best_weights)

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0 and self.verbose > 0:
                print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

        def get_monitor_value(self, logs):
            logs = logs or {}
            monitor_value = logs.get(self.monitor)
            if monitor_value is None:
                logging.warning(
                    "Early stopping conditioned on metric `%s` which is not available. Available metrics are: %s",
                    self.monitor,
                    ",".join(list(logs.keys())),
                )
            return monitor_value


class HistoryConfig(object):
    """
    Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models."""

    def __call__(self):
        def __init__(self):
            super(History, self).__init__()
            self.history = {}

        def on_train_begin(self, logs=None):
            self.epoch = []

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
            self.model.history = self


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

    def __call__(self):
        def __init__(
            self,
            on_epoch_begin=None,
            on_epoch_end=None,
            on_batch_begin=None,
            on_batch_end=None,
            on_train_begin=None,
            on_train_end=None,
            **kwargs
        ):
            super(LambdaCallback, self).__init__()
            self.__dict__.update(kwargs)
            if self.on_epoch_begin is not None:
                self.on_epoch_begin = self.on_epoch_begin
            else:
                self.on_epoch_begin = lambda epoch, logs: None
            if self.on_epoch_end is not None:
                self.on_epoch_end = self.on_epoch_end
            else:
                self.on_epoch_end = lambda epoch, logs: None
            if self.on_batch_begin is not None:
                self.on_batch_begin = self.on_batch_begin
            else:
                self.on_batch_begin = lambda batch, logs: None
            if self.on_batch_end is not None:
                self.on_batch_end = self.on_batch_end
            else:
                self.on_batch_end = lambda batch, logs: None
            if self.on_train_begin is not None:
                self.on_train_begin = self.on_train_begin
            else:
                self.on_train_begin = lambda logs: None
            if self.on_train_end is not None:
                self.on_train_end = self.on_train_end
            else:
                self.on_train_end = lambda logs: None


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

    def __call__(self):
        def __init__(self, schedule, verbose=0):
            super(LearningRateScheduler, self).__init__()
            self.schedule = self.schedule
            self.verbose = self.verbose

        def on_epoch_begin(self, epoch, logs=None):
            if not hasattr(self.model.optimizer, "lr"):
                raise ValueError('Optimizer must have a "lr" attribute.')
            try:
                lr = float(K.get_value(self.model.optimizer.lr))
                lr = self.schedule(epoch, lr)
            except TypeError:
                lr = self.schedule(epoch)
            if not isinstance(lr, (ops.Tensor, float, np.float32, np.float64)):
                raise ValueError(
                    'The output of the "schedule" function should be float.'
                )
            if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
                raise ValueError("The dtype of Tensor should be float")
            K.set_value(self.model.optimizer.lr, K.get_value(lr))
            if self.verbose > 0:
                print(
                    "\nEpoch %05d: LearningRateScheduler reducing learning rate to %s."
                    % (epoch + 1, lr)
                )

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs["lr"] = K.get_value(self.model.optimizer.lr)


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

    def __call__(self):
        def __init__(
            self,
            filepath,
            monitor="val_loss",
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            options=None,
            **kwargs
        ):
            super(ModelCheckpoint, self).__init__()
            self._supports_tf_logs = True
            self.monitor = self.monitor
            self.verbose = self.verbose
            self.filepath = path_to_string(self.filepath)
            self.save_best_only = self.save_best_only
            self.save_weights_only = self.save_weights_only
            self.save_freq = self.save_freq
            self.epochs_since_last_save = 0
            self._batches_seen_since_last_saving = 0
            self._last_batch_seen = 0
            if self.save_weights_only:
                if self.options is None or isinstance(
                    self.options, checkpoint_options_lib.CheckpointOptions
                ):
                    self._options = (
                        self.options or checkpoint_options_lib.CheckpointOptions()
                    )
                else:
                    raise TypeError(
                        "If save_weights_only is True, then `options` must beeither None or a tf.train.CheckpointOptions"
                    )
            elif self.options is None or isinstance(
                self.options, save_options_lib.SaveOptions
            ):
                self._options = self.options or save_options_lib.SaveOptions()
            else:
                raise TypeError(
                    "If save_weights_only is False, then `options` must beeither None or a tf.saved_model.SaveOptions"
                )
            if "load_weights_on_restart" in self.kwargs:
                self.load_weights_on_restart = self.kwargs["load_weights_on_restart"]
                logging.warning(
                    "`load_weights_on_restart` argument is deprecated. Please use `model.load_weights()` for loading weights before the start of `model.fit()`."
                )
            else:
                self.load_weights_on_restart = False
            if "period" in self.kwargs:
                self.period = self.kwargs["period"]
                logging.warning(
                    "`period` argument is deprecated. Please use `save_freq` to specify the frequency in number of batches seen."
                )
            else:
                self.period = 1
            if self.mode not in ["auto", "min", "max"]:
                logging.warning(
                    "ModelCheckpoint mode %s is unknown, fallback to auto mode.",
                    self.mode,
                )
                self.mode = "auto"
            if self.mode == "min":
                self.monitor_op = np.less
                self.best = np.Inf
            elif self.mode == "max":
                self.monitor_op = np.greater
                self.best = -np.Inf
            elif "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
            if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
                raise ValueError("Unrecognized save_freq: {}".format(self.save_freq))
            self._chief_worker_only = False

        def set_model(self, model):
            self.model = model
            if (
                not self.save_weights_only
                and not model._is_graph_network
                and model.__class__.__name__ != "Sequential"
            ):
                self.save_weights_only = True

        def on_train_begin(self, logs=None):
            if self.load_weights_on_restart:
                filepath_to_load = (
                    self._get_most_recently_modified_file_matching_pattern(
                        self.filepath
                    )
                )
                if filepath_to_load is not None and self._checkpoint_exists(
                    filepath_to_load
                ):
                    try:
                        self.model.load_weights(filepath_to_load)
                    except (IOError, ValueError) as e:
                        raise ValueError(
                            "Error loading file from {}. Reason: {}".format(
                                filepath_to_load, e
                            )
                        )

        def on_train_batch_end(self, batch, logs=None):
            if self._should_save_on_batch(batch):
                self._save_model(epoch=self._current_epoch, logs=logs)

        def on_epoch_begin(self, epoch, logs=None):
            self._current_epoch = epoch

        def on_epoch_end(self, epoch, logs=None):
            self.epochs_since_last_save += 1
            if self.save_freq == "epoch":
                self._save_model(epoch=epoch, logs=logs)

        def _should_save_on_batch(self, batch):
            """Handles batch-level saving logic, supports steps_per_execution."""
            if self.save_freq == "epoch":
                return False
            if batch <= self._last_batch_seen:
                add_batches = batch + 1
            else:
                add_batches = batch - self._last_batch_seen
            self._batches_seen_since_last_saving += add_batches
            self._last_batch_seen = batch
            if self._batches_seen_since_last_saving >= self.save_freq:
                self._batches_seen_since_last_saving = 0
                return True
            return False

        def _save_model(self, epoch, logs):
            """Saves the model.

            Args:
                epoch: the epoch this iteration is in.
                logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
            """
            logs = logs or {}
            if (
                isinstance(self.save_freq, int)
                or self.epochs_since_last_save >= self.period
            ):
                logs = tf_utils.to_numpy_or_python_type(logs)
                self.epochs_since_last_save = 0
                self.filepath = self._get_file_path(epoch, logs)
                try:
                    if self.save_best_only:
                        current = logs.get(self.monitor)
                        if current is None:
                            logging.warning(
                                "Can save best model only with %s available, skipping.",
                                self.monitor,
                            )
                        elif self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print(
                                    """
Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"""
                                    % (
                                        epoch + 1,
                                        self.monitor,
                                        self.best,
                                        current,
                                        self.filepath,
                                    )
                                )
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(
                                    self.filepath, overwrite=True, options=self._options
                                )
                            else:
                                self.model.save(
                                    self.filepath, overwrite=True, options=self._options
                                )
                        elif self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s did not improve from %0.5f"
                                % (epoch + 1, self.monitor, self.best)
                            )
                    else:
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: saving model to %s"
                                % (epoch + 1, self.filepath)
                            )
                        if self.save_weights_only:
                            self.model.save_weights(
                                self.filepath, overwrite=True, options=self._options
                            )
                        else:
                            self.model.save(
                                self.filepath, overwrite=True, options=self._options
                            )
                    self._maybe_remove_file()
                except IOError as e:
                    if "is a directory" in six.ensure_str(e.args[0]).lower():
                        raise IOError(
                            "Please specify a non-directory filepath for ModelCheckpoint. Filepath used is an existing directory: {}".format(
                                self.filepath
                            )
                        )

        def _get_file_path(self, epoch, logs):
            """Returns the file path for checkpoint."""
            try:
                file_path = self.filepath.format(epoch=epoch + 1, **logs)
            except KeyError as e:
                raise KeyError(
                    'Failed to format this callback filepath: "{}". Reason: {}'.format(
                        self.filepath, e
                    )
                )
            self._write_filepath = distributed_file_utils.write_filepath(
                file_path, self.model.distribute_strategy
            )
            return self._write_filepath

        def _maybe_remove_file(self):
            distributed_file_utils.remove_temp_dir_with_filepath(
                self._write_filepath, self.model.distribute_strategy
            )

        def _checkpoint_exists(self, filepath):
            """Returns whether the checkpoint `filepath` refers to exists."""
            if self.filepath.endswith(".h5"):
                return file_io.file_exists(self.filepath)
            tf_saved_model_exists = file_io.file_exists(self.filepath)
            tf_weights_only_checkpoint_exists = file_io.file_exists(
                self.filepath + ".index"
            )
            return tf_saved_model_exists or tf_weights_only_checkpoint_exists

        def _get_most_recently_modified_file_matching_pattern(self, pattern):
            """Returns the most recently modified filepath matching pattern.

            Pattern may contain python formatting placeholder. If
            `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
            check for most recently modified one that matches the pattern.

            In the rare case where there are more than one pattern-matching file having
            the same modified time that is most recent among all, return the filepath
            that is largest (by `>` operator, lexicographically using the numeric
            equivalents). This provides a tie-breaker when multiple files are most
            recent. Note that a larger `filepath` can sometimes indicate a later time of
            modification (for instance, when epoch/batch is used as formatting option),
            but not necessarily (when accuracy or loss is used). The tie-breaker is
            put in the logic as best effort to return the most recent, and to avoid
            undeterministic result.

            Modified time of a file is obtained with `os.path.getmtime()`.

            This utility function is best demonstrated via an example:

            ```python
            file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
            test_dir = self.get_temp_dir()
            path_pattern = os.path.join(test_dir, file_pattern)
            file_paths = [
                os.path.join(test_dir, file_name) for file_name in
                ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
            ]
            for file_path in file_paths:
              # Write something to each of the files
            self.assertEqual(
                _get_most_recently_modified_file_matching_pattern(path_pattern),
                file_paths[-1])
            ```

            Args:
                pattern: The file pattern that may optionally contain python placeholder
                    such as `{epoch:02d}`.

            Returns:
                The most recently modified file's full filepath matching `pattern`. If
                `pattern` does not contain any placeholder, this returns the filepath
                that
                exactly matches `pattern`. Returns `None` if no match is found.
            """
            dir_name = os.path.dirname(pattern)
            base_name = os.path.basename(pattern)
            base_name_regex = "^" + re.sub("{.*}", ".*", base_name) + "$"
            latest_tf_checkpoint = checkpoint_management.latest_checkpoint(dir_name)
            if latest_tf_checkpoint is not None and re.match(
                base_name_regex, os.path.basename(latest_tf_checkpoint)
            ):
                return latest_tf_checkpoint
            latest_mod_time = 0
            file_path_with_latest_mod_time = None
            n_file_with_latest_mod_time = 0
            file_path_with_largest_file_name = None
            if file_io.file_exists(dir_name):
                for file_name in os.listdir(dir_name):
                    if re.match(base_name_regex, file_name):
                        file_path = os.path.join(dir_name, file_name)
                        mod_time = os.path.getmtime(file_path)
                        if (
                            file_path_with_largest_file_name is None
                            or file_path > file_path_with_largest_file_name
                        ):
                            file_path_with_largest_file_name = file_path
                        if mod_time > latest_mod_time:
                            latest_mod_time = mod_time
                            file_path_with_latest_mod_time = file_path
                            n_file_with_latest_mod_time = 1
                        elif mod_time == latest_mod_time:
                            n_file_with_latest_mod_time += 1
            if n_file_with_latest_mod_time == 1:
                return file_path_with_latest_mod_time
            else:
                return file_path_with_largest_file_name


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

    def __call__(self):
        def __init__(self, count_mode="samples", stateful_metrics=None):
            super(ProgbarLogger, self).__init__()
            self._supports_tf_logs = True
            if self.count_mode == "samples":
                self.use_steps = False
            elif self.count_mode == "steps":
                self.use_steps = True
            else:
                raise ValueError("Unknown `count_mode`: " + str(self.count_mode))
            self.stateful_metrics = (
                set(self.stateful_metrics) if self.stateful_metrics else None
            )
            self.seen = 0
            self.progbar = None
            self.target = None
            self.verbose = 1
            self.epochs = 1
            self._called_in_fit = False

        def set_params(self, params):
            self.verbose = params["verbose"]
            self.epochs = params["epochs"]
            if self.use_steps and "steps" in params:
                self.target = params["steps"]
            elif not self.use_steps and "samples" in params:
                self.target = params["samples"]
            else:
                self.target = None

        def on_train_begin(self, logs=None):
            self._called_in_fit = True

        def on_test_begin(self, logs=None):
            if not self._called_in_fit:
                self._reset_progbar()

        def on_predict_begin(self, logs=None):
            self._reset_progbar()

        def on_epoch_begin(self, epoch, logs=None):
            self._reset_progbar()
            if self.verbose and self.epochs > 1:
                print("Epoch %d/%d" % (epoch + 1, self.epochs))

        def on_train_batch_end(self, batch, logs=None):
            self._batch_update_progbar(batch, logs)

        def on_test_batch_end(self, batch, logs=None):
            if not self._called_in_fit:
                self._batch_update_progbar(batch, logs)

        def on_predict_batch_end(self, batch, logs=None):
            self._batch_update_progbar(batch, None)

        def on_epoch_end(self, epoch, logs=None):
            self._finalize_progbar(logs)

        def on_test_end(self, logs=None):
            if not self._called_in_fit:
                self._finalize_progbar(logs)

        def on_predict_end(self, logs=None):
            self._finalize_progbar(logs)

        def _reset_progbar(self):
            self.seen = 0
            self.progbar = None

        def _maybe_init_progbar(self):
            if self.stateful_metrics is None:
                if self.model:
                    self.stateful_metrics = set(m.name for m in self.model.metrics)
                else:
                    self.stateful_metrics = set()
            if self.progbar is None:
                self.progbar = Progbar(
                    target=self.target,
                    verbose=self.verbose,
                    stateful_metrics=self.stateful_metrics,
                    unit_name="step" if self.use_steps else "sample",
                )

        def _batch_update_progbar(self, batch, logs=None):
            """Updates the progbar."""
            logs = logs or {}
            self._maybe_init_progbar()
            if self.use_steps:
                self.seen = batch + 1
            else:
                logs = copy.copy(logs)
                batch_size = logs.pop("size", 0)
                num_steps = logs.pop("num_steps", 1)
                logs.pop("batch", None)
                add_seen = num_steps * batch_size
                self.seen += add_seen
            if self.verbose == 1:
                logs = tf_utils.to_numpy_or_python_type(logs)
                self.progbar.update(self.seen, list(logs.items()), finalize=False)

        def _finalize_progbar(self, logs):
            logs = logs or {}
            self._maybe_init_progbar()
            if self.target is None:
                self.target = self.seen
                self.progbar.target = self.seen
            logs = tf_utils.to_numpy_or_python_type(logs)
            self.progbar.update(self.seen, list(logs.items()), finalize=True)


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

    def __call__(self):
        def __init__(
            self,
            monitor="val_loss",
            factor=0.1,
            patience=10,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
            **kwargs
        ):
            super(ReduceLROnPlateau, self).__init__()
            self.monitor = self.monitor
            if self.factor >= 1.0:
                raise ValueError("ReduceLROnPlateau does not support a factor >= 1.0.")
            if "epsilon" in kwargs:
                self.min_delta = kwargs.pop("epsilon")
                logging.warning(
                    "`epsilon` argument is deprecated and will be removed, use `min_delta` instead."
                )
            self.factor = self.factor
            self.min_lr = self.min_lr
            self.min_delta = self.min_delta
            self.patience = self.patience
            self.verbose = self.verbose
            self.cooldown = self.cooldown
            self.cooldown_counter = 0
            self.wait = 0
            self.best = 0
            self.mode = self.mode
            self.monitor_op = None
            self._reset()

        def _reset(self):
            """Resets wait counter and cooldown counter."""
            if self.mode not in ["auto", "min", "max"]:
                logging.warning(
                    "Learning Rate Plateau Reducing mode %s is unknown, fallback to auto mode.",
                    self.mode,
                )
                self.mode = "auto"
            if self.mode == "min" or self.mode == "auto" and "acc" not in self.monitor:
                self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
                self.best = np.Inf
            else:
                self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
                self.best = -np.Inf
            self.cooldown_counter = 0
            self.wait = 0

        def on_train_begin(self, logs=None):
            self._reset()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs["lr"] = K.get_value(self.model.optimizer.lr)
            current = logs.get(self.monitor)
            if current is None:
                logging.warning(
                    "Reduce LR on plateau conditioned on metric `%s` which is not available. Available metrics are: %s",
                    self.monitor,
                    ",".join(list(logs.keys())),
                )
            else:
                if self.in_cooldown():
                    self.cooldown_counter -= 1
                    self.wait = 0
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0
                elif not self.in_cooldown():
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = float(K.get_value(self.model.optimizer.lr))
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            K.set_value(self.model.optimizer.lr, new_lr)
                            if self.verbose > 0:
                                print(
                                    """
Epoch %05d: ReduceLROnPlateau reducing learning rate to %s."""
                                    % (epoch + 1, new_lr)
                                )
                            self.cooldown_counter = self.cooldown
                            self.wait = 0

        def in_cooldown(self):
            return self.cooldown_counter > 0


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

    def __call__(self):
        def __init__(
            self,
            root="http://localhost:9000",
            path="/publish/epoch/end/",
            field="data",
            headers=None,
            send_as_json=False,
        ):
            super(RemoteMonitor, self).__init__()
            self.root = self.root
            self.path = self.path
            self.field = self.field
            self.headers = self.headers
            self.send_as_json = self.send_as_json

        def on_epoch_end(self, epoch, logs=None):
            if requests is None:
                raise ImportError("RemoteMonitor requires the `requests` library.")
            logs = logs or {}
            send = {}
            send["epoch"] = epoch
            for k, v in logs.items():
                if isinstance(v, (np.ndarray, np.generic)):
                    send[k] = v.item()
                else:
                    send[k] = v
            try:
                if self.send_as_json:
                    requests.post(
                        self.root + self.path, json=send, headers=self.headers
                    )
                else:
                    requests.post(
                        self.root + self.path,
                        {self.field: json.dumps(send)},
                        headers=self.headers,
                    )
            except requests.exceptions.RequestException:
                logging.warning(
                    "Warning: could not reach RemoteMonitor root server at "
                    + str(self.root)
                )


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

    def __call__(self):
        def __init__(
            self,
            log_dir="logs",
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
            profile_batch=2,
            embeddings_freq=0,
            embeddings_metadata=None,
            **kwargs
        ):
            super(TensorBoard, self).__init__()
            self._supports_tf_logs = True
            self._validate_kwargs(kwargs)
            self.log_dir = path_to_string(self.log_dir)
            self.histogram_freq = self.histogram_freq
            self.write_graph = self.write_graph
            self.write_images = self.write_images
            self.update_freq = 1 if self.update_freq == "batch" else self.update_freq
            self.embeddings_freq = self.embeddings_freq
            self.embeddings_metadata = self.embeddings_metadata
            self._init_profile_batch(self.profile_batch)
            self._epoch = 0
            self._global_train_batch = 0
            self._writers = {}
            self._prev_summary_state = []

        def _validate_kwargs(self, kwargs):
            """Handle arguments were supported in V1."""
            if kwargs.get("write_grads", False):
                logging.warning(
                    "`write_grads` will be ignored in TensorFlow 2.0 for the `TensorBoard` Callback."
                )
            if kwargs.get("batch_size", False):
                logging.warning(
                    "`batch_size` is no longer needed in the `TensorBoard` Callback and will be ignored in TensorFlow 2.0."
                )
            if kwargs.get("embeddings_layer_names", False):
                logging.warning(
                    "`embeddings_layer_names` is not supported in TensorFlow 2.0. Instead, all `Embedding` layers will be visualized."
                )
            if kwargs.get("embeddings_data", False):
                logging.warning(
                    "`embeddings_data` is not supported in TensorFlow 2.0. Instead, all `Embedding` variables will be visualized."
                )
            unrecognized_kwargs = set(kwargs.keys()) - {
                "write_grads",
                "embeddings_layer_names",
                "embeddings_data",
                "batch_size",
            }
            if unrecognized_kwargs:
                raise ValueError(
                    "Unrecognized arguments in `TensorBoard` Callback: "
                    + str(unrecognized_kwargs)
                )

        def set_model(self, model):
            """Sets Keras model and writes graph if specified."""
            self.model = model
            self._log_write_dir = self._get_log_write_dir()
            self._train_dir = os.path.join(self._log_write_dir, "train")
            self._train_step = self.model._train_counter
            self._val_dir = os.path.join(self._log_write_dir, "validation")
            self._val_step = self.model._test_counter
            self._writers = {}
            if self.write_graph:
                self._write_keras_model_graph()
            if self.embeddings_freq:
                self._configure_embeddings()

        @property
        def _train_writer(self):
            if "train" not in self._writers:
                self._writers["train"] = summary_ops_v2.create_file_writer_v2(
                    self._train_dir
                )
            return self._writers["train"]

        @property
        def _val_writer(self):
            if "val" not in self._writers:
                self._writers["val"] = summary_ops_v2.create_file_writer_v2(
                    self._val_dir
                )
            return self._writers["val"]

        def _get_log_write_dir(self):
            """For multi-worker, only chief should write, others write to '/tmp'."""
            return distributed_file_utils.write_dirpath(
                self.log_dir, self.model.distribute_strategy
            )

        def _delete_tmp_write_dir(self):
            """Deletes tmp write directories for multi-worker."""
            distributed_file_utils.remove_temp_dirpath(
                self.log_dir, self.model.distribute_strategy
            )

        def _write_keras_model_graph(self):
            """Writes Keras graph networks to TensorBoard."""
            with self._train_writer.as_default():
                with summary_ops_v2.always_record_summaries():
                    if not self.model.run_eagerly:
                        summary_ops_v2.graph(K.get_graph(), step=0)
                    summary_writable = (
                        self.model._is_graph_network
                        or self.model.__class__.__name__ == "Sequential"
                    )
                    if summary_writable:
                        summary_ops_v2.keras_model("keras", self.model, step=0)

        def _configure_embeddings(self):
            """Configure the Projector for embeddings."""
            from google.protobuf import text_format
            from tensorflow.python.keras.layers import embeddings
            from tensorflow.python.keras.protobuf import projector_config_pb2

            config = projector_config_pb2.ProjectorConfig()
            for layer in self.model.layers:
                if isinstance(layer, embeddings.Embedding):
                    embedding = config.embeddings.add()
                    name = "layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE"
                    embedding.tensor_name = name
                    if self.embeddings_metadata is not None:
                        if isinstance(self.embeddings_metadata, str):
                            embedding.metadata_path = self.embeddings_metadata
                        elif layer.name in self.embeddings_metadata.keys():
                            embedding.metadata_path = self.embeddings_metadata.pop(
                                layer.name
                            )
            if self.embeddings_metadata and not isinstance(
                self.embeddings_metadata, str
            ):
                raise ValueError(
                    "Unrecognized `Embedding` layer names passed to `keras.callbacks.TensorBoard` `embeddings_metadata` argument: "
                    + str(self.embeddings_metadata.keys())
                )
            config_pbtxt = text_format.MessageToString(config)
            path = os.path.join(self._log_write_dir, "projector_config.pbtxt")
            with open(path, "w") as f:
                f.write(config_pbtxt)

        def _push_writer(self, writer, step):
            """Sets the default writer for custom batch-level summaries."""
            if self.update_freq == "epoch":
                return
            summary_state = summary_ops_v2._summary_state
            self._prev_summary_state.append(
                {
                    "is_recording": summary_state.is_recording,
                    "writer": summary_state.writer,
                    "step": summary_state.step,
                }
            )
            if self.update_freq == "epoch":
                should_record = False
                writer = None
            else:
                should_record = lambda: math_ops.equal(step % self.update_freq, 0)
            summary_state.is_recording = should_record
            summary_state.writer = writer
            summary_ops_v2.set_step(step.value())

        def _pop_writer(self):
            """Pops the current writer."""
            if self.update_freq == "epoch":
                return
            prev_state = self._prev_summary_state.pop()
            summary_state = summary_ops_v2._summary_state
            summary_state.is_recording = prev_state["is_recording"]
            summary_state.writer = prev_state["writer"]
            summary_ops_v2.set_step(prev_state["step"])

        def _close_writers(self):
            for writer in self._writers.values():
                writer.close()

        def _init_profile_batch(self, profile_batch):
            """Validate profile_batch value and set the range of batches to profile.

            Args:
              profile_batch: The range of batches to profile. Should be a non-negative
                integer or a comma separated string of pair of positive integers. A pair
                of positive integers signify a range of batches to profile.

            Returns:
              A pair of non-negative integers specifying the start and stop batch to
              profile.

            Raises:
              ValueError: If profile_batch is not an integer or a comma seperated pair
                          of positive integers.

            """
            profile_batch_error_message = "profile_batch must be a non-negative integer or 2-tuple of positive integers. A pair of positive integers signifies a range of batches to profile. Found: {}".format(
                self.profile_batch
            )
            if isinstance(self.profile_batch, six.string_types):
                self.profile_batch = str(self.profile_batch).split(",")
                self.profile_batch = nest.map_structure(int, self.profile_batch)
            if isinstance(self.profile_batch, int):
                self._start_batch = self.profile_batch
                self._stop_batch = self.profile_batch
            elif (
                isinstance(self.profile_batch, (tuple, list))
                and len(self.profile_batch) == 2
            ):
                self._start_batch, self._stop_batch = self.profile_batch
            else:
                raise ValueError(profile_batch_error_message)
            if self._start_batch < 0 or self._stop_batch < self._start_batch:
                raise ValueError(profile_batch_error_message)
            if self._start_batch > 0:
                profiler.warmup()
            self._is_tracing = False
            self._should_trace = not (self._start_batch == 0 and self._stop_batch == 0)

        def on_train_begin(self, logs=None):
            self._global_train_batch = 0
            self._push_writer(self._train_writer, self._train_step)

        def on_train_end(self, logs=None):
            self._pop_writer()
            if self._is_tracing:
                self._stop_trace()
            self._close_writers()
            self._delete_tmp_write_dir()

        def on_test_begin(self, logs=None):
            self._push_writer(self._val_writer, self._val_step)

        def on_test_end(self, logs=None):
            self._pop_writer()

        def on_train_batch_begin(self, batch, logs=None):
            self._global_train_batch += 1
            if not self._should_trace:
                return
            if self._global_train_batch == self._start_batch:
                self._start_trace()

        def on_train_batch_end(self, batch, logs=None):
            if not self._should_trace:
                return
            if self._is_tracing and self._global_train_batch >= self._stop_batch:
                self._stop_trace()

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch = epoch

        def on_epoch_end(self, epoch, logs=None):
            """Runs metrics and histogram summaries at epoch end."""
            self._log_epoch_metrics(epoch, logs)
            if self.histogram_freq and epoch % self.histogram_freq == 0:
                self._log_weights(epoch)
            if self.embeddings_freq and epoch % self.embeddings_freq == 0:
                self._log_embeddings(epoch)

        def _start_trace(self):
            summary_ops_v2.trace_on(graph=True, profiler=False)
            profiler.start(logdir=self._train_dir)
            self._is_tracing = True

        def _stop_trace(self, batch=None):
            """Logs the trace graph to TensorBoard."""
            if batch is None:
                batch = self._stop_batch
            with self._train_writer.as_default():
                with summary_ops_v2.always_record_summaries():
                    summary_ops_v2.trace_export(name="batch_%d" % batch, step=batch)
            profiler.stop()
            self._is_tracing = False

        def _log_epoch_metrics(self, epoch, logs):
            """Writes epoch metrics out as scalar summaries.

            Args:
                epoch: Int. The global step to use for TensorBoard.
                logs: Dict. Keys are scalar summary names, values are scalars.
            """
            if not logs:
                return
            train_logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
            val_logs = {k: v for k, v in logs.items() if k.startswith("val_")}
            with summary_ops_v2.always_record_summaries():
                if train_logs:
                    with self._train_writer.as_default():
                        for name, value in train_logs.items():
                            summary_ops_v2.scalar("epoch_" + name, value, step=epoch)
                if val_logs:
                    with self._val_writer.as_default():
                        for name, value in val_logs.items():
                            name = name[4:]
                            summary_ops_v2.scalar("epoch_" + name, value, step=epoch)

        def _log_weights(self, epoch):
            """Logs the weights of the Model to TensorBoard."""
            with self._train_writer.as_default():
                with summary_ops_v2.always_record_summaries():
                    for layer in self.model.layers:
                        for weight in layer.weights:
                            weight_name = weight.name.replace(":", "_")
                            summary_ops_v2.histogram(weight_name, weight, step=epoch)
                            if self.write_images:
                                self._log_weight_as_image(weight, weight_name, epoch)
                    self._train_writer.flush()

        def _log_weight_as_image(self, weight, weight_name, epoch):
            """Logs a weight as a TensorBoard image."""
            w_img = array_ops.squeeze(weight)
            shape = K.int_shape(w_img)
            if len(shape) == 1:
                w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
            elif len(shape) == 2:
                if shape[0] > shape[1]:
                    w_img = array_ops.transpose(w_img)
                    shape = K.int_shape(w_img)
                w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
            elif len(shape) == 3:
                if K.image_data_format() == "channels_last":
                    w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
                    shape = K.int_shape(w_img)
                w_img = array_ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])
            shape = K.int_shape(w_img)
            if len(shape) == 4 and shape[-1] in [1, 3, 4]:
                summary_ops_v2.image(weight_name, w_img, step=epoch)

        def _log_embeddings(self, epoch):
            embeddings_ckpt = os.path.join(
                self._log_write_dir, "train", "keras_embedding.ckpt-{}".format(epoch)
            )
            self.model.save_weights(embeddings_ckpt)


class TerminateOnNaNConfig(object):
    """
    Callback that terminates training when a NaN loss is encountered."""

    def __call__(self):
        def on_batch_end(self, batch, logs=None):
            logs = logs or {}
            loss = logs.get("loss")
            if loss is not None:
                if np.isnan(loss) or np.isinf(loss):
                    print("Batch %d: Invalid loss, terminating training" % batch)
                    self.model.stop_training = True
