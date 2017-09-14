from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import os, six, time, csv, copy
from collections import OrderedDict, Iterable

import numpy as np

import keras.backend as K
import keras.callbacks as cbks
from keras.callbacks import Callback


class CallbackManager:
    def __init__(self, epochs, batch_size,
                 models=None, callbacks=None,
                 do_validation=None, validation_data=None):
        self.models = models if models else {}
        self.callbacks = callbacks if callbacks else {}
        self.callback_lists = {}

        self.epochs = epochs
        self.batch_size = batch_size
        self.do_validation = do_validation \
            if validation_data is not None else []
        self.validation_data = validation_data

        self.is_setup = False

    @property
    def stop_training(self):
        for _, model in self.models.items():
            if model.stop_training:
                return True
        return False

    def add_model(self, model_name, model):
        self.models[model_name] = model

    def add_callbacks(self, model_name, callbacks):
        self.callbacks[model_name] = callbacks

    def setup(self):
        if not self.is_setup:
            for model_name, model in self.models.items():
                if model_name in self.callbacks:
                    callback_list = self.get_callback_list(model_name)
                else:
                    callback_list = self.get_callback_list(model)
                model.stop_training = False

                if model_name in self.do_validation:
                    val_ins = self.get_val_ins(model_name)
                    for cbk in callback_list:
                        cbk.validation_data = val_ins

                self.callback_lists[model_name] = callback_list

            self.is_setup = True

    def get_callback_list(self, model_name):
        if model_name in self.callback_lists:
            return self.callback_lists[model_name]

        model = self.models[model_name]
        callbacks = self.callbacks[model_name] \
            if model_name in self.callbacks else []

        # Prepare callbacks for autoencoder model
        all_callbacks = [cbks.BaseLogger()] + callbacks + [cbks.History()]
        all_callbacks = cbks.CallbackList(all_callbacks)
        out_labels = model.metrics_names

        if self.do_validation:
            callback_metrics = copy.copy(out_labels) + \
                               ["val_" + l for l in out_labels]
        else:
            callback_metrics = copy.copy(out_labels)

        callback_list = cbks.CallbackList(all_callbacks)
        callback_list.set_params({
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'verbose': 2,
            'do_validation': model_name in self.do_validation,
            'metrics': callback_metrics or [],
        })
        callback_list.set_model(model)

        return callback_list

    def get_val_ins(self, model_name):
        model = self.models[model_name]

        val_x, val_y = self.validation_data
        val_x, val_y, val_sample_weights = model._standardize_user_data(
                val_x, val_y, batch_size=self.batch_size)
        if model.uses_learning_phase and \
                not isinstance(K.learning_phase(), int):
            val_ins = val_x + val_y + val_sample_weights + [0.]
        else:
            val_ins = val_x + val_y + val_sample_weights
        return val_ins

    def on_epoch_begin(self, epoch, logs=None, model_name=None):
        """Called at the start of an epoch.

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        if not self.is_setup:
            raise Exception("Callback lists have not been setup yet")

        logs = logs or {}

        if model_name is not None:
            self.callback_lists[model_name].on_epoch_begin(epoch, logs)
        else:
            for _, callback_list in self.callback_lists.items():
                callback_list.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None, model_name=None):
        """Called at the end of an epoch.

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        if model_name is not None:
            self.callback_lists[model_name].on_epoch_end(epoch, logs)
        else:
            for _, callback_list in self.callback_lists.items():
                callback_list.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None, model_name=None):
        """Called right before processing a batch.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        if not self.is_setup:
            raise Exception("Callback lists have not been setup yet")

        logs = logs or {}
        if model_name is not None:
            self.callback_lists[model_name].on_batch_begin(batch, logs)
        else:
            for _, callback_list in self.callback_lists.items():
                callback_list.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None, model_name=None):
        """Called at the end of a batch.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        if model_name is not None:
            self.callback_lists[model_name].on_batch_end(batch, logs)
        else:
            for _, callback_list in self.callback_lists.items():
                callback_list.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None, model_name=None):
        """Called at the beginning of training.

        # Arguments
            logs: dictionary of logs.
        """
        if not self.is_setup:
            raise Exception("Callback lists have not been setup yet")

        logs = logs or {}
        if model_name is not None:
            self.callback_lists[model_name].on_train_begin(logs)
        else:
            for _, callback_list in self.callback_lists.items():
                callback_list.on_train_begin(logs)

    def on_train_end(self, logs=None, model_name=None):
        """Called at the end of training.

        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        if model_name is not None:
            self.callback_lists[model_name].on_train_end(logs)
        else:
            for _, callback_list in self.callback_lists.items():
                callback_list.on_train_end(logs)


class TerminateOnNaN(Callback):
    """
    Callback that terminates training when a NaN loss is encountered.

    Note: To prevent errors with other callbacks,
    the model stops training after the epoch ends
    """

    def __init__(self):
        super(TerminateOnNaN, self).__init__()
        self.stop_training = False

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                self.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        if self.stop_training:
            print("Epoch %d: Invalid loss, terminating training" % epoch)
            self.model.stop_training = True


class TimeLogger(Callback):
    def __init__(self):
        super(TimeLogger, self).__init__()
        self.epoch_start_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = int(round(time.time() * 1000))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        end_time = int(round(time.time() * 1000))
        logs["epoch_computation_time"] = \
            str(end_time - self.epoch_start_time) + "ms"


class FileLogger(Callback):
    """
    Custom file logging callback (based on keras.callbacks.CSVLogger)
    """

    def __init__(self, filepath, delimiter="\t", append=False):
        """
        Constructor

        Args:
            filepath: Path to file (ex. logs/training.log)
            delimiter: Character separating elements in a row
            append: Whether to append to log file (if it exists)
        """
        self.filepath = filepath
        self.append = append
        self.append_header = True
        self.writer = None
        self.keys = None
        self.delimiter = str(delimiter)
        self.file = None
        self.file_flags = "b" if six.PY2 and os.name == "nt" else ""
        super(FileLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        if self.append:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.file = open(self.filepath, "a" + self.file_flags)
        else:
            self.file = open(self.filepath, "w" + self.file_flags)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return str(k)
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '[%s]' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.Dialect):
                delimiter = self.delimiter
                quoting = csv.QUOTE_MINIMAL
                quotechar = str('"')
                lineterminator = "\n"

            self.writer = csv.DictWriter(
                self.file, fieldnames=['epoch'] + self.keys,
                dialect=CustomDialect)

            if self.append_header:
                self.writer.writeheader()

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict(
                [(k, logs[k]) if k in logs else (k, "NA") for k in self.keys])

        row_dict = OrderedDict({"epoch": epoch + 1})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.file.flush()

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        self.file.close()
        self.writer = None
