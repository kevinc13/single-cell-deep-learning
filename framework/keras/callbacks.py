from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import os, six, time, csv
from collections import OrderedDict, Iterable

import numpy as np

from keras.callbacks import Callback


class TimeLogger(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = int(round(time.time() * 1000))

    def on_epoch_end(self, epoch, logs={}):
        logs["epoch_start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
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
        self.delimiter = str(delimiter)
        self.file_flags = "b" if six.PY2 and os.name == "nt" else ""
        super(FileLogger, self).__init__()

    def on_train_begin(self, logs={}):
        if self.append:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.file = open(self.filepath, "a" + self.file_flags)
        else:
            self.file = open(self.filepath, "w" + self.file_flags)

    def on_epoch_end(self, epoch, logs={}):
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return str(k)
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '[%s]' % (', '.join(map(str, k)))
            else:
                return k

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict(
                [(k, logs[k]) if k in logs else (k, "NA") for k in self.keys])

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

        row_dict = OrderedDict({"epoch": epoch + 1})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.file.flush()

    def on_train_end(self, logs={}):
        self.file.close()
        self.writer = None
