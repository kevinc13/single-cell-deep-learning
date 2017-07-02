from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
from six import iteritems
from six.moves import cPickle as pickle

import tensorflow as tf


class MonitorList:
    def __init__(self, monitors=[]):
        self.monitors = monitors

    def add_monitor(self, monitor):
        self.monitors.append(monitor)

    def set_params(self, params):
        for monitor in self.monitors:
            monitor.params = params

    def set_model(self, model):
        for monitor in self.monitors:
            monitor.model = model

    def set_logger(self, logger):
        for monitor in self.monitors:
            monitor.logger = logger

    def on_train_begin(self, logs={}):
        for monitor in self.monitors:
            monitor.on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs={}):
        for monitor in self.monitors:
            monitor.on_epoch_begin(epoch, logs)

    def on_batch_begin(self, batch, logs={}):
        for monitor in self.monitors:
            monitor.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs={}):
        for monitor in self.monitors:
            monitor.on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs={}):
        for monitor in self.monitors:
            monitor.on_epoch_end(epoch, logs)

    def on_train_end(self, logs={}):
        for monitor in self.monitors:
            monitor.on_train_end(logs)


class Monitor(object):
    def __init__(self):
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass


class MetricMonitor(Monitor):
    def on_train_begin(self, logs={}):
        self.num_batches = self.params["num_batches"]

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_metric_totals = {}

    def on_batch_end(self, batch, logs={}):
        batch_metrics = logs["batch_metrics"]
        for k, v in iteritems(batch_metrics):
            if k in self.epoch_metric_totals:
                self.epoch_metric_totals[k] += v
            else:
                self.epoch_metric_totals[k] = v

    def on_epoch_end(self, epoch, logs={}):
        for k, v in iteritems(self.epoch_metric_totals):
            logs["epoch_metrics"][k] = v / self.num_batches


class LoggingMonitor(Monitor):
    def on_train_begin(self, logs={}):
        self.num_epochs = self.params["num_epochs"]
        self.num_batches = self.params["num_batches"]

    def on_epoch_begin(self, epoch, logs={}):
        if self.params["epoch_log_verbosity"] is not None and \
                epoch % self.params["epoch_log_verbosity"] == 0:
            self.logger.info("Epoch {0}/{1}:".format(epoch, self.num_epochs))

    def on_batch_end(self, batch, logs={}):
        if self.params["batch_log_verbosity"] is not None and \
                batch % self.params["batch_log_verbosity"] == 0:
            metrics_string = ""
            for k, v in iteritems(logs["batch_metrics"]):
                metrics_string += "{}={:.9f}; ".format(k, v)

            self.logger.info("Batch {0}/{1}: {2}".format(
                batch, self.num_batches, metrics_string))

    def on_epoch_end(self, epoch, logs={}):
        if self.params["epoch_log_verbosity"] is not None and \
                epoch % self.params["epoch_log_verbosity"] == 0:
            metrics_string = ""
            for k, v in iteritems(logs["epoch_metrics"]):
                metrics_string += "{}={:.9f}; ".format(k, v)

            self.logger.info(metrics_string)


class CheckpointMonitor(Monitor):
    def __init__(self, checkpoint_dir=""):
        super(CheckpointMonitor, self).__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_train_begin(self, logs={}):
        self.saver = tf.train.Saver()

    def on_epoch_end(self, epoch, logs={}):
        self.saver.save(self.model.sess, self.checkpoint_dir + "/model",
                        global_step=epoch)
        self.logger.info("Saved checkpoint for epoch {0}".format(epoch))


class EarlyStoppingMonitor(Monitor):
    def __init__(self, min_delta, patience, metric="train_cost", mode="max"):
        self.min_delta = min_delta
        self.patience = patience
        self.metric = metric
        self.mode = mode

        if self.mode == "min":
            self.min_delta *= -1

        self.no_improvement_count = 0

    def on_train_begin(self, logs={}):
        self.prev_epoch_metric = 0.0

    def on_epoch_end(self, epoch, logs={}):
        delta = logs["epoch_metrics"][self.metric] - self.prev_epoch_metric
        if self.mode == "max":
            if delta < self.min_delta:
                self.no_improvement_count += 1
        elif self.mode == "min":
            if delta > self.min_delta:
                self.no_improvement_count += 1
        else:
            # Reset no improvement count if epoch passes min delta
            self.no_improvement_count = 0

        if self.no_improvement_count == self.patience:
            self.model.stop_training = True
            self.logger.info(
                "Early stopping patience exceeded, stopping training")

        self.prev_epoch_metric = logs["epoch_metrics"][self.metric]


class TensorBoardMonitor(Monitor):
    def __init__(self, log_dir="", batch_summarize_verbosity=10):
        super(TensorBoardMonitor, self).__init__()
        self.log_dir = log_dir
        self.batch_summarize_verbosity = batch_summarize_verbosity

    def on_train_begin(self, logs={}):
        # Merge all summaries
        self.merge_summaries = tf.summary.merge_all()

        # Create TF Summary Writer
        self.train_writer = tf.summary.FileWriter(self.log_dir,
                                                  self.model.sess.graph)

    def on_epoch_begin(self, epoch, logs={}):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs={}):
        if batch % self.batch_summarize_verbosity == 0:
            feed_dict = {
                self.model.x: self.model.batch_x,
            }

            if hasattr(self.model, "y"):
                feed_dict[self.model.y] = self.model.batch_y

            if hasattr(self.model, "feed_dict") \
                    and self.model.feed_dict is not None:
                feed_dict.update(self.model.feed_dict)

            summary = self.model.sess.run(
                self.merge_summaries, feed_dict=feed_dict)
            self.train_writer.add_summary(summary, self.current_epoch)

    def on_epoch_end(self, epoch, logs={}):
        self.train_writer.flush()

    def on_train_end(self, logs={}):
        self.train_writer.close()
