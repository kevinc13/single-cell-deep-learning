from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
from six import iteritems
from six.moves import xrange
from six.moves import cPickle as pickle

import tensorflow as tf
import sys, math, logging, copy

from . import monitors as mntrs


class Model(object):
    
    def __init__(self, config):
        self.config = copy.deepcopy(config)

        # Define core model attributes
        self.name = self.config["name"] if "name" in self.config else "Model"
        self.model_dir = self.config["model_dir"] \
            if "model_dir" in self.config else None
        self.monitors = []

        # Create logger
        self.logger = self.create_logger(self.name)

        # Run any extra setup steps
        self.setup()

        # Build TF graph
        self.build()

        # Create TF session
        self.sess = tf.Session()

    def setup(self):
        pass

    def create_logger(self, name):
        logger = logging.getLogger(name)
        if len(logger.handlers) > 0:
            return logger

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(name)s] %(asctime)s - %(message)s")

        if self.model_dir is not None:
            file_handler = logging.FileHandler(
                "{0}/model.log".format(self.model_dir))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        return logger

    @staticmethod
    def save_config(model_dir, model_config):
        with open(model_dir + "/model_config.pkl", "wb") \
                as f:
            pickle.dump(model_config, f, protocol=2)

    @staticmethod
    def get_config(model_dir):
        with open(model_dir + "/model_config.pkl", "rb") as f:
            return pickle.load(f)

    def build(self, graph):
        raise Exception("The model must implement the build method")

    def restore_from_checkpoint(self):
        if self.model_dir is not None:
            checkpoint_dir = self.model_dir + "/checkpoints"
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            
            saver = tf.train.Saver()
            saver.restore(self.sess, latest_checkpoint)

            self.logger.info("Restored from latest checkpoint: {0}"
                             .format(latest_checkpoint))


class SequentialModel(Model):
    
    def __init__(self, config):
        self.layers = []
        self.outputs = []

        super(SequentialModel, self).__init__(config)

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("The added layer must be an "
                             "instance of class Layer. "
                             "Found: " + str(layer))

        if len(self.layers) == 0:
            self.outputs.append(layer.compile(self.x))
        else:
            self.outputs.append(layer.compile(self.outputs[-1]))

        self.layers.append(layer)

    def _train_loop(self, train_dataset, validation_dataset=None,
                  num_epochs=100, batch_size=100, epoch_log_verbosity=1,
                  batch_log_verbosity=None, monitors=[]):
        """
        Trains the model

        Args:
            train_dataset (DataSet): DataSet of training examples
            monitors: List of Monitors to attach to the training loop
        """

        assert(type(self.sess) is tf.Session)

        # Initialize all variables in the graph
        self.sess.run(tf.global_variables_initializer())

        self.stop_training = False
        num_batches = int(math.ceil(train_dataset.num_examples / batch_size))

        monitors = [mntrs.MetricMonitor(), mntrs.LoggingMonitor()] + monitors
        monitors = mntrs.MonitorList(monitors)

        monitors.set_params({
            "num_epochs": num_epochs,
            "num_batches": num_batches,
            "epoch_log_verbosity": epoch_log_verbosity,
            "batch_log_verbosity": batch_log_verbosity
        })
        monitors.set_model(self)
        monitors.set_logger(self.logger)

        monitors.on_train_begin()

        for epoch in xrange(1, num_epochs + 1):
            monitors.on_epoch_begin(epoch)

            epoch_logs = {}
            epoch_logs["epoch_metrics"] = {}

            for batch in xrange(1, num_batches + 1):
                monitors.on_batch_begin(batch)

                self.batch_x, self.batch_y = train_dataset.next_batch(
                    batch_size)

                # Run train step
                feed_dict = {
                    self.x: self.batch_x,
                    self.y: self.batch_y
                }

                if hasattr(self, "feed_dict") and self.feed_dict is not None:
                    feed_dict.update(self.feed_dict)

                _, batch_cost = self.sess.run([self.train_step, self.cost],
                                              feed_dict=feed_dict)

                epoch_logs["batch_metrics"] = {}
                epoch_logs["batch_metrics"]["train_cost"] = batch_cost

                monitors.on_batch_end(batch, epoch_logs)

            if validation_dataset is not None:
                val_cost, val_metrics = self.evaluate(
                    validation_dataset, batch_size=batch_size)
                epoch_logs["epoch_metrics"]["val_cost"] = val_cost

                if val_metrics is not None:
                    for k, v in iteritems(val_metrics.as_dict()):
                        epoch_logs["epoch_metrics"]["val_{0}".format(k)] = v

            monitors.on_epoch_end(epoch, epoch_logs)

            if self.stop_training:
                break

        monitors.on_train_end()


class Layer(object):

    def create_variable(
            self, name, shape, initializer,
            dtype=tf.float32, summary=True):
        """ Create variable within the variable scope of the layer """
        with tf.variable_scope(self.name):
            var = tf.get_variable(
                    name,
                    shape,
                    initializer=initializer,
                    dtype=dtype
                )

        if summary:
            self.create_variable_summary(var, name)

        return var

    def create_variable_summary(self, var, name):
        with tf.name_scope("{}_summary".format(name)):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
            tf.summary.scalar("max", tf.reduce_max(var))
            tf.summary.scalar("min", tf.reduce_min(var))
            tf.summary.histogram("histogram", var)

    def create_weight_variable(
            self, shape,
            initializer=tf.contrib.layers.xavier_initializer(),
            summary=True):
        """ Create weight variable """
        return self.create_variable("weights", shape, initializer, summary)

    def create_bias_variable(
            self, shape,
            initializer=tf.constant_initializer(0.1),
            summary=True):
        """ Intialize neurons with slightly positive bias """
        return self.create_variable("biases", shape, initializer, summary)
