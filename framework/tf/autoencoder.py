from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
from six.moves import xrange

import tensorflow as tf
import math

from .core import Model


class DeepAutoencoder(Model):
    """
    Deep autoencoder
    """
    def setup(self):
        self.pretrain_weights = self.config["pretrain_weights"] \
            if "pretrain_weights" in self.config else None
        self.pretrain_biases = self.config["pretrain_biases"] \
            if "pretrain_biases" in self.config else None

    def build(self):
        decoder_hidden_layers = list(reversed(
            self.config["encoder_hidden_layers"][:-1]))
        all_hidden_layers = self.config["encoder_hidden_layers"] + \
            decoder_hidden_layers

        if type(self.config["activations"]) == list:
            assert((len(all_hidden_layers) + 1) \
                == len(self.config["activations"]))
            self.activations = self.config["activations"]
        else:
            self.activations = [self.config["activations"] \
                for i in range(0, (len(all_hidden_layers) + 1))]

        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.config["input_size"]],
                                name="x")

        # Same as input
        self.y = tf.placeholder(tf.float32,
                                shape=[None, self.config["input_size"]],
                                name="y")

        for index, size in enumerate(all_hidden_layers):
            if self.pretrain_weights is not None and \
                    self.pretrain_biases is not None:
                self.add(Dense(
                    "dense_{0}".format(index + 1),
                    size, activation=self.activations[index],
                    regularizer=self.config["regularizer"],
                    weights=self.pretrain_weights[index],
                    biases=self.pretrain_biases[index]))
            else:
                self.add(Dense(
                    "dense_{0}".format(index + 1),
                    size, activation=self.activations[index],
                    regularizer=self.config["regularizer"]))

        if self.pretrain_weights is not None and \
                self.pretrain_biases is not None:
            self.add(Dense("output", self.config["input_size"],
                           activation=self.activations[-1],
                           regularizer=self.config["regularizer"],
                           weights=self.pretrain_weights[-1],
                           biases=self.pretrain_biases[-1]))
        else:
            self.add(Dense("output", self.config["input_size"],
                           activation=self.activations[-1],
                           regularizer=self.config["regularizer"]))

        tf.add_to_collection("losses", 
            self.config["loss"](self.y, self.layers[-1].preactivation_tensor))
        self.cost = tf.reduce_mean(tf.add_n(tf.get_collection("losses"), 
                             name=self.config["loss"].__name__))
        tf.summary.scalar("cost", self.cost)

        self.global_step = tf.Variable(
            0, name="global_step", trainable=False)
        self.train_step = self.config["optimizer"].minimize(
            self.cost, global_step=self.global_step)

    def train(self, train_dataset, validation_dataset=None,
              num_epochs=100, batch_size=100,
              epoch_log_verbosity=1, batch_log_verbosity=None):
        """
        Trains the Autoencoder using a given training data set

        Args:
            train_dataset (DataSet): DataSet of training examples
            :param epoch_log_verbosity:
            :param num_epochs:
            :param train_dataset:
            :param batch_size:
            :param batch_log_verbosity:
            :param validation_dataset:
        """
        monitors = []

        if self.model_dir is not None:
            monitors.append(CheckpointMonitor(self.model_dir + "/checkpoints"))
            monitors.append(TensorBoardMonitor(
                            self.model_dir + "/tensorboard"))

        if "early_stopping_metric" in self.config \
                and "early_stopping_min_delta" in self.config \
                and "early_stopping_patience" in self.config:
            monitors.append(EarlyStoppingMonitor(
                min_delta=self.config["early_stopping_min_delta"],
                patience=self.config["early_stopping_patience"],
                metric=self.config["early_stopping_metric"]))

        self._train_loop(train_dataset, validation_dataset=validation_dataset,
                         num_epochs=num_epochs, batch_size=batch_size,
                         epoch_log_verbosity=epoch_log_verbosity,
                         batch_log_verbosity=batch_log_verbosity,
                         monitors=monitors)

    def predict(self, features):
        """
        Output autoencoder logits

        Args:
            features (numpy.ndarray): Features with the same shape as the CNN's
                                      input layer

        Returns:
            Numpy array of the CNN's predictions
        """

        return self.sess.run(self.outputs[-1], feed_dict={self.x: features})

    def evaluate(self, test_dataset, batch_size=100):
        """
        Evaluates the autoencoder on a test dataset

        Args:
            test_dataset (DataSet): DataSet of test examples
            batch_size: Size of the batches that are run one-by-one through
                        the model (default is 100)

        Returns:
            A tuple of the test dataset's total cost and None
        """

        total_cost = 0.0
        total_pred = []

        num_batches = int(math.ceil(test_dataset.num_examples / batch_size))
        for i in xrange(num_batches):
            batch_x, batch_y = test_dataset.next_batch(batch_size)

            batch_pred = self.predict(batch_x)

            total_cost += self.sess.run(self.cost, feed_dict={
                self.x: batch_x,
                self.y: batch_y
            })
            total_pred.extend(batch_pred.tolist())

        total_cost /= num_batches

        return total_cost, None

