from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
from six.moves import xrange
import math

import tensorflow as tf
import numpy as np

from .evaluation import ClassificationMetrics
from .core import SequentialModel
from .layers import (
    Dense, Activation,
    Convolution1D, Convolution2D, MaxPooling2D,
    Flatten, Dropout
)

from .monitors import (
    CheckpointMonitor, TensorBoardMonitor, EarlyStoppingMonitor
)


class CNN(SequentialModel):
    """
    Implements a Convolutional Neural Network in TensorFlow

    Usage:
        from tensorflow.examples.tutorials.mnist import input_data
        from cnn import CNN

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        mnist_cnn = CNN(mnist_config, mnist.train)
        mnist_cnn.train()
    """

    def __init__(self, name, session, config, model_dir=None):
        """
        Initializes the CNN

        Args:
            session: Tensorflow session
            config (dict): A dictionary of configuration parameters
                architecture (list) - Defines the architecture of the CNN
                    "INPUT:h,w,d" - Input layer of size [h x w x d]
                    "CONV:k,fh,fw,fd:s:f"
                        - Convolutional layer with k filters (or features),
                          filter size of [fh x fw x fd], strides of
                          [1, s, s, 1], SAME zero-padding, and activation
                          function f
                    "MAXPOOL:h,w:s"
                        - Max pooling layer with subsample size of [h x w],
                          strides of [1, s, s, 1], and SAME zero-padding
                    "FC:n:f" - Fully-connected layer with n hidden neurons and
                               activation function f
                             - Default activation function = relu
                    "DROPOUT:p" - Dropout layer with a dropout probability of p
                    "OUTPUT:n:f" - Output layer of size n with activation
                                   function f

                cost_function - Cost function of CNN
                optimizer - Initialized TF optimizer
                regularizer - TF regularizer

                num_epochs - Number of epochs to run
                batch_size - Size of training batches
                display_step - When to display logs per epoch step

                model_dir - Directory where model checkpoints
                            and logs are saved
                restore_model - Whether or not to restore model from the latest
                                checkpoint saved in the model directory

                For example:
                mnist_config = {
                    "architecture": ["INPUT:28,28,1",
                                  "CONV:5,5,1,32:1:relu",
                                  "MAXPOOL:2,2:2",
                                  "CONV:5,5,32,64:1:relu",
                                  "MAXPOOL:2,2:2",
                                  "FC:1024:relu",
                                  "OUTPUT:10"],
                    "activation": tf.nn.relu,
                    "optimizer": tf.train.AdamOptimizer(learning_rate=1e-4,
                        epsilon=1.0),
                    "cost_function": tf.nn.softmax_cross_entropy_with_logits,
                    "num_epochs": 1,
                    "batch_size": 100,
                    "display_step": 1
                }
        """

        self.name = name
        self.sess = session
        self.config = config
        self.model_dir = model_dir

        # Setup parent class
        super(CNN, self).__init__()

        # Build model
        self.build()

    def build(self):
        """
        Build the TensorFlow graph using the model configuration
        """

        dense_layer_counter = 0
        conv_layer_counter = 0
        maxpool_layer_counter = 0
        flatten_layer_counter = 0
        dropout_layer_counter = 0

        for index, layer in enumerate(self.config["architecture"]):
            if "INPUT" in layer:
                # Parse input layer dimensions
                input_layer_def = self.config["architecture"][0]
                shape = list(
                    map(int, input_layer_def.split(":")[1].split(",")))

                self.x = tf.placeholder(tf.float32,
                                        shape=[None] + shape, name="x")

            elif "OUTPUT" in layer:
                _, output_layer_size, output_layer_activation = \
                    self.config["architecture"][-1].split(":")
                output_layer_size = int(output_layer_size)

                self.y = tf.placeholder(tf.float32,
                                        shape=[None, output_layer_size],
                                        name="y")

            elif "CONV2D" in layer:
                conv_layer_counter += 1

                _, n_filters, filter_dim, stride, activation = layer.split(":")

                stride = int(stride)
                n_filters = int(n_filters)

                filter_height, filter_width, filter_depth = tuple(
                    map(int, filter_dim.split(",")))

                self.add(
                    Convolution2D("conv2d_" + str(conv_layer_counter),
                                  n_filters,
                                  [filter_height, filter_width, filter_depth],
                                  strides=[1, stride, stride, 1],
                                  activation=activation,
                                  regularizer=self.config["regularizer"])
                )

            elif "CONV1D" in layer:
                conv_layer_counter += 1

                _, n_filters, filter_dim, stride, activation = layer.split(":")

                stride = int(stride)
                n_filters = int(n_filters)

                filter_dim = list(map(int, filter_dim.split(",")))

                self.add(
                    Convolution1D("conv1d_" + str(conv_layer_counter),
                                  n_filters, filter_dim,
                                  stride=stride,
                                  activation=activation,
                                  regularizer=self.config["regularizer"])
                )

            elif "MAXPOOL" in layer:
                maxpool_layer_counter += 1

                (_, window_dim, strides) = layer.split(":")
                stride_h, stride_w = tuple(map(int, strides.split(",")))
                pool_h, pool_w = tuple(map(int, window_dim.split(",")))

                self.add(MaxPooling2D(
                    "maxpool_" + str(maxpool_layer_counter),
                    [1, pool_h, pool_w, 1],
                    [1, stride_h, stride_w, 1]
                ))

            elif "FC" in layer:
                dense_layer_counter += 1

                num_units = int(layer.split(":")[1])
                activation = layer.split(":")[2]

                # Check if FC layer is the first layer after the input layer
                prev_layer = self.x if index == 0 else self.outputs[-1]

                if len(prev_layer.get_shape().dims) > 2:
                    flatten_layer_counter += 1
                    self.add(Flatten("flatten_" + str(flatten_layer_counter)))

                self.add(Dense("dense_" + str(dense_layer_counter),
                               num_units, activation=activation,
                               regularizer=self.config["regularizer"]))
            elif "DROPOUT" in layer:
                self.dropout_layer = Dropout("dropout")
                self.add(self.dropout_layer)

                self.feed_dict = {
                    self.dropout_layer.keep_prob: float(layer.split(":")[1])
                }

        # Linear transformation of last hidden layer (output layer)
        self.add(Dense("output", output_layer_size,
                       regularizer=self.config["regularizer"]))
        self.add(Activation(output_layer_activation))

        # Output prediction of model
        self.y_pred = tf.cast(tf.argmax(self.outputs[-1], 1), tf.float32)

        # Add overall model cost function to total loss
        tf.add_to_collection("losses", tf.reduce_mean(
            self.config["cost_function"](self.outputs[-2], self.y)))

        # Compute total loss
        self.cost = tf.add_n(tf.get_collection("losses"), "cost")
        tf.scalar_summary("cost", self.cost)

        # Define training step
        self.train_step = self.config["optimizer"].minimize(self.cost)

    def fit(self, train_dataset, validation_dataset=None,
            num_epochs=100, batch_size=100,
            epoch_log_verbosity=1, batch_log_verbosity=None):
        """
        Trains the CNN using a given training data set

        Args:
            train_dataset (DataSet): DataSet of training examples
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

        self._fit_loop(train_dataset, validation_dataset=validation_dataset,
                       num_epochs=num_epochs, batch_size=batch_size,
                       epoch_log_verbosity=epoch_log_verbosity,
                       batch_log_verbosity=batch_log_verbosity,
                       monitors=monitors)

    def predict(self, features):
        """
        Predict the labels of a given set of features using the CNN

        Args:
            features (numpy.ndarray): Features with the same shape as the CNN's
                                      input layer

        Returns:
            Numpy array of the CNN's predictions
        """

        feed_dict = {
            self.x: features
        }

        if hasattr(self, "dropout_layer"):
            feed_dict[self.dropout_layer.keep_prob] = 1.0

        return self.sess.run(self.y_pred, feed_dict=feed_dict)

    def evaluate(self, test_dataset, batch_size=100):
        """
        Evaluates the CNN on a test dataset

        Args:
            test_dataset (DataSet): DataSet of test examples
            batch_size: Size of the batches that are run one-by-one through
                        the model (default is 100)

        Returns:
            A tuple of the test dataset's total cost and classification results
        """

        total_cost = 0.0
        total_pred = []

        num_batches = int(math.ceil(test_dataset.num_examples / batch_size))
        for i in xrange(num_batches):
            batch_x, batch_y = test_dataset.next_batch(batch_size)
            feed_dict = {
                self.x: batch_x,
                self.y: batch_y
            }

            if hasattr(self, "dropout_layer"):
                feed_dict[self.dropout_layer.keep_prob] = 1.0

            batch_pred = self.predict(batch_x)

            total_cost += self.sess.run(self.cost, feed_dict=feed_dict)
            total_pred.extend(batch_pred.tolist())

        total_cost /= num_batches

        return (total_cost, ClassificationMetrics(
                np.argmax(test_dataset.labels, axis=1),
                np.asarray(total_pred)))
