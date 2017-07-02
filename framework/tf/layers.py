from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import tensorflow as tf
try:
    from functools import reduce
except:
    pass

from .core import Layer
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


tf_activation_functions = {
    "linear": lambda x: x,
    "relu": tf.nn.relu,
    "sigmoid": tf.sigmoid,
    "softmax": tf.nn.softmax,
    "tanh": tf.tanh,
    "relu6": tf.nn.relu6,
    "softplus": tf.nn.softplus,
    "softsign": tf.nn.softsign,
    "elu": tf.nn.elu
}


class Input(Layer):
    def __init__(self, input_shape, input_dtype=tf.float32):
        self.input_tensor = tf.placeholder(input_dtype,
                                           shape=input_shape, name="x")

    def compile(self):
        self.output_tensor = self.input_tensor
        return self.output_tensor


class Activation(Layer):
    def __init__(self, activation):
        self.activation_function = tf_activation_functions[activation]

    def compile(self, input_tensor):
        self.output_tensor = self.activation_function(input_tensor)
        return self.output_tensor


class Dense(Layer):
    def __init__(self, name, output_dim, activation="linear",
                 regularizer=None, weights=None, biases=None):
        self.name = name
        self.output_dim = output_dim
        self.activation = activation
        self.regularizer = regularizer

        # If specified, use pre-initialized weights and biases
        if weights is not None and biases is not None:
            with tf.variable_scope(self.name):
                self.weights = tf.Variable(weights, dtype=tf.float32,
                                           name="weights")
                self.biases = tf.Variable(biases, dtype=tf.float32,
                                          name="biases")
        else:
            self.weights = None
            self.biases = None

    def build(self, input_tensor):
        # Get shape of input tensor
        input_shape = input_tensor.get_shape().dims

        # Require 2D input tensor
        assert(len(input_shape) == 2)
        input_dim = int(input_shape[1])

        if self.weights is None:
            self.weights = self.create_weight_variable(
                [input_dim, self.output_dim])

        if self.biases is None:
            self.biases = self.create_bias_variable([self.output_dim])

        if self.regularizer is not None:
            self.regularizer(self.weights)

    def compile(self, input_tensor):
        with tf.name_scope(self.name):
            self.build(input_tensor)  # Build layer
            self.preactivation_tensor = tf.matmul(
                input_tensor, self.weights) + self.biases
            tf.summary.histogram("preactivations", self.preactivation_tensor)

            self.output_tensor = Activation(self.activation).compile(
                self.preactivation_tensor)
            tf.summary.histogram("activations", self.output_tensor)
        
        return self.output_tensor


class Convolution1D(Layer):
    def __init__(self, name, num_filters, filter_size, stride=1,
                 padding="SAME", data_format="NHWC",
                 activation="linear", regularizer=None):
        self.name = name

        assert(len(filter_size) == 2)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.data_format = data_format

        self.stride = stride
        self.padding = padding

        self.activation = activation
        self.regularizer = regularizer

    def build(self):
        weights_dim = self.filter_size
        weights_dim.append(self.num_filters)

        self.weights = self.create_weight_variable(weights_dim)
        self.biases = self.create_bias_variable([self.num_filters])

        if self.regularizer is not None:
            self.regularizer(self.weights)

    def compile(self, input_tensor):
        with tf.name_scope(self.name):
            self.build()
            self.preactivation_tensor = tf.expand_dims(
                tf.nn.conv1d(
                    input_tensor, self.weights,
                    stride=self.stride, padding=self.padding,
                    data_format=self.data_format) + self.biases,
                axis=3)
            tf.summary.histogram("preactivations", self.preactivation_tensor)

            self.output_tensor = Activation(self.activation).compile(
                self.preactivation_tensor)
            tf.summary.histogram("activations", self.output_tensor)

        return self.output_tensor


class Convolution2D(Layer):
    def __init__(self, name, num_filters, filter_size, strides=[1, 1, 1, 1],
                 padding="SAME", activation="linear", regularizer=None):
        self.name = name

        assert(len(filter_size) == 3)
        self.num_filters = num_filters
        self.filter_size = filter_size

        self.strides = strides
        self.padding = padding

        self.activation = activation
        self.regularizer = regularizer

    def build(self):
        weights_dim = self.filter_size
        weights_dim.append(self.num_filters)

        self.weights = self.create_weight_variable(weights_dim)
        self.biases = self.create_bias_variable([self.num_filters])

        if self.regularizer is not None:
            self.regularizer(self.weights)

    def compile(self, input_tensor):
        with tf.name_scope(self.name):
            self.build()
            self.preactivation_tensor = tf.nn.conv2d(
                input_tensor, self.weights,
                strides=self.strides, padding=self.padding) + self.biases
            tf.summary.histogram("preactivations", self.preactivation_tensor)

            self.output_tensor = Activation(self.activation).compile(
                self.preactivation_tensor)
            tf.summary.histogram("activations", self.output_tensor)

        return self.output_tensor


class MaxPooling2D(Layer):
    def __init__(self, name, pooling_dim,
                 strides=[1, 1, 1, 1], padding="SAME"):
        self.name = name
        self.pooling_dim = pooling_dim
        self.strides = strides
        self.padding = padding

    def compile(self, input_tensor):
        with tf.name_scope(self.name):
            self.output_tensor = tf.nn.max_pool(
                input_tensor, ksize=self.pooling_dim,
                strides=self.strides, padding=self.padding
            )
            return self.output_tensor


class Flatten(Layer):
    def __init__(self, name):
        self.name = name

    def compile(self, input_tensor):
        input_dim = input_tensor.get_shape().dims
        if len(input_dim) <= 2:
            return input_tensor

        layer_dim = list(map(int, input_dim[1:]))  # Ignore batch size dim

        with tf.name_scope(self.name):
            self.output_tensor = tf.reshape(
                input_tensor,
                [-1, reduce(lambda x, y: x * y, layer_dim, 1)]
            )
            return self.output_tensor


class Dropout(Layer):
    def __init__(self, name):
        self.name = name

    def compile(self, input_tensor):
        with tf.name_scope(self.name):
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            self.output_tensor = tf.nn.dropout(
                input_tensor, self.keep_prob)
            return self.output_tensor


class BatchNorm(Layer):
    def __init__(self, name, decay=0.9):
        self.name = name
        self.decay = decay

    def compile(self, input_tensor):
        self.is_training = tf.placeholder(tf.bool)

        with tf.name_scope(self.name) as scope:
            return batch_norm(input_tensor, is_training=self.is_training,
                              center=True, scale=True,
                              decay=self.decay,
                              activation_fn=tf.nn.relu,
                              updates_collections=None, scope=scope)



