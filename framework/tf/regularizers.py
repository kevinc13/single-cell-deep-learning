from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import tensorflow as tf


class L1:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, weights):
        tf.add_to_collection("losses", tf.mul(tf.reduce_sum(tf.abs(weights)),
                             self.scale, name="l1_regularizer"))


class L2:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, weights):
        tf.add_to_collection("losses", tf.mul(tf.nn.l2_loss(weights),
                             self.scale, name="l2_regularizer"))
