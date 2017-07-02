from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
from six.moves import xrange

import tensorflow as tf
import numpy as np
import math
import sys
import logging

from .core import Model
from . import monitors as mntrs
from .monitors import (
    CheckpointMonitor, TensorBoardMonitor
)


class RBMModel(Model):

    def create_variable_summary(self, var):
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
            tf.summary.scalar("max", tf.reduce_max(var))
            tf.summary.scalar("min", tf.reduce_min(var))
            tf.summary.histogram("histogram", var)

    def fit(self, train_dataset, validation_dataset=None,
            num_epochs=100, batch_size=100,
            epoch_log_verbosity=1, batch_log_verbosity=None):

        self.sess.run(tf.global_variables_initializer())

        self.stop_training = False
        num_batches = int(math.ceil(train_dataset.num_examples / batch_size))

        monitors = [mntrs.MetricMonitor(), mntrs.LoggingMonitor()]
        if self.model_dir is not None:
            monitors.append(CheckpointMonitor(self.model_dir + "/checkpoints"))
            monitors.append(TensorBoardMonitor(
                            self.model_dir + "/tensorboard"))

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

            if epoch > 5:
                m = self.config["final_momentum"]
            else:
                m = self.config["initial_momentum"]

            for batch in xrange(1, num_batches + 1):
                monitors.on_batch_begin(batch)

                self.batch_x, _ = train_dataset.next_batch(batch_size)

                # Run train step
                feed_dict = {
                    self.x: self.batch_x,
                    self.momentum: m
                }

                # w, vb, hb, batch_cost = self.sess.run(self.train_step,
                #                                       feed_dict=feed_dict)
                w, vb, hb, batch_cost, pos_hid_sample = self.sess.run(
                    self.train_step, feed_dict=feed_dict)
                # if batch % batch_size == 0:
                #     print("Proportion of activated hidden nodes: {}"
                #         .format(np.mean(pos_hid_sample)))

                epoch_logs["batch_metrics"] = {}
                epoch_logs["batch_metrics"]["train_cost"] = batch_cost

                monitors.on_batch_end(batch, epoch_logs)

            if validation_dataset is not None:
                val_cost = self.sess.run(self.cost, feed_dict={
                    self.x: validation_dataset.features
                })
                epoch_logs["epoch_metrics"]["val_cost"] = val_cost

            monitors.on_epoch_end(epoch, epoch_logs)

            if self.stop_training:
                break

        monitors.on_train_end()


class RBM(RBMModel):
    
    def __init__(self, name, session, config, model_dir=None):
        super(RBM, self).__init__()

        self.name = name
        self.sess = session
        self.config = config
        self.model_dir = model_dir

        self.logger = self.create_logger(getattr(
            self, "name", "RBM"))

        with tf.name_scope(self.name):
            self._build()

    def _build(self):
        input_size = self.config["input_size"]
        hidden_layer_size = self.config["hidden_layer_size"]

        self.x = tf.placeholder(tf.float32,
                                shape=[None, input_size],
                                name="x")

        num_samples = tf.cast(
            tf.shape(self.x)[0], tf.float32, name="n_samples")

        self.learning_rate = tf.constant(
            self.config["learning_rate"],
            dtype=tf.float32, name="learning_rate")
        self.momentum = tf.placeholder(tf.float32, name="momentum")
        self.weight_decay = tf.constant(
            self.config["weight_decay"],
            dtype=tf.float32, name="weight_decay")

        self.weights = tf.Variable(tf.random_normal(
            [input_size, hidden_layer_size], stddev=0.01), name="weights")
        self.hid_biases = tf.Variable(
            tf.zeros([hidden_layer_size]), name="hidden_biases")
        self.vis_biases = tf.Variable(
            tf.zeros([input_size]), name="visible_biases")

        self.create_variable_summary(self.weights)
        self.create_variable_summary(self.hid_biases)
        self.create_variable_summary(self.vis_biases)

        weights_increment = tf.Variable(
            tf.zeros([input_size, hidden_layer_size]),
            name="weights_increment")
        vis_bias_increment = tf.Variable(
            tf.zeros([input_size]), name="visible_biases_increment")
        hid_bias_increment = tf.Variable(
            tf.zeros([hidden_layer_size]), name="hidden_biases_increment")

        # BUILD MODEL
        # Start positive phase of 1-step Contrastive Divergence
        self.pos_hid_probs = tf.nn.sigmoid(
            tf.add(tf.matmul(self.x, self.weights), self.hid_biases))
        pos_associations = tf.matmul(
            tf.transpose(self.x), self.pos_hid_probs)
        # Number of active (1) hidden nodes
        pos_hid_act = tf.reduce_sum(self.pos_hid_probs, 0)
        # Number of active (1) visible nodes
        pos_vis_act = tf.reduce_sum(self.x, 0)

        # Start negative phase
        # Sample the hidden unit states {0,1}
        # Hidden node turns on if
        # P(h=1) > random number uniformally distributed [0, 1)
        self.pos_hid_sample = self.pos_hid_probs > tf.random_uniform(
            tf.shape(self.pos_hid_probs), 0, 1)
        self.pos_hid_sample = tf.to_float(self.pos_hid_sample)

        # pos_hid_sample_image = tf.expand_dims(
        #     tf.expand_dims(self.pos_hid_sample, 2),
        #     3, name="pos_hid_sample_image")
        # self.create_image_summary(pos_hid_sample_image)

        # Visible states (probabilities, not binary states)
        # AKA sample of x
        self.neg_vis_probs = tf.nn.sigmoid(tf.add(
            tf.matmul(self.pos_hid_sample, tf.transpose(self.weights)),
            self.vis_biases))

        rbm_version = self.config["rbm_version"]
        if rbm_version == 'Hinton_2006':
            neg_hid_probs = tf.nn.sigmoid(tf.add(
                tf.matmul(self.neg_vis_probs, self.weights),
                self.hid_biases))
            neg_associations = tf.matmul(
                tf.transpose(self.neg_vis_probs), neg_hid_probs)
            neg_hid_act = tf.reduce_sum(neg_hid_probs, 0)
            neg_vis_act = tf.reduce_sum(self.neg_vis_probs, 0)

            # Calculate mean squared error
            mse = tf.reduce_mean(tf.reduce_sum(
                tf.square(self.x - self.neg_vis_probs), 1))
            tf.add_to_collection("losses", mse)

        elif rbm_version in ['Ruslan_new', 'Bengio']:
            # Sample the visible unit states {0,1} using distribution
            # determined by self.neg_vis_probs
            self.neg_vis_sample = tf.to_float(
                self.neg_vis_probs > tf.random_uniform(
                    tf.shape(self.neg_vis_probs), 0, 1))
            neg_hid_probs = tf.nn.sigmoid(tf.add(
                tf.matmul(self.neg_vis_sample, self.weights),
                self.hid_biases))
            neg_associations = tf.matmul(tf.transpose(
                self.neg_vis_sample), neg_hid_probs)
            neg_hid_act = tf.reduce_sum(neg_hid_probs, 0)
            neg_vis_act = tf.reduce_sum(self.neg_vis_sample, 0)
            # Calculate mean squared error
            mse = tf.reduce_mean(tf.reduce_sum(
                tf.square(self.x - self.neg_vis_sample), 1))
            tf.add_to_collection("losses", mse)

            if rbm_version == 'Bengio':
                print('Using rbm_verison:  Bengio')
                pos_associations = tf.matmul(
                    tf.transpose(self.x), self.pos_hid_sample)
                pos_hid_act = tf.reduce_sum(self.pos_hid_sample, 0)
            else:
                print('Using rbm_verison:  Ruslan_new')

        # Calculate directions to move weights based on gradient (how to change
        # the weights)
        new_weights_increment = (self.momentum * weights_increment +
                                 (self.learning_rate / num_samples) * (
                                        (pos_associations - neg_associations) -
                                        self.weight_decay * self.weights
                                    ))
        new_vis_bias_increment = (self.momentum * vis_bias_increment +
                                  (self.learning_rate / num_samples) *
                                  (pos_vis_act - neg_vis_act))
        new_hid_bias_increment = (self.momentum * hid_bias_increment +
                                  (self.learning_rate / num_samples) *
                                  (pos_hid_act - neg_hid_act))

        update_weights = tf.assign(
            weights_increment, new_weights_increment)
        update_vis_bias = tf.assign(
            vis_bias_increment, new_vis_bias_increment)
        update_hid_bias = tf.assign(
            hid_bias_increment, new_hid_bias_increment)

        self.cost = tf.add_n(tf.get_collection("losses"), "cost")
        tf.scalar_summary("cost", self.cost)

        self.create_variable_summary(weights_increment)
        self.create_variable_summary(vis_bias_increment)
        self.create_variable_summary(hid_bias_increment)

        # Update weights and biases
        self.train_step = [self.weights.assign_add(update_weights),
                           self.vis_biases.assign_add(update_vis_bias),
                           self.hid_biases.assign_add(update_hid_bias),
                           self.cost, self.pos_hid_sample]


class SparseRBM(RBMModel):
    def __init__(self, name, session, config, model_dir=None)
        super(SparseRBM, self).__init__()

        self.name = name
        self.sess = session
        self.config = config
        self.model_dir = model_dir

        self.logger = self.create_logger(getattr(
            self, "name", "SparseRBM"))

        with tf.name_scope(self.name):
            self._build()

    def _build(self):
        input_size = self.config["input_size"]
        hidden_layer_size = self.config["hidden_layer_size"]

        self.x = tf.placeholder(tf.float32,
                                shape=[None, input_size],
                                name="x")

        num_samples = tf.cast(tf.shape(self.x)[0], tf.float32)

        self.learning_rate = tf.constant(
            self.config["learning_rate"],
            dtype=tf.float32, name="learning_rate")
        self.momentum = tf.placeholder(tf.float32, name="momentum")

        self.weight_decay = tf.constant(
            self.config["weight_decay"],
            dtype=tf.float32, name="weight_decay")

        self.sparsity_reg_const = tf.constant(
            self.config["sparsity_regularization_constant"],
            dtype=tf.float32, name="sparsity_reg_const")
        # "Target" sparsity
        self.sparsity_target = tf.constant(
            self.config["sparsity_target"],
            dtype=tf.float32, name="sparsity_target")

        self.weights = tf.Variable(tf.random_normal(
            [input_size, hidden_layer_size], stddev=0.01), name="weights")
        self.hid_biases = tf.Variable(
            tf.zeros([hidden_layer_size]), name="hidden_biases")
        # self.vis_biases = tf.Variable(
        #     tf.zeros([input_size]), name="visible_biases")
        self.vis_biases = tf.Variable(
            tf.constant(
                math.log(1.0/(1.0 - self.config["sparsity_target"])),
                shape=[input_size]),
            name="visible_biases")

        self.create_variable_summary(self.weights)
        self.create_variable_summary(self.hid_biases)
        self.create_variable_summary(self.vis_biases)

        weights_increment = tf.Variable(
            tf.zeros([input_size, hidden_layer_size]),
            name="weights_increment")
        vis_bias_increment = tf.Variable(
            tf.zeros([input_size]), name="visible_biases_increment")
        hid_bias_increment = tf.Variable(
            tf.zeros([hidden_layer_size]), name="hidden_biases_increment")

        # Positive Phase of 1-Step Contrastive Divergence
        self.pos_hid_probs = tf.nn.sigmoid(
            tf.add(tf.matmul(self.x, self.weights), self.hid_biases),
            name="pos_hid_probs")
        self.create_variable_summary(self.pos_hid_probs)

        sparse_version = self.config["sparse_version"]
        if sparse_version == "Goh_2011":
            # Create hidden activation bias matrix
            # self.hidden_activation_biases = tf.fill(tf.pack(
            #     tf.shape(self.pos_hid_probs)), 0.0)
            flat_size = tf.to_int32(num_samples * hidden_layer_size)
            flat_hid_probs = tf.reshape(self.pos_hid_probs, [-1])
            order_indices = tf.to_int64(
                tf.expand_dims(
                    tf.reverse(tf.nn.top_k(
                        flat_hid_probs,
                        k = flat_size
                    )[1], [True]), dim=1))
            values = tf.range(0, flat_size, 1)
            shape = [tf.to_int64(flat_size)]

            flat_hid_prob_ranks = tf.sparse_tensor_to_dense(tf.sparse_reorder(
                tf.SparseTensor(order_indices, values, shape)))
            pos_hid_prob_ranks = tf.to_float(tf.reshape(
                flat_hid_prob_ranks, 
                tf.pack([tf.to_int32(num_samples), hidden_layer_size])))
            
            pos_hid_prob_ranks_scaled = \
                (pos_hid_prob_ranks - tf.reduce_min(pos_hid_prob_ranks)) / (
                    tf.reduce_max(pos_hid_prob_ranks) -
                    tf.reduce_min(pos_hid_prob_ranks)
                )

            self.hidden_activation_biases = \
                tf.pow(pos_hid_prob_ranks_scaled,
                       (1/self.sparsity_target) - 1)

            # Adjust hidden activations by the bias matrix (induce sparsity)
            self.modified_pos_hid_probs = \
                self.sparsity_reg_const * self.hidden_activation_biases + \
                (1 - self.sparsity_reg_const) * self.pos_hid_probs

            # Calculate v+ * s (modified positive associations)
            pos_associations = tf.matmul(
                tf.transpose(self.x), self.modified_pos_hid_probs)
        elif sparse_version == "Lee":
            hid_bias_sparsity_grad = \
                (self.sparsity_target - tf.reduce_mean(
                    self.pos_hid_probs, reduction_indices=0)) * \
                tf.reduce_mean(self.pos_hid_probs *
                               (1-self.pos_hid_probs), reduction_indices=0)
            pos_associations = tf.matmul(
                tf.transpose(self.x), self.pos_hid_probs)

        pos_hid_act = tf.reduce_sum(self.pos_hid_probs, 0)
        pos_vis_act = tf.reduce_sum(self.x, 0)

        # Start negative phase of contrastive divergence
        # Sample the hidden unit states {0,1}
        # Hidden node turns on if
        # P(h=1) > random number uniformally distributed [0, 1)
        self.pos_hid_sample = self.pos_hid_probs > tf.random_uniform(
            tf.shape(self.pos_hid_probs), 0, 1)
        self.pos_hid_sample = tf.to_float(self.pos_hid_sample)
        self.create_variable_summary(self.pos_hid_sample)

        # pos_hid_sample_image = tf.expand_dims(
        #     tf.expand_dims(self.pos_hid_sample, 2),
        #     3, name="pos_hid_sample_image")
        # self.create_image_summary(pos_hid_sample_image)

        # Visible states (probabilities, not binary states)
        # Note: visible nodes store value of the sample
        self.neg_vis_probs = tf.nn.sigmoid(tf.add(
            tf.matmul(self.pos_hid_sample, tf.transpose(self.weights)),
            self.vis_biases))

        rbm_version = self.config["rbm_version"]
        if rbm_version == 'Hinton_2006':
            neg_hid_probs = tf.nn.sigmoid(tf.add(
                tf.matmul(self.neg_vis_probs, self.weights),
                self.hid_biases), name="neg_hid_probs")
            neg_associations = tf.matmul(
                tf.transpose(self.neg_vis_probs), neg_hid_probs)
            neg_hid_act = tf.reduce_sum(neg_hid_probs, 0)
            neg_vis_act = tf.reduce_sum(self.neg_vis_probs, 0)

            # Calculate mean squared error
            mse = tf.reduce_mean(tf.reduce_sum(
                tf.square(self.x - self.neg_vis_probs), 1))
            tf.add_to_collection("losses", mse)

        elif rbm_version in ['Ruslan_new', 'Bengio']:
            # Sample the visible unit states {0,1} using distribution
            # determined by self.neg_vis_probs
            self.neg_vis_sample = tf.to_float(
                self.neg_vis_probs > tf.random_uniform(
                    tf.shape(self.neg_vis_probs), 0, 1))
            neg_hid_probs = tf.nn.sigmoid(tf.add(
                tf.matmul(self.neg_vis_sample, self.weights),
                self.hid_biases), name="neg_hid_probs")
            neg_associations = tf.matmul(tf.transpose(
                self.neg_vis_sample), neg_hid_probs)
            neg_hid_act = tf.reduce_sum(neg_hid_probs, 0)
            neg_vis_act = tf.reduce_sum(self.neg_vis_sample, 0)
            # Calculate mean squared error
            mse = tf.reduce_mean(tf.reduce_sum(
                tf.square(self.x - self.neg_vis_sample), 1))
            tf.add_to_collection("losses", mse)

            if rbm_version == 'Bengio':
                print('Using rbm_verison:  Bengio')
                pos_associations = tf.matmul(
                    tf.transpose(self.x), self.pos_hid_sample)
                pos_hid_act = tf.reduce_sum(self.pos_hid_sample, 0)
            else:
                print('Using rbm_verison:  Ruslan_new')

        # Calculate directions to move weights based on gradient (how to change
        # the weights)

        new_weights_increment = (self.momentum * weights_increment +
                                 (self.learning_rate / num_samples) *
                                 (pos_associations - neg_associations) -
                                 self.weight_decay * self.weights)
        new_vis_bias_increment = (self.momentum * vis_bias_increment +
                                  (self.learning_rate / num_samples) *
                                  (pos_vis_act - neg_vis_act))
        if sparse_version == "Lee":
            new_hid_bias_increment = (self.momentum * hid_bias_increment +
                                      (self.learning_rate / num_samples) *
                                      (pos_hid_act - neg_hid_act) +
                                      self.sparsity_reg_const * 
                                      hid_bias_sparsity_grad)
        else:
            new_hid_bias_increment = (self.momentum * hid_bias_increment +
                                      (self.learning_rate / num_samples) *
                                      (pos_hid_act - neg_hid_act))
        
        update_weights = tf.assign(
            weights_increment, new_weights_increment)
        update_vis_bias = tf.assign(
            vis_bias_increment, new_vis_bias_increment)
        update_hid_bias = tf.assign(
            hid_bias_increment, new_hid_bias_increment)

        self.cost = tf.add_n(tf.get_collection("losses"), "cost")
        tf.scalar_summary("cost", self.cost)

        self.create_variable_summary(weights_increment)
        self.create_variable_summary(vis_bias_increment)
        self.create_variable_summary(hid_bias_increment)

        # Update weights and biases
        self.train_step = [self.weights.assign_add(update_weights),
                           self.vis_biases.assign_add(update_vis_bias),
                           self.hid_biases.assign_add(update_hid_bias),
                           self.cost, self.pos_hid_sample]
