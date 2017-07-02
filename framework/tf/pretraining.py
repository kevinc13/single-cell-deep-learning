import tensorflow as tf
import os

from ..common.dataset import DataSet
from .core import SequentialModel
from .autoencoder import DeepAutoencoder as DA


class StackedAutoencoderPretraining(object):
    def __init__(self, name, config, model_dir=None):
        self.name = name
        self.config = config
        self.model_dir = model_dir

        self.encoder_weights = []
        self.encoder_biases = []

        self.decoder_weights = []
        self.decoder_biases = []

    def train(self, train_dataset, validation_dataset=None,
              num_epochs=100, batch_size=100,
              epoch_log_verbosity=1, batch_log_verbosity=None):

        autoencoders = []
        for i, size in enumerate(self.config["encoder_hidden_layers"]):
            autoencoder_name = "{}_AE_{}".format(self.name, i + 1)

            config = self.config.copy()

            if i > 0:
                config["input_size"] = \
                    self.config["encoder_hidden_layers"][i - 1]

            config["encoder_hidden_layers"] = [size]

            if self.model_dir is not None:
                autoencoder_model_dir = self.model_dir + \
                    "/{}".format(autoencoder_name)
                if not os.path.exists(autoencoder_model_dir):
                    os.makedirs(autoencoder_model_dir)
                    os.makedirs(autoencoder_model_dir + "/checkpoints")
                    os.makedirs(autoencoder_model_dir + "/tensorboard")
            else:
                autoencoder_model_dir = None

            with tf.Session() as sess:
                autoencoders.append(self.config["autoencoder_class"](
                    autoencoder_name, sess,
                    config, model_dir=autoencoder_model_dir))

                autoencoders[i].fit(train_dataset,
                                    validation_dataset=validation_dataset,
                                    num_epochs=num_epochs,
                                    batch_size=batch_size,
                                    epoch_log_verbosity=epoch_log_verbosity,
                                    batch_log_verbosity=batch_log_verbosity)

                new_train_data = sess.run(
                    autoencoders[i].outputs[0],
                    feed_dict={autoencoders[i].x: train_dataset.features}
                )

                new_validation_data = sess.run(
                    autoencoders[i].outputs[0],
                    feed_dict={autoencoders[i].x: validation_dataset.features}
                )

                # Save encoder and decoder weights and biases
                # for single hidden layer autoencoder
                self.encoder_weights.append(
                    sess.run(autoencoders[i].layers[0].weights))
                self.encoder_biases.append(
                    sess.run(autoencoders[i].layers[0].biases))

                self.decoder_weights.append(
                    sess.run(autoencoders[i].layers[1].weights))
                self.decoder_biases.append(
                    sess.run(autoencoders[i].layers[1].biases))

            # Reset TF graph
            tf.reset_default_graph()

            # Create DataSet objects of new data
            train_dataset = DataSet(new_train_data, new_train_data)
            validation_dataset = DataSet(
                new_validation_data, new_validation_data)

    def get_pretraining_weights_and_biases(self):
        pretrain_weights = self.encoder_weights + \
            list(reversed(self.decoder_weights))
        pretrain_biases = self.encoder_biases + \
            list(reversed(self.decoder_biases))
        return pretrain_weights, pretrain_biases


class StackedRBMPretraining(object):
    def __init__(self, name, config, model_dir=None):
        self.name = name
        self.config = config
        self.model_dir = model_dir

        self.encoder_weights = []
        self.encoder_biases = []

        self.decoder_weights = []
        self.decoder_biases = []

    def train(self, train_dataset, validation_dataset=None,
              num_epochs=100, batch_size=100,
              epoch_log_verbosity=1, batch_log_verbosity=None):

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        rbms = []
        for i, size in enumerate(self.config["encoder_hidden_layers"]):
            rbm_name = "{}_RBM_{}".format(self.name, i + 1)

            config = self.config.copy()
            del config["encoder_hidden_layers"]

            if i > 0:
                config["input_size"] = \
                    self.config["encoder_hidden_layers"][i - 1]

            config["hidden_layer_size"] = size

            if self.model_dir is not None:
                rbm_model_dir = self.model_dir + "/{}".format(rbm_name)
                if not os.path.exists(rbm_model_dir):
                    os.makedirs(rbm_model_dir)
                    os.makedirs(rbm_model_dir + "/checkpoints")
                    os.makedirs(rbm_model_dir + "/tensorboard")
            else:
                rbm_model_dir = None

            if isinstance(self.config["rbm_class"], list):
                rbm_class = self.config["rbm_class"][i]
            else:
                rbm_class = self.config["rbm_class"]

            with tf.Session() as sess:
                rbms.append(rbm_class(rbm_name, sess, config, 
                                      model_dir=rbm_model_dir))

                rbms[i].fit(self.train_dataset,
                            validation_dataset=self.validation_dataset,
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            epoch_log_verbosity=epoch_log_verbosity,
                            batch_log_verbosity=batch_log_verbosity)

                # Get RBM hidden layer to use as input to next RBM
                if self.config["rbm_version"] == 'Hinton_2006':
                    # use probabilities
                    new_train_data = sess.run(
                        rbms[i].pos_hid_probs,
                        feed_dict={rbms[i].x: self.train_dataset.features}
                    )

                    new_validation_data = sess.run(
                        rbms[i].pos_hid_probs,
                        feed_dict={rbms[i].x: self.validation_dataset.features}
                    )
                elif self.config["rbm_version"] in ['Ruslan_new', 'Bengio']:
                    # use binary sample
                    new_train_data = sess.run(
                        rbms[i].pos_hid_sample,
                        feed_dict={rbms[i].x: self.train_dataset.features}
                    )

                    new_validation_data = sess.run(
                        rbms[i].pos_hid_sample,
                        feed_dict={rbms[i].x: self.validation_dataset.features}
                    )

                # Save encoder and decoder weights and biases
                # for single hidden layer autoencoder
                weights = sess.run(rbms[i].weights)
                self.encoder_weights.append(weights)
                self.encoder_biases.append(
                    sess.run(rbms[i].hid_biases))

                self.decoder_weights.append(weights.T)
                self.decoder_biases.append(
                    sess.run(rbms[i].vis_biases))

            # Reset TF graph
            tf.reset_default_graph()

            # Create DataSet objects of new data
            self.train_dataset = DataSet(new_train_data, new_train_data)
            self.validation_dataset = DataSet(
                new_validation_data, new_validation_data)

    def get_pretraining_weights_and_biases(self):
        pretrain_weights = self.encoder_weights + \
            list(reversed(self.decoder_weights))
        pretrain_biases = self.encoder_biases + \
            list(reversed(self.decoder_biases))
        return pretrain_weights, pretrain_biases
