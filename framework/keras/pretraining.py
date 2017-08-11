from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import copy
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from ..common.dataset import Dataset

from .parser import DefinitionParser

class StackedAutoencoderPretraining(object):
    def __init__(self, autoencoder_class, config):
        self.config = config
        self.autoencoder_class = autoencoder_class

        self.encoder_weights_and_biases = []
        self.decoder_weights_and_biases = []

    def reset_session(self):
        K.clear_session()
        self.sess = tf.Session()
        K.set_session(self.sess)

    def train(self, train_dataset,
              batch_size=100, epochs=50,
              validation_dataset=None):        
        encoder_layer_defs = self.config["encoder_layers"]

        autoencoders = []
        previous_output_size = self.config["input_size"]

        new_train_dataset = train_dataset
        new_valid_dataset = validation_dataset

        for i, layer_def in enumerate(encoder_layer_defs):
            if layer_def.split(":")[0] == "Dense":
                self.reset_session()

                config = self.config.copy()

                config["name"] = "PretrainingAutoencoder_{}".format(i)
                config["input_size"] = previous_output_size 
                config["encoder_layers"] = [layer_def]

                autoencoders.append(self.autoencoder_class(config))
                autoencoders[-1].train(new_train_dataset,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       validation_dataset=new_valid_dataset)

                self.encoder_weights_and_biases.append(
                    autoencoders[-1].encoder_model.layers[0].get_weights())
                self.decoder_weights_and_biases.append(
                    autoencoders[-1].keras_model.layers[-1].get_weights())

                new_train_data = autoencoders[-1].encode(
                    new_train_dataset.features)
                new_valid_data = autoencoders[-1].encode(
                    new_valid_dataset.features)

                new_train_dataset = Dataset(new_train_data, new_train_data)
                new_valid_dataset = Dataset(
                    new_valid_data, new_valid_data)

                previous_output_size = int(layer_def.split(":")[1])

    def get_pretraining_weights_and_biases(self):
        return self.encoder_weights_and_biases + \
            list(reversed(self.decoder_weights_and_biases))
