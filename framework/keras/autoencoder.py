from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import copy

import keras.backend as K
import numpy as np
from keras.callbacks import LambdaCallback
from keras.layers import (
    Dense, Input, Lambda
)
from keras.layers.noise import GaussianNoise
from keras.models import Model

from .core import BaseModel
from .parser import DefinitionParser


class DeepAutoencoder(BaseModel):
    """
    Deep Autoencoder
    """
    def __init__(self, config, restore=False):
        """
        Initializes a deep autoencoder model

        Args:
            config: Dictionary of configuration settings
            restore: Whether to restore a previously saved model
        """
        self.x = None
        self.bernoulli = False
        self.encoder_layers = []
        self.decoder_layers = []
        self.reconstructions = None

        self.encoder_model = None
        self.keras_model = None

        super(DeepAutoencoder, self).__init__(config, restore=restore)

    def setup(self):
        if "bernoulli" in self.config:
            self.bernoulli = self.config["bernoulli"]
        else:
            raise Exception(
                "Must specify whether the model output follows a Bernoulli \
                distribution or continuous (Gaussian) distribution")

    def build(self):
        """
        Builds the Keras model
        """
        encoder_layer_defs = self.config["encoder_layers"]

        if "decoder_layers" in self.config:
            decoder_layer_defs = self.config["decoder_layers"]
        else:
            decoder_layer_defs = list(reversed(encoder_layer_defs[:-1]))

        self.x = Input(
            shape=(self.config["input_size"],))

        # Encoder
        self.encoder_layers = []
        for index, layer_def in enumerate(encoder_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            if index == 0:
                self.encoder_layers.append(layer(self.x))
            else:
                self.encoder_layers.append(layer(self.encoder_layers[-1]))

        self.decoder_layers = []

        for index, layer_def in enumerate(decoder_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            if index == 0:
                self.decoder_layers.append(layer(self.encoder_layers[-1]))
            else:
                self.decoder_layers.append(layer(self.decoder_layers[-1]))

        self.encoder_model = Model(self.x, self.encoder_layers[-1])

        last_layer = self.decoder_layers[-1] \
            if len(self.decoder_layers) > 0 else self.encoder_layers[-1]

        if self.bernoulli:
            self.reconstructions = Dense(
                self.config["input_size"], activation="sigmoid"
            )(last_layer)
        else:
            self.reconstructions = Dense(
                self.config["input_size"], activation="linear"
            )(last_layer)

        self.keras_model = Model(self.x, self.reconstructions)

        metrics = [] if "metrics" not in self.config else \
            self.config["metrics"]
        self.keras_model.compile(
            optimizer=DefinitionParser.parse_optimizer_definition(
                self.config["optimizer"]),
            loss=self.config["loss"],
            metrics=metrics)

    def train(self, train_dataset,
              epochs=50, batch_size=100, validation_dataset=None):
        """
        Train the model

        Args:
            train_dataset: Dataset of training examples
            epochs: Maximum number of training epochs
            batch_size: Mini-batch size to use
            validation_dataset:
                Optional Dataset of validation examples

        Returns:
            Keras History object
        """
        if validation_dataset is not None:
            return self.keras_model.fit(train_dataset.features,
                                        train_dataset.features,
                                        batch_size=batch_size, epochs=epochs,
                                        shuffle=True, callbacks=self.callbacks,
                                        verbose=2, validation_data=(
                                            validation_dataset.features,
                                            validation_dataset.features))
        else:
            return self.keras_model.fit(train_dataset.features,
                                        train_dataset.features,
                                        batch_size=batch_size, epochs=epochs,
                                        shuffle=True, callbacks=self.callbacks,
                                        verbose=2)

    def encode(self, x, batch_size=100):
        return self.encoder_model.predict(
            x, batch_size=batch_size, verbose=0)

    def predict(self, features, batch_size=100):
        return self.keras_model.predict(
            features, batch_size=batch_size, verbose=0)

    def evaluate(self, test_dataset, batch_size=100):
        return self.keras_model.evaluate(
            test_dataset.features, test_dataset.features,
            batch_size=batch_size, verbose=0)


class DenoisingAutoencoder(DeepAutoencoder):
    def __init__(self, config, restore=False):
        self.x_noise = None
        self.noise_stdev = 0.0
        super(DenoisingAutoencoder, self).__init__(config, restore=restore)

    def setup(self):
        if "noise_stdev" in self.config:
            self.noise_stdev = self.config["noise_stdev"]
        else:
            raise Exception(
                "Must specify std. deviation of Gaussian noise distribution")

        super(DenoisingAutoencoder, self).setup()

    def build(self):
        encoder_layer_defs = self.config["encoder_layers"]

        if "decoder_layers" in self.config:
            decoder_layer_defs = self.config["decoder_layers"]
        else:
            decoder_layer_defs = list(reversed(encoder_layer_defs[:-1]))

        self.x = Input(
            shape=(self.config["input_size"],))
        self.x_noise = GaussianNoise(self.config["noise_stdev"])(self.x)

        # Encoder
        self.encoder_layers = []
        for index, layer_def in enumerate(encoder_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            if index == 0:
                self.encoder_layers.append(layer(self.x_noise))
            else:
                self.encoder_layers.append(layer(self.encoder_layers[-1]))

        self.decoder_layers = []

        for index, layer_def in enumerate(decoder_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            if index == 0:
                self.decoder_layers.append(layer(self.encoder_layers[-1]))
            else:
                self.decoder_layers.append(layer(self.decoder_layers[-1]))

        self.encoder_model = Model(self.x_noise, self.encoder_layers[-1])
        self.reconstructions = Dense(
            self.config["input_size"], activation="sigmoid"
        )(self.decoder_layers[-1])
        self.keras_model = Model(self.x_noise, self.reconstructions)

        self.keras_model.compile(
            optimizer=DefinitionParser.parse_optimizer_definition(
                self.config["optimizer"]),
            loss=self.config["loss"],
            metrics=self.config["metrics"])


class VariationalAutoencoder(DeepAutoencoder):
    def __init__(self, config, restore=False):
        self.latent_size = None
        self.warmup_beta = None

        self.generator_layers = []

        self.z_mean = None
        self.z_log_var = None
        self.z = None

        self.generator_x = None
        self.generator_model = None

        super(VariationalAutoencoder, self).__init__(config, restore=restore)

    def setup(self):
        if "latent_size" in self.config:
            self.latent_size = self.config["latent_size"]
        else:
            raise Exception(
                "Must specify size of latent representation in the \
                variational autoencoder model configuration")

        super(VariationalAutoencoder, self).setup()

    def build(self):
        encoder_layer_defs = self.config["encoder_layers"]

        if "decoder_layers" in self.config:
            decoder_layer_defs = self.config["decoder_layers"]
        else:
            decoder_layer_defs = list(reversed(encoder_layer_defs))

        self.x = Input(
            shape=(self.config["input_size"],))

        if "n_warmup_epochs" in self.config:
            self.warmup_beta = K.variable(value=0.1)
            self.callbacks.append(LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._warmup(epoch)))

        # Encoder
        for index, layer_def in enumerate(encoder_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            if index == 0:
                self.encoder_layers.append(layer(self.x))
            else:
                self.encoder_layers.append(layer(self.encoder_layers[-1]))

        # Latent space
        self.z_mean = Dense(self.latent_size)(self.encoder_layers[-1])
        self.z_log_var = Dense(self.latent_size)(self.encoder_layers[-1])

        self.z = Lambda(self._sample)([self.z_mean, self.z_log_var])

        self.encoder_model = Model(self.x, self.z)

        # Decoder and Generator Models
        self.generator_x = Input(shape=(self.latent_size,))

        for index, layer_def in enumerate(decoder_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            if index == 0:
                self.decoder_layers.append(layer(self.z))
                self.generator_layers.append(layer(self.generator_x))
            else:
                self.decoder_layers.append(layer(self.decoder_layers[-1]))
                self.generator_layers.append(layer(self.generator_layers[-1]))

        if self.bernoulli:
            # Decoder output
            self.reconstructions = Dense(
                self.config["input_size"], activation="sigmoid"
            )(self.decoder_layers[-1])

            # Output of generator model
            self.generator_output = Dense(
                self.config["input_size"], activation="sigmoid"
            )(self.generator_layers[-1])
        else:
            # Output means
            self.reconstructions = Dense(
                self.config["input_size"], activation="linear"
            )(self.decoder_layers[-1])
            # Output log-variances
            self.decoder_log_var = Dense(
                self.config["input_size"], activation="linear"
            )(self.decoder_layers[-1])

            # Output of generator model
            self.generator_output = Dense(
                self.config["input_size"], activation="linear"
            )(self.generator_layers[-1])

        self.keras_model = Model(self.x, self.reconstructions)
        self.generator_model = Model(self.generator_x, self.generator_output)

        metrics = [] if "metrics" not in self.config else \
            copy.copy(self.config["metrics"])
        metrics += [self.reconstruction_loss, self.kl_divergence_loss]
        self.keras_model.compile(
            optimizer=DefinitionParser.parse_optimizer_definition(
                self.config["optimizer"]),
            loss=self._loss,
            metrics=metrics)

    def _sample(self, args):
        mean, log_var = args
        epsilon = K.random_normal(
            shape=(K.shape(self.x)[0], self.latent_size), mean=0., stddev=1.0)
        return mean + K.exp(log_var / 2) * epsilon

    def _loss(self, y_true, y_pred):
        """
        Loss (objective) function of VAE:
            L = E[log(P(x|z))] - KL(Q(z|x)||P(z))
        """
        reconstruction_loss = self.reconstruction_loss(y_true, y_pred)
        kl_divergence_loss = self.kl_divergence_loss(y_true, y_pred)

        if hasattr(self, "warmup_beta") and self.warmup_beta is not None:
            return K.mean(
                reconstruction_loss + self.warmup_beta * kl_divergence_loss)
        else:
            return K.mean(reconstruction_loss + kl_divergence_loss)

    def reconstruction_loss(self, y_true, y_pred):
        if self.bernoulli:
            # Calculate binary cross-entropy
            return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
        else:
            # Calculate negative log-likelihood of P(X|z)
            return - K.sum(
                -0.5 * np.log(2 * np.pi) - 0.5 * self.decoder_log_var -
                0.5 * K.square(y_true - y_pred) / K.exp(self.decoder_log_var),
                axis=-1)

    def kl_divergence_loss(self, *_):
        return -0.5 * K.sum(
            1 + self.z_log_var - K.exp(self.z_log_var) - K.square(self.z_mean),
            axis=-1)

    def _warmup(self, epoch):
        n_epochs = self.config["n_warmup_epochs"]
        warmup_beta = (epoch / n_epochs) * (epoch <= n_epochs) + \
            1.0 * (epoch > n_epochs)
        K.set_value(self.warmup_beta, warmup_beta)

    def generate(self, z, batch_size=50):
        return self.generator_model.predict(
            z, batch_size=batch_size, verbose=0)
