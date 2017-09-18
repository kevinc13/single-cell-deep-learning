from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import copy
from operator import mul
from functools import reduce

import keras.backend as K
import numpy as np
from keras.callbacks import LambdaCallback
from keras.layers import (
    Dense, Input, Lambda
)
from keras.losses import mean_squared_error, binary_crossentropy
from keras.models import Model

from framework.keras.callbacks import CallbackManager
from .core import BaseModel
from .parser import DefinitionParser


class BaseAutoencoder(BaseModel):
    def __init__(self, config, restore=False):
        """
        Initializes a deep autoencoder model

        Args:
            config: Dictionary of configuration settings
            restore: Whether to restore a previously saved model
        """
        self.x = None
        self.continuous = False

        self.latent_output = None
        self.reconstructions = None

        self.encoder_model = None
        self.autoencoder_model = None

        self.autoencoder_callbacks = []

        super(BaseAutoencoder, self).__init__(config, restore=restore)

    def setup(self):
        if "continuous" in self.config:
            self.continuous = self.config["continuous"]

        if "input_shape" not in self.config and \
                not isinstance(self.config["input_shape"], tuple):
            raise Exception(
                "Must specify an input shape")

    def train(self, train_dataset,
              epochs=100, batch_size=100, validation_dataset=None):
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
        if "autoencoder_callbacks" in self.config and \
                isinstance(self.config["autoencoder_callbacks"], dict):
            self.autoencoder_callbacks = self.setup_callbacks(
                self.config["autoencoder_callbacks"])

        if validation_dataset is not None:
            return self.autoencoder_model.fit(
                train_dataset.features, train_dataset.features,
                batch_size=batch_size, epochs=epochs, shuffle=True,
                callbacks=self.autoencoder_callbacks, verbose=2,
                validation_data=(validation_dataset.features,
                                 validation_dataset.features))
        else:
            return self.autoencoder_model.fit(
                train_dataset.features, train_dataset.features,
                batch_size=batch_size, epochs=epochs, shuffle=True,
                callbacks=self.autoencoder_callbacks, verbose=2)

    def encode(self, x, batch_size=100):
        return self.encoder_model.predict(
            x, batch_size=batch_size, verbose=0)

    def predict(self, features, batch_size=100):
        return self.autoencoder_model.predict(
            features, batch_size=batch_size, verbose=0)

    def evaluate(self, test_dataset, batch_size=100):
        return self.autoencoder_model.evaluate(
            test_dataset.features, test_dataset.features,
            batch_size=batch_size, verbose=0)


class DeepAutoencoder(BaseAutoencoder):
    """
    Vanilla (Deterministic) Deep Autoencoder
    """
    def build(self):
        """
        Builds the Keras model
        """

        self._build_deterministic_encoder()
        self._build_deterministic_decoder()

        self.autoencoder_model = Model(self.x, self.reconstructions)
        self.add_saveable_model("autoencoder_model", self.autoencoder_model)

        metrics = [] if "metrics" not in self.config else \
            self.config["metrics"]
        loss_fn = "mse" if self.continuous else "binary_crossentropy"
        self.autoencoder_model.compile(
            optimizer=DefinitionParser.parse_optimizer_definition(
                self.config["optimizer"]),
            loss=loss_fn,
            metrics=metrics)

    def _build_deterministic_encoder(self):
        encoder_layer_defs = self.config["encoder_layers"]
        self.x = Input(shape=self.config["input_shape"], name="x")

        # Encoder
        with K.name_scope("encoder"):
            encoder_output = self.x
            for index, layer_def in enumerate(encoder_layer_defs):
                layer_ref, args, kw_args = \
                    DefinitionParser.parse_layer_definition(layer_def)
                layer = layer_ref(*args, **kw_args)
                encoder_output = layer(encoder_output)

            if "latent_layer" in self.config:
                latent_layer_ref, args, kw_args = \
                    DefinitionParser.parse_layer_definition(
                        self.config["latent_layer"])

                if latent_layer_ref != Dense:
                    raise Exception("Latent layer must be a Dense layer")

                encoder_output = latent_layer_ref(
                    *args, name="latent_layer", **kw_args)(encoder_output)

            self.latent_output = encoder_output
            self.encoder_model = Model(self.x, self.latent_output)

    def _build_deterministic_decoder(self):
        if "decoder_layers" in self.config:
            decoder_layer_defs = self.config["decoder_layers"]
        else:
            encoder_layers = self.config["encoder_layers"] \
                if "latent_layer" in self.config else \
                self.config["encoder_layers"][:-1]

            decoder_layer_defs = list(reversed(encoder_layers))

        output_activation = "linear" if self.continuous else "sigmoid"

        with K.name_scope("decoder"):
            decoder_output = self.latent_output
            for index, layer_def in enumerate(decoder_layer_defs):
                layer_ref, args, kw_args = \
                    DefinitionParser.parse_layer_definition(layer_def)
                layer = layer_ref(*args, **kw_args)
                decoder_output = layer(decoder_output)

            # Check output shape
            output_shape = tuple([int(x) for x in decoder_output.shape[1:]])
            if output_shape != self.config["input_shape"] and \
                    len(self.config["input_shape"]) == 1:
                self.reconstructions = Dense(
                    self.config["input_shape"][0],
                    activation=output_activation, name="reconstructions"
                )(decoder_output)
            elif output_shape == self.config["input_shape"]:
                self.reconstructions = decoder_output
            else:
                raise Exception(
                    "Output shape of decoder must match input shape")


class GenerativeAutoencoder(BaseAutoencoder):
    def __init__(self, config, restore=False):
        self.latent_size = None
        self.z_mean = None
        self.z_log_var = None
        self.decoder_log_var = None

        self.generator_model = None

        super(GenerativeAutoencoder, self).__init__(config, restore=restore)

    def setup(self):
        if "latent_size" in self.config:
            self.latent_size = self.config["latent_size"]
        else:
            raise Exception(
                "Must specify size of latent representation in the \
                 model configuration")

        super(GenerativeAutoencoder, self).setup()

    def _build_stochastic_encoder(self):
        encoder_layer_defs = self.config["encoder_layers"]

        self.x = Input(shape=self.config["input_shape"], name="x")
        with K.name_scope("encoder"):
            # Encoder
            encoder_output = self.x
            for index, layer_def in enumerate(encoder_layer_defs):
                layer_ref, args, kw_args = \
                    DefinitionParser.parse_layer_definition(layer_def)
                layer = layer_ref(*args, **kw_args)
                encoder_output = layer(encoder_output)

            # Latent space
            self.z_mean = Dense(self.latent_size, name="z_mean")(
                encoder_output)
            self.z_log_var = Dense(self.latent_size, name="z_log_var")(
                encoder_output)

            self.latent_output = Lambda(self._sample, name="latent_layer")(
                [self.z_mean, self.z_log_var])
            self.encoder_model = Model(self.x, self.latent_output)

    def _build_stochastic_decoder(self):
        if "decoder_layers" in self.config:
            decoder_layer_defs = self.config["decoder_layers"]
        else:
            decoder_layer_defs = list(reversed(self.config["encoder_layers"]))

        # Decoder and Generator Models
        generator_x = Input(shape=(self.latent_size,), name="generator_x")

        decoder_output = self.latent_output
        generator_output = generator_x

        for index, layer_def in enumerate(decoder_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            decoder_output = layer(decoder_output)
            generator_output = layer(generator_output)

        if self.continuous:
            # Output means
            self.reconstructions = Dense(
                self.config["input_shape"][0], activation="linear",
                name="reconstructions"
            )(decoder_output)
            # Output log-variances
            self.decoder_log_var = Dense(
                self.config["input_shape"][0], activation="linear",
                name="decoder_log_var"
            )(decoder_output)

            # Output of generator model
            generator_output = Dense(
                self.config["input_shape"][0], activation="linear",
                name="generator_output"
            )(generator_output)
        else:
            # Decoder output
            self.reconstructions = Dense(
                self.config["input_shape"][0], activation="sigmoid",
                name="reconstructions"
            )(decoder_output)

            # Output of generator model
            generator_output = Dense(
                self.config["input_shape"][0], activation="sigmoid",
                name="generator_output"
            )(generator_output)

        self.generator_model = Model(generator_x, generator_output)

    def _sample(self, args):
        with K.name_scope("sampling"):
            mean, log_var = args
            epsilon = K.random_normal(
                shape=(K.shape(self.x)[0], self.latent_size),
                mean=0., stddev=1.0)
            return mean + K.exp(log_var / 2) * epsilon

    def negative_log_likelihood(self, y_true, y_pred):
        with K.name_scope("negative_log_likelihood"):
            return - K.sum(
                -0.5 * np.log(2 * np.pi) - 0.5 * self.decoder_log_var -
                0.5 * K.square(y_true - y_pred) / K.exp(self.decoder_log_var),
                axis=-1)

    def generate(self, z, batch_size=50):
        return self.generator_model.predict(
            z, batch_size=batch_size, verbose=0)


class VariationalAutoencoder(GenerativeAutoencoder):
    def __init__(self, config, restore=False):
        self.warmup_beta = None
        super(VariationalAutoencoder, self).__init__(config, restore=restore)

    def build(self):
        self._build_stochastic_encoder()
        self._build_stochastic_decoder()

        if "n_warmup_epochs" in self.config:
            self.warmup_beta = K.variable(value=0.1, name="warmup_beta")
            self.autoencoder_callbacks.append(LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._warmup(epoch)))

        self.autoencoder_model = Model(self.x, self.reconstructions)
        self.add_saveable_model("autoencoder_model", self.autoencoder_model)

        metrics = [] if "metrics" not in self.config else \
            copy.copy(self.config["metrics"])
        metrics += [self.reconstruction_loss, self.kl_divergence_loss]
        self.autoencoder_model.compile(
            optimizer=DefinitionParser.parse_optimizer_definition(
                self.config["optimizer"]),
            loss=self._loss,
            metrics=metrics)

    def _loss(self, y_true, y_pred):
        """
        Loss (objective) function of VAE:
            L = E[log(P(x|z))] - KL(Q(z|x)||P(z))
        """
        with K.name_scope("vae_loss"):
            reconstruction_loss = self.reconstruction_loss(y_true, y_pred)
            kl_divergence_loss = self.kl_divergence_loss(y_true, y_pred)

            if hasattr(self, "warmup_beta") and self.warmup_beta is not None:
                return K.mean(reconstruction_loss +
                              self.warmup_beta * kl_divergence_loss)
            else:
                return K.mean(reconstruction_loss + kl_divergence_loss)

    def reconstruction_loss(self, y_true, y_pred):
        if self.continuous:
            # Calculate negative log-likelihood of P(X|z)
            return self.negative_log_likelihood(y_true, y_pred)
        else:
            # Calculate binary cross-entropy
            return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    def kl_divergence_loss(self, *_):
        with K.name_scope("kl_divergence_loss"):
            return -0.5 * K.sum(
                1 + self.z_log_var - K.exp(self.z_log_var) -
                K.square(self.z_mean), axis=-1)

    def _warmup(self, epoch):
        n_epochs = self.config["n_warmup_epochs"]
        warmup_beta = (epoch / n_epochs) * (epoch <= n_epochs) + \
            1.0 * (epoch > n_epochs)
        K.set_value(self.warmup_beta, warmup_beta)


class AdversarialAutoencoder(GenerativeAutoencoder, DeepAutoencoder):
    def __init__(self, config, restore=False):
        self.stochastic = False

        self.discriminator_x = None
        self.discriminator_layers = []
        self.discriminator_model = None
        self.discriminator_callbacks = []

        self.adversarial_model = None

        super(AdversarialAutoencoder, self).__init__(config, restore=restore)

    def setup(self):
        if "stochastic" in self.config:
            self.stochastic = self.config["stochastic"]

        super(AdversarialAutoencoder, self).setup()

    def build(self):
        if self.stochastic:
            self._build_stochastic_encoder()
            self._build_stochastic_decoder()
        else:
            self._build_deterministic_encoder()
            self._build_deterministic_decoder()

        self.autoencoder_model = Model(self.x, self.reconstructions)
        self.add_saveable_model("autoencoder", self.autoencoder_model)

        self._build_discriminator()
        self.adversarial_model = Model(
            self.x, self.discriminator_model(self.encoder_model(self.x)))

        ae_optimizer = DefinitionParser.parse_optimizer_definition(
            self.config["autoencoder_optimizer"])
        if "discriminator_optimizer" in self.config:
            d_optimizer = DefinitionParser.parse_optimizer_definition(
                self.config["discriminator_optimizer"])
        else:
            d_optimizer = ae_optimizer

        self.discriminator_model.compile(optimizer=d_optimizer,
                                         loss="binary_crossentropy")
        self.autoencoder_model.compile(optimizer=ae_optimizer,
                                       loss=self.reconstruction_loss)
        # Freeze the discriminator model when training
        # the adversarial network (enocder + discriminator)
        self.discriminator_model.trainable = False
        self.adversarial_model.compile(optimizer=d_optimizer,
                                       loss="binary_crossentropy")

    def reconstruction_loss(self, y_true, y_pred):
        with K.name_scope("reconstruction_loss"):
            if self.stochastic and self.continuous:
                return K.mean(self.negative_log_likelihood(y_true, y_pred))
            elif not self.stochastic and self.continuous:
                return mean_squared_error(y_true, y_pred)
            else:
                return binary_crossentropy(y_true, y_pred)

    def _build_discriminator(self):
        discriminator_layer_defs = self.config["discriminator_layers"]

        if self.latent_size is None:
            self.latent_size = int(self.latent_output.shape[1])

        self.discriminator_x = Input(
            shape=(self.latent_size,), name="discriminator_x")
        with K.name_scope("discriminator"):
            discriminator_output = self.discriminator_x
            for index, layer_def in enumerate(discriminator_layer_defs):
                layer_ref, args, kw_args = \
                    DefinitionParser.parse_layer_definition(layer_def)
                layer = layer_ref(*args, **kw_args)
                discriminator_output = layer(discriminator_output)

            if discriminator_output.shape[1] != 1:
                discriminator_output = Dense(
                    1, activation="sigmoid", name="discriminator_output")(
                    discriminator_output)

        self.discriminator_model = Model(
            self.discriminator_x, discriminator_output)

    def train(self, train_dataset,
              epochs=100, batch_size=100, validation_dataset=None):

        do_validation = validation_dataset is not None
        n_batches = train_dataset.num_examples // batch_size

        callback_manager = CallbackManager(
            epochs=epochs, batch_size=batch_size,
            models={
                "autoencoder": self.autoencoder_model,
                "discriminator": self.discriminator_model
            },
            do_validation=["autoencoder"],
            validation_data=(
                validation_dataset.features,
                validation_dataset.features))

        # Add any additional callbacks for autoencoder model
        if "autoencoder_callbacks" in self.config and \
                isinstance(self.config["autoencoder_callbacks"], dict):
            callback_manager.add_callbacks(
                "autoencoder", self.setup_callbacks(
                    self.config["autoencoder_callbacks"]))

        # Add any additional callbacks for discriminator model
        if "discriminator_callbacks" in self.config and \
                isinstance(self.config["discriminator_callbacks"], dict):
            callback_manager.add_callbacks(
                "discriminator", self.setup_callbacks(
                    self.config["discriminator_callbacks"]))

        callback_manager.setup()
        callback_manager.on_train_begin()
        for epoch in range(epochs):
            callback_manager.on_epoch_begin(epoch)
            ae_epoch_logs = {}
            disc_epoch_logs = {}

            train_dataset.shuffle()
            for batch in range(n_batches):
                ae_batch_logs = {"batch": batch, "size": batch_size}
                disc_batch_logs = {"batch": batch, "size": batch_size}

                callback_manager.on_batch_begin(
                    batch, ae_batch_logs, model_name="autoencoder")
                callback_manager.on_batch_begin(
                    batch, disc_batch_logs, model_name="discriminator")

                # Get next batch of samples
                x, _ = train_dataset.next_batch(batch_size)

                # 1. Reconstruction Phase
                reconstruction_loss = \
                    self.autoencoder_model.train_on_batch(x, x)
                ae_batch_logs["loss"] = reconstruction_loss
                callback_manager.on_batch_end(
                    batch, ae_batch_logs, model_name="autoencoder")
                if callback_manager.stop_training:
                    break

                # 2. Regularization Phase
                # 2a. Train discriminator
                z_posterior = self.encoder_model.predict(x)
                z_prior = np.random.standard_normal(
                    (batch_size, self.latent_size))

                d_loss_prior = self.discriminator_model.train_on_batch(
                    z_prior, [0.0] * batch_size)
                d_loss_posterior = self.discriminator_model.train_on_batch(
                    z_posterior, [1.0] * batch_size)
                d_loss = d_loss_prior + d_loss_posterior

                disc_batch_logs["loss"] = d_loss
                callback_manager.on_batch_end(
                    batch, disc_batch_logs, model_name="discriminator")
                if callback_manager.stop_training:
                    break

                # 2b. Train encoder ("generator" of latent space)
                adversarial_loss = self.adversarial_model.train_on_batch(
                    x, [0.0] * batch_size)

                print("\nBatch {}/{} - Recon. Loss: {:f}|Disc. Loss: {:f}"
                      .format(batch + 1, n_batches,
                              reconstruction_loss, d_loss))

            if do_validation:
                x_valid = validation_dataset.features
                val_recon_loss = self.autoencoder_model.evaluate(
                    x_valid, x_valid, batch_size=batch_size)

                val_z_prior = np.random.standard_normal((
                    len(x_valid), self.latent_size))
                val_z_posterior = self.encoder_model.predict(
                    x_valid, batch_size=batch_size)

                val_d_loss_prior = self.discriminator_model.evaluate(
                    val_z_prior, [0.0] * len(val_z_prior),
                    batch_size=batch_size)
                val_d_loss_posterior = self.discriminator_model.evaluate(
                    val_z_posterior, [1.0] * len(val_z_posterior),
                    batch_size=batch_size)
                val_d_loss = val_d_loss_prior + val_d_loss_posterior

                val_adversarial_loss = self.adversarial_model.evaluate(
                    x_valid, np.zeros(len(x_valid)))

                ae_epoch_logs["val_loss"] = val_recon_loss
                disc_epoch_logs["val_loss"] = val_d_loss

            callback_manager.on_epoch_end(
                epoch, ae_epoch_logs, model_name="autoencoder")
            callback_manager.on_epoch_end(
                epoch, disc_epoch_logs, model_name="discriminator")

            if callback_manager.stop_training:
                break

            print("\nEpoch {}/{} - Recon. Loss: {:f}|Disc. Loss: {:f}"
                  .format(epoch + 1, epochs,
                          ae_epoch_logs["loss"],
                          disc_epoch_logs["loss"]))

        callback_manager.on_train_end()
        return [l[-1] for l in callback_manager.callback_lists]

