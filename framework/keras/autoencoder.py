from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
from functools import reduce
from operator import mul

import keras.backend as K
import numpy as np
from keras.callbacks import LambdaCallback
from keras.layers import Concatenate, Dense, Input
from keras.models import Model

from framework.keras.callbacks import CallbackManager
from framework.keras.distributions import (Bernoulli, Categorical, Gaussian,
                                           MeanGaussian)

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
        self.reconstructions = None

        self.encoder_model = None
        self.autoencoder_model = None

        self.autoencoder_callbacks = []

        super(BaseAutoencoder, self).__init__(config, restore=restore)

    def setup(self):
        if "input_shape" not in self.config and \
                not isinstance(self.config["input_shape"], tuple):
            raise Exception("Must specify an input shape")

        if "autoencoder_callbacks" in self.config and \
                isinstance(self.config["autoencoder_callbacks"], dict):
            self.autoencoder_callbacks = self.setup_callbacks(
                self.config["autoencoder_callbacks"])

    def train(self, train_dataset,
              epochs=100, batch_size=100, validation_dataset=None):
        """
        Train the autoencoder model

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
        metrics = self.autoencoder_model.evaluate(
            test_dataset.features, test_dataset.features,
            batch_size=batch_size, verbose=0)

        return dict(zip(self.autoencoder_model.metrics_names, metrics))


class DeepAutoencoder(BaseAutoencoder):
    """
    Vanilla (Deterministic) Deep Autoencoder
    """
    def __init__(self, config, restore=False):
        self.continuous = False
        self.deterministic_latent_space = None
        super(DeepAutoencoder, self).__init__(config, restore=restore)

    def build(self):
        """
        Builds the Keras model
        """
        if "continuous" in self.config and self.config["continuous"]:
            self.continuous = True

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
        if "latent_layer" in self.config:
            latent_layer_def = self.config["latent_layer"]
        else:
            raise Exception("Must provide latent layer specification "
                            "for a deterministic autoencoder")

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

            latent_layer_ref, latent_layer_args, latent_layer_kw_args = \
                DefinitionParser.parse_layer_definition(latent_layer_def)
            if latent_layer_ref is not Dense:
                raise Exception("Latent layer must be a Dense layer")
            self.deterministic_latent_space = latent_layer_ref(
                *latent_layer_args, **latent_layer_kw_args)(encoder_output)
            self.encoder_model = Model(self.x, self.deterministic_latent_space)

    def _build_deterministic_decoder(self):
        if "decoder_layers" in self.config:
            decoder_layer_defs = self.config["decoder_layers"]
        else:
            decoder_layer_defs = list(reversed(self.config["encoder_layers"]))

        output_activation = "linear" if self.continuous else "sigmoid"
        with K.name_scope("decoder"):
            decoder_output = self.deterministic_latent_space
            for index, layer_def in enumerate(decoder_layer_defs):
                layer_ref, args, kw_args = \
                    DefinitionParser.parse_layer_definition(layer_def)
                layer = layer_ref(*args, **kw_args)
                decoder_output = layer(decoder_output)

            # Check output shape
            self.reconstructions = Dense(
                self.config["input_shape"][0],
                activation=output_activation,
                name="reconstructions")(decoder_output)


class GenerativeAutoencoder(BaseAutoencoder):
    def __init__(self, config, restore=False):
        self.generator_model = None
        self.generator_callbacks = []
        super(GenerativeAutoencoder, self).__init__(config, restore=restore)

    def _build_stochastic_encoder(self, *latent_dists):
        self.x = Input(shape=self.config["input_shape"], name="x")

        encoder_layer_defs = self.config["encoder_layers"]
        with K.name_scope("encoder"):
            # Encoder
            encoder_output = self.x
            for index, layer_def in enumerate(encoder_layer_defs):
                layer_ref, args, kw_args = \
                    DefinitionParser.parse_layer_definition(layer_def)
                layer = layer_ref(*args, **kw_args)
                encoder_output = layer(encoder_output)

            latent_space = []
            for latent_dist in latent_dists:
                latent_space.append(latent_dist(encoder_output))

            if len(latent_space) == 1:
                return latent_space[0]
            else:
                return latent_space

    def _build_stochastic_decoder(self, latent_space, output_dist):
        if "decoder_layers" in self.config:
            decoder_layer_defs = self.config["decoder_layers"]
        else:
            decoder_layer_defs = list(reversed(self.config["encoder_layers"]))

        # Decoder and Generator Models
        latent_size = int(latent_space.shape[1])
        generator_x = Input(shape=(latent_size,), name="generator_x")

        decoder_output = latent_space
        generator_output = generator_x

        for index, layer_def in enumerate(decoder_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            decoder_output = layer(decoder_output)
            generator_output = layer(generator_output)

        generator_output = output_dist(generator_output, name="reconstructions")
        decoder_output = output_dist(decoder_output)
        self.generator_model = Model(generator_x, generator_output)

        return decoder_output

    def generate(self, z, batch_size=50):
        return self.generator_model.predict(
            z, batch_size=batch_size, verbose=0)


class VariationalAutoencoder(GenerativeAutoencoder):
    def __init__(self, config, restore=False):
        self.z_latent_dist = None
        self.output_dist = None
        self.warmup_beta = None
        self.optimizer = None
        super(VariationalAutoencoder, self).__init__(config, restore=restore)

    def setup(self):
        super(VariationalAutoencoder, self).setup()

        if "latent_size" not in self.config:
            raise Exception("Must specify size of latent space for VAE")
        else:
            self.z_latent_dist = Gaussian(self.config["latent_size"])

        flat_input_size = reduce(mul, self.config["input_shape"])
        if "continuous" in self.config and self.config["continuous"]:
            self.output_dist = MeanGaussian(flat_input_size)
        else:
            self.output_dist = Bernoulli(flat_input_size)

        if "optimizer" not in self.config:
            raise Exception("Must specify optimizer to use for training")
        else:
            self.optimizer = DefinitionParser.parse_optimizer_definition(
                self.config["optimizer"])

    def build(self):
        latent_space = self._build_stochastic_encoder(self.z_latent_dist)
        self.encoder_model = Model(self.x, latent_space)

        self.reconstructions = self._build_stochastic_decoder(
            latent_space, self.output_dist)
        self.autoencoder_model = Model(self.x, self.reconstructions)
        self.add_saveable_model("autoencoder_model", self.autoencoder_model)

        if "n_warmup_epochs" in self.config:
            self.warmup_beta = K.variable(value=0.1, name="warmup_beta")
            self.autoencoder_callbacks.append(LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._warmup(epoch)))

        metrics = [] if "metrics" not in self.config else \
            copy.copy(self.config["metrics"])
        metrics += [self.reconstruction_loss, self.kl_divergence_loss]
        self.autoencoder_model.compile(
            optimizer=self.optimizer,
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

    def reconstruction_loss(self, y_true, _):
        return self.output_dist.negative_log_likelihood(y_true)

    def kl_divergence_loss(self, *_):
        with K.name_scope("kl_divergence_loss"):
            return -0.5 * K.sum(
                1 + self.z_latent_dist.log_var
                - K.exp(self.z_latent_dist.log_var)
                - K.square(self.z_latent_dist.mean), axis=-1)

    def _warmup(self, epoch):
        n_epochs = self.config["n_warmup_epochs"]
        warmup_beta = (epoch / n_epochs) * (epoch <= n_epochs) + \
            1.0 * (epoch > n_epochs)
        K.set_value(self.warmup_beta, warmup_beta)


class AdversarialAutoencoder(GenerativeAutoencoder):
    def __init__(self, config, restore=False):
        self.z_latent_dist = None
        self.z_prior_dist = None
        self.output_dist = None

        self.z_discriminator_model = None
        self.z_adversarial_model = None

        self.ae_optimizer = None
        self.z_disc_optimizer = None

        super(AdversarialAutoencoder, self).__init__(config, restore=restore)

    def setup(self):
        super(AdversarialAutoencoder, self).setup()

        if "z_prior_distribution" not in self.config:
            raise Exception("Must specify prior distribution p(z) for a "
                            "stochastic adversarial autoencoder")
        else:
            self.z_prior_dist = \
                DefinitionParser.parse_distribution_definition(
                    self.config["z_prior_distribution"])

        if "z_latent_distribution" not in self.config:
            raise Exception("Must specify latent distribution q(z|x) for a "
                            "stochastic adversarial autoencoder")
        else:
            self.z_latent_dist = \
                DefinitionParser.parse_distribution_definition(
                    self.config["z_latent_distribution"])

        if "output_distribution" not in self.config:
            raise Exception("Must specify output distribution p(x|z) for a "
                            "stochastic adversarial autoencoder")
        else:
            self.output_dist = \
                DefinitionParser.parse_distribution_definition(
                    self.config["output_distribution"])

        if "autoencoder_optimizer" not in self.config:
            raise Exception("Must specify the optimizer to use for the "
                            "reconstruction training phase")
        else:
            self.ae_optimizer = DefinitionParser.parse_optimizer_definition(
                self.config["autoencoder_optimizer"])

        if "z_discriminator_optimizer" in self.config:
            self.z_disc_optimizer = \
                DefinitionParser.parse_optimizer_definition(
                    self.config["z_discriminator_optimizer"])
        else:
            self.z_disc_optimizer = self.ae_optimizer

    def reconstruction_loss(self, y_true, _):
        return K.mean(self.output_dist.negative_log_likelihood(y_true))

    def _build_discriminator(self, latent_space, layer_defs=None):
        if layer_defs is None:
            discriminator_layer_defs = self.config["z_discriminator_layers"]
        else:
            discriminator_layer_defs = layer_defs

        latent_size = int(latent_space.shape[1])
        discriminator_x = Input(
            shape=(latent_size,), name="discriminator_input")
        with K.name_scope("discriminator"):
            discriminator_output = discriminator_x
            for index, layer_def in enumerate(discriminator_layer_defs):
                layer_ref, args, kw_args = \
                    DefinitionParser.parse_layer_definition(layer_def)
                layer = layer_ref(*args, **kw_args)
                discriminator_output = layer(discriminator_output)

            if discriminator_output.shape[1] != 1:
                discriminator_output = Dense(
                    1, activation="sigmoid", name="discriminator_output")(
                    discriminator_output)

        return Model(discriminator_x, discriminator_output)

    def _setup_callback_manager(self, **train_params):
        do_validation = "validation_dataset" in train_params and \
                        train_params["validation_dataset"] is not None

        callback_manager_params = {
            "epochs": train_params["epochs"],
            "batch_size": train_params["batch_size"],
            "models": {
                "autoencoder": self.autoencoder_model,
                "z_discriminator": self.z_discriminator_model,
                "z_adversarial": self.z_adversarial_model
            }
        }

        if do_validation:
            callback_manager_params["do_validation"] = ["autoencoder"]
            callback_manager_params["validation_data"] = (
                train_params["validation_dataset"].features,
                train_params["validation_dataset"].features
            )

        callback_manager = CallbackManager(**callback_manager_params)

        # Add any additional callbacks for autoencoder model
        if "autoencoder_callbacks" in self.config and \
                isinstance(self.config["autoencoder_callbacks"], dict):
            callback_manager.add_callbacks(
                "autoencoder", self.setup_callbacks(
                    self.config["autoencoder_callbacks"]))

        # Add any additional callbacks for discriminator model
        if "z_discriminator_callbacks" in self.config and \
                isinstance(self.config["z_discriminator_callbacks"], dict):
            callback_manager.add_callbacks(
                "z_discriminator", self.setup_callbacks(
                    self.config["z_discriminator_callbacks"]))

        # Add any additional callbacks for adversarial model
        if "z_adversarial_callbacks" in self.config and \
                isinstance(self.config["z_adversarial_callbacks"], dict):
            callback_manager.add_callbacks(
                "z_adversarial", self.setup_callbacks(
                    self.config["z_adversarial_callbacks"]))

        return callback_manager

    def train(self, train_dataset, epochs=100, batch_size=100,
              validation_dataset=None, verbose=1):
        do_validation = validation_dataset is not None
        n_batches = train_dataset.num_examples // batch_size

        callback_manager = self._setup_callback_manager(
            epochs=epochs, batch_size=batch_size,
            validation_dataset=validation_dataset)
        callback_manager.setup()
        callback_manager.on_train_begin()

        for epoch in range(epochs):
            # Begin epoch
            callback_manager.on_epoch_begin(epoch)

            # Initialize logs to non-empty dictionaries (hence 'name' key)
            ae_epoch_logs = {"name": "autoencoder"}
            disc_epoch_logs = {"name": "z_discriminator"}
            adv_epoch_logs = {"name": "z_adversarial"}

            train_dataset.shuffle()
            for batch in range(n_batches):
                ae_batch_logs = {"batch": batch, "size": batch_size}
                disc_batch_logs = {"batch": batch, "size": batch_size}
                adv_batch_logs = {"batch": batch, "size": batch_size}
                log_string = "Batch {}/{} - ".format(batch + 1, n_batches)

                # Get next batch of samples
                x, _ = train_dataset.next_batch(batch_size)

                # 1) Reconstruction Phase
                callback_manager.on_batch_begin(
                    batch, ae_batch_logs, model_name="autoencoder")
                reconstruction_loss = \
                    self.autoencoder_model.train_on_batch(x, x)
                log_string += "Recon Loss: {:f}".format(reconstruction_loss)
                ae_batch_logs["loss"] = reconstruction_loss
                callback_manager.on_batch_end(
                    batch, ae_batch_logs, model_name="autoencoder")
                if callback_manager.stop_training:
                    break

                # 2) Regularization Phase
                # 2A) Train discriminator
                callback_manager.on_batch_begin(
                    batch, disc_batch_logs, model_name="z_discriminator")
                z_prior = self.z_prior_dist.sample(batch_size)
                z_posterior = self.encoder_model.predict(x)
                d_loss_prior = self.z_discriminator_model.train_on_batch(
                    z_prior, [0.0] * batch_size)
                d_loss_posterior = self.z_discriminator_model.train_on_batch(
                    z_posterior, [1.0] * batch_size)
                d_loss = d_loss_prior + d_loss_posterior
                log_string += "|Disc Loss: {:f}".format(d_loss)
                disc_batch_logs["loss"] = d_loss
                disc_batch_logs["loss_prior"] = d_loss_prior
                disc_batch_logs["loss_posterior"] = d_loss_posterior
                callback_manager.on_batch_end(
                    batch, disc_batch_logs, model_name="z_discriminator")
                if callback_manager.stop_training:
                    break

                # 2B) Train encoder ("generator" of latent space)
                callback_manager.on_batch_begin(
                    batch, adv_batch_logs, model_name="z_adversarial")
                adversarial_loss = self.z_adversarial_model.train_on_batch(
                    x, [0.0] * batch_size)
                log_string += "|Adv Loss: {:f}".format(adversarial_loss)
                adv_batch_logs["loss"] = adversarial_loss
                callback_manager.on_batch_end(
                    batch, adv_batch_logs, model_name="z_adversarial")
                if callback_manager.stop_training:
                    break

                if verbose == 2:
                    print(log_string)

            # Validation step
            if do_validation:
                val_losses = self.evaluate(
                    validation_dataset, batch_size=batch_size)
                ae_epoch_logs["val_loss"] = val_losses["ae_loss"]
                disc_epoch_logs["val_loss"] = val_losses["z_disc_loss"]
                adv_epoch_logs["val_loss"] = val_losses["z_adv_loss"]

            callback_manager.on_epoch_end(
                epoch, ae_epoch_logs, model_name="autoencoder")
            callback_manager.on_epoch_end(
                epoch, disc_epoch_logs, model_name="z_discriminator")
            callback_manager.on_epoch_end(
                epoch, adv_epoch_logs, model_name="z_adversarial")
            if callback_manager.stop_training:
                break

            if verbose >= 1:
                print("Epoch {}/{} - Recon Loss: {:f}|"
                      "Disc Loss: {:f}|Adv Loss: {:f}"
                      .format(epoch + 1, epochs, ae_epoch_logs["loss"],
                              disc_epoch_logs["loss"], adv_epoch_logs["loss"]))
        callback_manager.on_train_end()
        return {k: v.callbacks[-1]
                for k, v in callback_manager.callback_lists.items()}

    def evaluate(self, test_dataset, batch_size=100):
        x = test_dataset.features

        ae_loss = self.autoencoder_model.evaluate(
            x, x, batch_size=batch_size, verbose=0)

        z_prior = self.z_prior_dist.sample(len(x))
        z_posterior = self.encoder_model.predict(x)
        d_loss_prior = self.z_discriminator_model.evaluate(
            z_prior, [0.0] * len(z_prior), batch_size=batch_size, verbose=0)
        d_loss_posterior = self.z_discriminator_model.evaluate(
            z_posterior, [1.0] * len(z_posterior),
            batch_size=batch_size, verbose=0)
        d_loss = d_loss_prior + d_loss_posterior

        adv_loss = self.z_adversarial_model.evaluate(
            x, [0.0] * len(x), batch_size=batch_size, verbose=0)

        total_loss = ae_loss + d_loss + adv_loss

        return {
            "ae_loss": ae_loss,
            "z_disc_loss": d_loss,
            "z_adv_loss": adv_loss,
            "z_disc_loss_prior": d_loss_prior,
            "z_disc_loss_posterior": d_loss_posterior,
            "loss": total_loss
        }


class KadurinAdversarialAutoencoder(AdversarialAutoencoder):
    def __init__(self, config, restore=False):
        self.discriminative_power = None
        self.z_combined_model = None

        super(KadurinAdversarialAutoencoder, self).__init__(
            config, restore=restore)

    def setup(self):
        super(KadurinAdversarialAutoencoder, self).setup()

        if "discriminative_power" in self.config:
            self.discriminative_power = self.config["discriminative_power"]

    def build(self):
        # Build encoder
        latent_space = self._build_stochastic_encoder(self.z_latent_dist)
        self.encoder_model = Model(self.x, latent_space)

        # Build discriminator q(z|x)
        self.z_discriminator_model = self._build_discriminator(latent_space)
        self.z_discriminator_model.name = "discriminator_model"
        self.add_saveable_model("z_discriminator", self.z_discriminator_model)

        # Build decoder
        self.reconstructions = self._build_stochastic_decoder(
            latent_space, self.output_dist)

        # Build combined autoencoder + adversarial model
        self.autoencoder_model = Model(self.x, self.reconstructions)
        self.z_adversarial_model = Model(
            self.x, self.z_discriminator_model(self.encoder_model(self.x)))
        self.z_combined_model = Model(
            self.x, [
                self.reconstructions,
                self.z_discriminator_model(self.encoder_model(self.x))
            ])
        self.add_saveable_model("z_combined", self.z_combined_model)

        # Compile models
        self.z_discriminator_model.compile(optimizer=self.z_disc_optimizer,
                                           loss="binary_crossentropy",
                                           metrics=["accuracy"])
        # Freeze the discriminator model when training
        # the combined model
        self.z_discriminator_model.trainable = False
        self.z_combined_model.compile(
            optimizer=self.ae_optimizer,
            loss={
                "reconstructions": self.reconstruction_loss,
                "discriminator_model": "binary_crossentropy"
            },
            metrics={
                "discriminator_model": "accuracy"
            })

    def _setup_callback_manager(self, **train_params):
        do_validation = "validation_dataset" in train_params and \
                        train_params["validation_dataset"] is not None

        callback_manager_params = {
            "epochs": train_params["epochs"],
            "batch_size": train_params["batch_size"],
            "models": {
                "z_combined": self.z_combined_model,
                "z_discriminator": self.z_discriminator_model,
            }
        }

        callback_manager = CallbackManager(**callback_manager_params)

        # Add any additional callbacks for autoencoder model
        if "z_combined_callbacks" in self.config and \
                isinstance(self.config["z_combined_callbacks"], dict):
            callback_manager.add_callbacks(
                "z_combined", self.setup_callbacks(
                    self.config["z_combined_callbacks"]))

        # Add any additional callbacks for discriminator model
        if "z_discriminator_callbacks" in self.config and \
                isinstance(self.config["z_discriminator_callbacks"], dict):
            callback_manager.add_callbacks(
                "z_discriminator", self.setup_callbacks(
                    self.config["z_discriminator_callbacks"]))

        return callback_manager

    def train(self, train_dataset, epochs=100, batch_size=100,
              validation_dataset=None, verbose=1):
        do_validation = validation_dataset is not None
        n_batches = train_dataset.num_examples // batch_size

        # Setup callback manager
        callback_manager = self._setup_callback_manager(
            epochs=epochs, batch_size=batch_size,
            validation_dataset=validation_dataset)
        callback_manager.setup()
        callback_manager.on_train_begin()

        # At the start of training, train both
        # discriminator and generator for at least one batch
        discriminative_power = None

        for epoch in range(epochs):
            # Begin epoch
            callback_manager.on_epoch_begin(epoch)

            # Initialize logs to non-empty dictionaries (hence 'name' key)
            comb_epoch_logs = {"name": "z_combined"}
            disc_epoch_logs = {"name": "z_discriminator"}

            # Shuffle dataset
            train_dataset.shuffle()

            trained_disc = False
            trained_combined = False
            for batch in range(n_batches):
                # Begin batch
                comb_batch_logs = {"batch": batch, "size": batch_size}
                disc_batch_logs = {"batch": batch, "size": batch_size}
                log_string = "Batch {}/{} - ".format(batch + 1, n_batches)

                # Get next batch of samples
                x, _ = train_dataset.next_batch(batch_size)

                if discriminative_power is None or \
                        discriminative_power >= self.discriminative_power:
                    # Train generator (combined model)
                    callback_manager.on_batch_begin(
                        batch, comb_batch_logs, model_name="z_combined")
                    total_loss, ae_loss, adv_loss, disc_error = \
                        self.z_combined_model.train_on_batch(x, {
                            "reconstructions": x,
                            "discriminator_model": np.array([0.0] * batch_size)
                        })
                    discriminative_power = 1.0 - disc_error
                    log_string += "|Recon. Loss: {:f}".format(ae_loss)
                    log_string += "|Adv. Loss: {:f}".format(adv_loss)
                    comb_batch_logs["loss"] = total_loss
                    comb_batch_logs["ae_loss"] = ae_loss
                    comb_batch_logs["adv_loss"] = adv_loss

                    callback_manager.on_batch_end(
                        batch, comb_batch_logs, model_name="z_combined")
                    if callback_manager.stop_training:
                        break

                    trained_combined = True

                if discriminative_power is None or \
                        discriminative_power < self.discriminative_power:
                    # 2A) Train discriminator
                    callback_manager.on_batch_begin(
                        batch, disc_batch_logs, model_name="z_discriminator")
                    z_prior = self.z_prior_dist.sample(batch_size)
                    z_posterior = self.encoder_model.predict(x)
                    d_loss_prior, _ = self.z_discriminator_model.train_on_batch(
                        z_prior, [0.0] * batch_size)
                    d_loss_posterior, d_acc_posterior = \
                        self.z_discriminator_model.train_on_batch(
                            z_posterior, [1.0] * batch_size)
                    d_loss = d_loss_prior + d_loss_posterior
                    discriminative_power = d_acc_posterior
                    log_string += "|Disc Loss: {:f}".format(d_loss)
                    disc_batch_logs["loss"] = d_loss
                    disc_batch_logs["loss_prior"] = d_loss_prior
                    disc_batch_logs["loss_posterior"] = d_loss_posterior
                    callback_manager.on_batch_end(
                        batch, disc_batch_logs, model_name="z_discriminator")
                    if callback_manager.stop_training:
                        break

                    trained_disc = True

                log_string += "|Disc. Power: {:f}".format(discriminative_power)

                if verbose == 2:
                    print(log_string)

            if not trained_disc:
                z_prior = self.z_prior_dist.sample(train_dataset.num_examples)
                z_posterior = self.encoder_model.predict(
                    train_dataset.features)
                d_loss_prior, _ = self.z_discriminator_model.evaluate(
                    z_prior, np.array([0.0] * train_dataset.num_examples),
                    verbose=0)
                d_loss_posterior, d_acc_posterior = \
                    self.z_discriminator_model.evaluate(
                        z_posterior,
                        np.array([1.0] * train_dataset.num_examples),
                        verbose=0)
                d_loss = d_loss_prior + d_loss_posterior
                disc_epoch_logs["loss"] = d_loss
                discriminative_power = d_acc_posterior

            if not trained_combined:
                total_loss, ae_loss, adv_loss, d_error = \
                    self.z_combined_model.evaluate(
                        train_dataset.features, {
                            "reconstructions": train_dataset.features,
                            "discriminator_model":
                                np.array([0.0] * train_dataset.num_examples)
                        }, verbose=0)
                comb_epoch_logs["loss"] = total_loss
                comb_epoch_logs["ae_loss"] = ae_loss
                comb_epoch_logs["adv_loss"] = adv_loss
                discriminative_power = 1.0 - d_error

            # Validation step
            if do_validation:
                val_losses = self.evaluate(
                    validation_dataset, batch_size=batch_size)
                comb_epoch_logs["val_loss"] = val_losses["ae_loss"]
                disc_epoch_logs["val_loss"] = val_losses["z_disc_loss"]

            callback_manager.on_epoch_end(
                epoch, disc_epoch_logs, model_name="z_discriminator")
            callback_manager.on_epoch_end(
                epoch, comb_epoch_logs, model_name="z_combined")
            if callback_manager.stop_training:
                break

            if verbose >= 1:
                print("Epoch {}/{} - Combined Loss: {:f}|Disc Loss: {:f}"
                      "|Disc. Power {:f}"
                      .format(epoch + 1, epochs, comb_epoch_logs["loss"],
                              disc_epoch_logs["loss"], discriminative_power))
        callback_manager.on_train_end()
        return {k: v.callbacks[-1]
                for k, v in callback_manager.callback_lists.items()}

    def evaluate(self, test_dataset, batch_size=100):
        x = test_dataset.features

        comb_loss, ae_loss, adv_loss, d_error = \
            self.z_combined_model.evaluate(
                x, {
                    "reconstructions": x,
                    "discriminator_model": np.array(
                        [0.0] * test_dataset.num_examples)
                }, verbose=0)

        z_prior = self.z_prior_dist.sample(len(x))
        z_posterior = self.encoder_model.predict(x)
        d_loss_prior, d_acc_prior = self.z_discriminator_model.evaluate(
            z_prior, [0.0] * len(z_prior), batch_size=batch_size, verbose=0)
        d_loss_posterior, d_acc_posterior = \
            self.z_discriminator_model.evaluate(
                z_posterior, [1.0] * len(z_posterior),
                batch_size=batch_size, verbose=0)
        d_loss = d_loss_prior + d_loss_posterior

        total_loss = ae_loss + d_loss + adv_loss

        return {
            "ae_loss": ae_loss,
            "z_disc_loss": d_loss,
            "z_adv_loss": adv_loss,
            "z_disc_loss_prior": d_loss_prior,
            "z_disc_loss_posterior": d_loss_posterior,
            "z_disc_acc_prior": d_acc_prior,
            "z_disc_acc_posterior": d_acc_posterior,
            "loss": total_loss
        }


class UnsupervisedClusteringAdversarialAutoencoder(AdversarialAutoencoder):
    def __init__(self, config, restore=False):
        self.y_latent_dist = None
        self.y_prior_dist = None

        self.y_discriminator_model = None
        self.y_adversarial_model = None

        self.y_disc_optimizer = None

        super(UnsupervisedClusteringAdversarialAutoencoder, self).__init__(
            config, restore=restore)

    def setup(self):
        super(UnsupervisedClusteringAdversarialAutoencoder, self).setup()

        if "n_clusters" not in self.config:
            raise Exception("Must specify number of clusters for an "
                            "unsupervised clustering adversarial autoencoder")
        else:
            n_clusters = self.config["n_clusters"]
            self.y_prior_dist = Categorical(n_clusters)
            self.y_latent_dist = Categorical(n_clusters)

        if "y_discriminator_optimizer" in self.config:
            self.y_disc_optimizer = \
                DefinitionParser.parse_optimizer_definition(
                    self.config["y_discriminator_optimizer"])
        else:
            self.y_disc_optimizer = self.z_disc_optimizer

    def build(self):
        latent_space = self._build_stochastic_encoder(
            self.z_latent_dist, self.y_latent_dist)
        z_latent_space, y_latent_space = latent_space[0], latent_space[1]
        self.encoder_model = Model(self.x, latent_space)
        encoder_model_z = Model(self.x, z_latent_space)
        encoder_model_y = Model(self.x, y_latent_space)

        decoder_input = Concatenate()([z_latent_space, y_latent_space])
        self.reconstructions = self._build_stochastic_decoder(
            decoder_input, self.output_dist)

        self.autoencoder_model = Model(self.x, self.reconstructions)
        self.z_discriminator_model = self._build_discriminator_z(z_latent_space)
        self.y_discriminator_model = self._build_discriminator_y(y_latent_space)

        self.add_saveable_model("autoencoder", self.autoencoder_model)
        self.add_saveable_model("z_discriminator", self.z_discriminator_model)
        self.add_saveable_model("y_discriminator", self.y_discriminator_model)

        self.z_adversarial_model = Model(
            self.x, self.z_discriminator_model(encoder_model_z(self.x)))
        self.y_adversarial_model = Model(
            self.x, self.y_discriminator_model(encoder_model_y(self.x)))

        self.z_discriminator_model.compile(optimizer=self.z_disc_optimizer,
                                           loss="binary_crossentropy")
        self.y_discriminator_model.compile(optimizer=self.y_disc_optimizer,
                                           loss="binary_crossentropy")
        self.autoencoder_model.compile(optimizer=self.ae_optimizer,
                                       loss=self.reconstruction_loss)

        # Freeze the discriminator models when training
        # the adversarial networks
        self.z_discriminator_model.trainable = False
        self.z_adversarial_model.compile(optimizer=self.z_disc_optimizer,
                                         loss="binary_crossentropy")
        self.y_discriminator_model.trainable = False
        self.y_adversarial_model.compile(optimizer=self.y_disc_optimizer,
                                         loss="binary_crossentropy")

    def _build_discriminator_z(self, z_latent_space):
        return self._build_discriminator(z_latent_space)

    def _build_discriminator_y(self, y_latent_space):
        if "y_discriminator_layers" not in self.config:
            return self._build_discriminator(y_latent_space)
        else:
            return self._build_discriminator(
                y_latent_space, self.config["y_discriminator_layers"])

    def _setup_callback_manager(self, **train_params):
        do_validation = "validation_dataset" in train_params and \
                        train_params["validation_dataset"] is not None

        model_names = ["autoencoder", "z_discriminator", "z_adversarial",
                       "y_discriminator", "y_adversarial"]

        callback_manager_params = {
            "epochs": train_params["epochs"],
            "batch_size": train_params["batch_size"],
            "models": dict(zip(model_names, [
                self.autoencoder_model,
                self.z_discriminator_model,
                self.z_adversarial_model,
                self.y_discriminator_model,
                self.y_adversarial_model
            ]))
        }

        if do_validation:
            callback_manager_params["do_validation"] = ["autoencoder"]
            callback_manager_params["validation_data"] = (
                train_params["validation_dataset"].features,
                train_params["validation_dataset"].features
            )

        callback_manager = CallbackManager(**callback_manager_params)

        # Add any additional callbacks for each model
        for name in model_names:
            key = "{}_callbacks".format(name)
            if key in self.config and isinstance(self.config[key], dict):
                callback_manager.add_callbacks(
                    name, self.setup_callbacks(self.config[key]))

        return callback_manager

    def train(self, train_dataset,
              epochs=100, batch_size=100, validation_dataset=None,
              verbose=1):
        do_validation = validation_dataset is not None
        n_batches = train_dataset.num_examples // batch_size

        callback_manager = self._setup_callback_manager(
            epochs=epochs, batch_size=batch_size,
            validation_dataset=validation_dataset)
        callback_manager.setup()
        callback_manager.on_train_begin()

        for epoch in range(epochs):
            # Begin epoch
            callback_manager.on_epoch_begin(epoch)

            # Initialize logs to non-empty dictionaries
            ae_epoch_logs = {"name": "autoencoder"}
            z_disc_epoch_logs = {"name": "z_discriminator"}
            z_adv_epoch_logs = {"name": "z_adversarial"}
            y_disc_epoch_logs = {"name": "y_discriminator"}
            y_adv_epoch_logs = {"name": "y_adversarial"}

            train_dataset.shuffle()
            for batch in range(n_batches):
                ae_batch_logs = {"batch": batch, "size": batch_size}
                z_disc_batch_logs = {"batch": batch, "size": batch_size}
                z_adv_batch_logs = {"batch": batch, "size": batch_size}
                y_disc_batch_logs = {"batch": batch, "size": batch_size}
                y_adv_batch_logs = {"batch": batch, "size": batch_size}

                # Get next batch of samples
                x, _ = train_dataset.next_batch(batch_size)

                # --------------------
                # Reconstruction Phase
                # --------------------
                callback_manager.on_batch_begin(
                    batch, ae_batch_logs, model_name="autoencoder")
                reconstruction_loss = \
                    self.autoencoder_model.train_on_batch(x, x)
                ae_batch_logs["loss"] = reconstruction_loss
                callback_manager.on_batch_end(
                    batch, ae_batch_logs, model_name="autoencoder")
                if callback_manager.stop_training:
                    break

                # --------------------
                # Regularization Phase
                # --------------------

                # Regularize q(z|x)
                callback_manager.on_batch_begin(
                    batch, z_disc_batch_logs, model_name="z_discriminator")

                z_prior = self.z_prior_dist.sample(batch_size)
                z_posterior = self.encoder_model.predict(x)[0]

                z_d_loss_prior = self.z_discriminator_model.train_on_batch(
                    z_prior, [0.0] * batch_size)
                z_d_loss_posterior = self.z_discriminator_model.train_on_batch(
                    z_posterior, [1.0] * batch_size)
                z_d_loss = z_d_loss_prior + z_d_loss_posterior

                z_disc_batch_logs["loss"] = z_d_loss
                z_disc_batch_logs["loss_prior"] = z_d_loss_prior
                z_disc_batch_logs["loss_posterior"] = z_d_loss_posterior
                callback_manager.on_batch_end(
                    batch, z_disc_batch_logs, model_name="z_discriminator")
                if callback_manager.stop_training:
                    break

                # Regularize q(y|x)
                callback_manager.on_batch_begin(
                    batch, y_disc_batch_logs, model_name="y_discriminator")

                y_prior = self.y_prior_dist.sample(batch_size)
                y_posterior = self.encoder_model.predict(x)[1]

                y_d_loss_prior = self.y_discriminator_model.train_on_batch(
                    y_prior, [0.0] * batch_size)
                y_d_loss_posterior = self.y_discriminator_model.train_on_batch(
                    y_posterior, [1.0] * batch_size)
                y_d_loss = y_d_loss_prior + y_d_loss_posterior

                y_disc_batch_logs["loss"] = y_d_loss
                y_disc_batch_logs["loss_prior"] = y_d_loss_prior
                y_disc_batch_logs["loss_posterior"] = y_d_loss_posterior
                callback_manager.on_batch_end(
                    batch, y_disc_batch_logs, model_name="y_discriminator")
                if callback_manager.stop_training:
                    break

                # Train encoder q(z|x)
                callback_manager.on_batch_begin(
                    batch, z_adv_batch_logs, model_name="z_adversarial")

                z_adv_loss = self.z_adversarial_model.train_on_batch(
                    x, [0.0] * batch_size)
                z_adv_batch_logs["loss"] = z_adv_loss
                callback_manager.on_batch_end(
                    batch, z_adv_batch_logs, model_name="z_adversarial")
                if callback_manager.stop_training:
                    break

                # Train encoder q(y|x)
                callback_manager.on_batch_begin(
                    batch, y_adv_batch_logs, model_name="y_adversarial")

                y_adv_loss = self.y_adversarial_model.train_on_batch(
                    x, [0.0] * batch_size)
                y_adv_batch_logs["loss"] = y_adv_loss
                callback_manager.on_batch_end(
                    batch, y_adv_batch_logs, model_name="y_adversarial")
                if callback_manager.stop_training:
                    break

                if verbose == 2:
                    print("Batch {}/{} - Recon Loss: {:f}|"
                          "Z Disc Loss: {:f}|Z Adv Loss: {:f}|"
                          "Y Disc Loss: {:f}|Y Adv Loss: {:f}"
                          .format(batch + 1, n_batches, reconstruction_loss,
                                  z_d_loss, z_adv_loss, y_d_loss, y_adv_loss))

            # Validation step
            if do_validation:
                val_losses = self.evaluate(
                    validation_dataset, batch_size=batch_size)
                ae_epoch_logs["val_loss"] = val_losses["ae_loss"]
                z_disc_epoch_logs["val_loss"] = val_losses["z_disc_loss"]
                z_adv_epoch_logs["val_loss"] = val_losses["z_adv_loss"]
                y_disc_epoch_logs["val_loss"] = val_losses["y_disc_loss"]
                y_adv_epoch_logs["val_loss"] = val_losses["y_adv_loss"]

            callback_manager.on_epoch_end(
                epoch, ae_epoch_logs, model_name="autoencoder")
            callback_manager.on_epoch_end(
                epoch, z_disc_epoch_logs, model_name="z_discriminator")
            callback_manager.on_epoch_end(
                epoch, z_adv_epoch_logs, model_name="z_adversarial")
            callback_manager.on_epoch_end(
                epoch, y_disc_epoch_logs, model_name="y_discriminator")
            callback_manager.on_epoch_end(
                epoch, y_adv_epoch_logs, model_name="y_adversarial")
            if callback_manager.stop_training:
                break

            if verbose >= 1:
                print("Epoch {}/{} - Recon Loss: {:f}|"
                      "Z Disc Loss: {:f}|Z Adv Loss: {:f}|"
                      "Y Disc Loss: {:f}|Y Adv loss: {:f}"
                      .format(epoch + 1, epochs,
                              ae_epoch_logs["loss"],
                              z_disc_epoch_logs["loss"],
                              z_adv_epoch_logs["loss"],
                              y_disc_epoch_logs["loss"],
                              y_adv_epoch_logs["loss"]))
        callback_manager.on_train_end()
        return {k: v.callbacks[-1]
                for k, v in callback_manager.callback_lists.items()}

    def evaluate(self, test_dataset, batch_size=100):
        x = test_dataset.features

        ae_loss = self.autoencoder_model.evaluate(
            x, x, batch_size=batch_size, verbose=0)

        z_prior = self.z_prior_dist.sample(len(x))
        z_posterior = self.encoder_model.predict(x)[0]
        z_d_loss_prior = self.z_discriminator_model.evaluate(
            z_prior, [0.0] * len(z_prior), batch_size=batch_size, verbose=0)
        z_d_loss_posterior = self.z_discriminator_model.evaluate(
            z_posterior, [1.0] * len(z_posterior),
            batch_size=batch_size, verbose=0)
        z_d_loss = z_d_loss_prior + z_d_loss_posterior
        z_adv_loss = self.z_adversarial_model.evaluate(
            x, [0.0] * len(x), batch_size=batch_size, verbose=0)

        y_prior = self.y_prior_dist.sample(len(x))
        y_posterior = self.encoder_model.predict(x)[1]
        y_d_loss_prior = self.y_discriminator_model.evaluate(
            y_prior, [0.0] * len(y_prior), batch_size=batch_size, verbose=0)
        y_d_loss_posterior = self.y_discriminator_model.evaluate(
            y_posterior, [1.0] * len(y_posterior),
            batch_size=batch_size, verbose=0)
        y_d_loss = y_d_loss_prior + y_d_loss_posterior
        y_adv_loss = self.y_adversarial_model.evaluate(
            x, [0.0] * len(x), batch_size=batch_size, verbose=0)

        total_loss = ae_loss + z_d_loss + z_adv_loss + y_d_loss + y_adv_loss

        return {
            "ae_loss": ae_loss,
            "z_disc_loss": z_d_loss,
            "z_adv_loss": z_adv_loss,
            "z_disc_loss_prior": z_d_loss_prior,
            "z_disc_loss_posterior": z_d_loss_posterior,
            "y_disc_loss": y_d_loss,
            "y_adv_loss": y_adv_loss,
            "y_disc_loss_prior": y_d_loss_prior,
            "y_disc_loss_posterior": y_d_loss_posterior,
            "loss": total_loss
        }

    def cluster(self, x, batch_size=100):
        return np.argmax(self.encoder_model.predict(
            x, batch_size=batch_size, verbose=0)[1], axis=-1)
