import copy

import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input

from framework.keras.core import BaseModel
from framework.keras.parser import DefinitionParser


def make_trainable(model, value):
    model.trainable = value
    for l in model.layers: l.trainable = value


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


class GAN(BaseModel):
    def __init__(self, config, restore=False):
        self.prior_size = None

        self.generator_model = None
        self.discriminator_model = None
        self.gan_model = None

        super(GAN, self).__init__(config, restore=restore)

    def setup(self):
        if "prior_size" in self.config:
            self.prior_size = self.config["prior_size"]
        else:
            raise Exception("Must specify the size of the "
                            "generator's noise prior")

    def load_weights(self):
        if self.generator_model is not None:
            self.generator_model.load_weights(
                self.model_dir + "/generator_model.weights.h5",
                by_name=True)
        if self.discriminator_model is not None:
            self.discriminator_model.load_weights(
                self.model_dir + "/discriminator_model.weights.h5",
                by_name=True)

    def build(self):
        self._build_generator()
        self._build_discriminator()

        output = self.discriminator_model(
            self.generator_model(self.generator_x))
        self.gan_model = Model(inputs=self.generator_x, outputs=output)

        gan_optimizer = DefinitionParser.parse_optimizer_definition(
            self.config["gan_optimizer"])
        d_optimizer = DefinitionParser.parse_optimizer_definition(
            self.config["discriminator_optimizer"])

        metrics = [] if "metrics" not in self.config else \
            copy.copy(self.config["metrics"])
        self.discriminator_model.compile(
            optimizer=d_optimizer,
            loss=self.config["discriminator_loss"],
            metrics=metrics)

        # Freeze discriminator when training generator
        self.discriminator_model.trainable = False
        self.gan_model.compile(optimizer=gan_optimizer,
                               loss=self.config["discriminator_loss"],
                               metrics=metrics)

    def _build_generator(self):
        # Generator
        self.generator_x = Input(shape=(self.prior_size,))
        generator_layers = []
        generator_layer_defs = self.config["generator_layers"]
        for index, layer_def in enumerate(generator_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            if index == 0:
                generator_layers.append(layer(self.generator_x))
            else:
                generator_layers.append(layer(generator_layers[-1]))

        self.generator_model = Model(inputs=self.generator_x,
                                     outputs=generator_layers[-1])

    def _build_discriminator(self):
        # Input to discriminator is either a generated or real example
        discriminator_x = Input(shape=self.config["input_shape"])
        discriminator_layers = []
        discriminator_layer_defs = self.config["discriminator_layers"]
        for index, layer_def in enumerate(discriminator_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            if index == 0:
                discriminator_layers.append(layer(discriminator_x))
            else:
                discriminator_layers.append(
                    layer(discriminator_layers[-1]))

        self.discriminator_model = Model(inputs=discriminator_x,
                                         outputs=discriminator_layers[-1])

    def train(self, train_dataset, n_iterations,
              n_generator_steps=2, n_discriminator_steps=1,
              batch_size=50, validation_dataset=None, verbose=2):

        losses = {
            "generator": [],
            "discriminator_real": [],
            "discriminator_generated": []
        }

        for iteration in range(n_iterations):
            # Train discriminator
            avg_d_loss_real = 0.0
            avg_d_loss_gen = 0.0
            for _ in range(n_discriminator_steps):
                z = np.random.normal(size=(batch_size, self.prior_size))

                real_samples, _ = train_dataset.next_batch(batch_size)
                generated_samples = self.generator_model.predict(z)

                d_loss_real = self.discriminator_model.train_on_batch(
                    real_samples, [0.0] * batch_size)
                d_loss_gen = self.discriminator_model.train_on_batch(
                    generated_samples, [1.0] * batch_size)

                losses["discriminator_real"].append(d_loss_real)
                losses["discriminator_generated"].append(d_loss_gen)
                avg_d_loss_real += d_loss_real
                avg_d_loss_gen += d_loss_gen

            avg_d_loss_real /= n_discriminator_steps
            avg_d_loss_gen /= n_discriminator_steps

            # Train generator
            avg_g_loss = 0.0
            for _ in range(n_generator_steps):
                z = np.random.normal(size=(batch_size, self.prior_size))

                g_loss = self.gan_model.train_on_batch(
                    z, np.zeros(batch_size))
                losses["generator"].append(g_loss)
                avg_g_loss += g_loss
            avg_g_loss /= n_generator_steps

            if self.model_dir is not None and (
                    losses["generator"][-1] <=
                    min(losses["generator"]) or (
                    losses["discriminator_real"][-1] <=
                    min(losses["discriminator_real"]) and
                    losses["discriminator_generated"][-1] <=
                    min(losses["discriminator_generated"]))):
                self.generator_model.save_weights(
                    "generator_model.weights.h5")
                self.discriminator_model.save_weights(
                    "discriminator_model.weights.h5")

            print("Step {}/{} - Generator. Loss: {:f}|Discriminator Loss "
                  "(Real): {:f}|Discriminator Loss (Gen.): {:f}"
                  .format(iteration + 1, n_iterations, avg_g_loss,
                          avg_d_loss_real, avg_d_loss_gen))

        return losses["generator"],\
            losses["discriminator_real"], losses["discriminator_generated"]

    def generate(self, n_samples=1):
        return self.generator_model.predict(
            np.random.normal(size=(n_samples, self.prior_size)))


class WassersteinGAN(GAN):
    def __init__(self, config, restore=False):
        self.clamp_min = -0.01
        self.clamp_max = 0.01

        super(WassersteinGAN, self).__init__(config, restore=restore)

    def setup(self):
        if "clamp_min" in self.config:
            self.clamp_min = self.config["clamp_min"]
        if "clamp_max" in self.config:
            self.clamp_max = self.config["clamp_max"]

        super(WassersteinGAN, self).setup()

    def build(self):
        self._build_generator()
        self._build_discriminator()

        output = self.discriminator_model(
            self.generator_model(self.generator_x))
        self.gan_model = Model(inputs=self.generator_x, outputs=output)

        g_optimizer = DefinitionParser.parse_optimizer_definition(
            self.config["generator_optimizer"])
        d_optimizer = DefinitionParser.parse_optimizer_definition(
            self.config["discriminator_optimizer"])

        metrics = [] if "metrics" not in self.config else \
            copy.copy(self.config["metrics"])
        self.discriminator_model.compile(
            optimizer=d_optimizer,
            loss=wasserstein_loss,
            metrics=metrics)
        make_trainable(self.discriminator_model, False)

        self.gan_model.compile(optimizer=g_optimizer,
                               loss=wasserstein_loss,
                               metrics=metrics)

    def train(self, train_dataset, n_iterations, batch_size=100,
              validation_dataset=None, verbose=2, **kwargs):

        losses = {
            "generator": [],
            "discriminator_real": [],
            "discriminator_generated": []
        }

        generator_step = 0
        for iteration in range(n_iterations):
            # Train the critic (referred to as discriminator here)
            n_critic_steps = 100 if generator_step < 25 or \
                                    generator_step % 500 == 0 else 5

            avg_d_loss_real = 0.0
            avg_d_loss_gen = 0.0
            make_trainable(self.discriminator_model, True)
            for _ in range(n_critic_steps):
                for l in self.discriminator_model.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w,
                                       self.clamp_min,
                                       self.clamp_max)
                               for w in weights]
                    l.set_weights(weights)

                z = np.random.normal(size=(batch_size, self.prior_size))

                real_samples, _ = train_dataset.next_batch(batch_size)
                generated_samples = self.generator_model.predict(z)

                d_loss_real = self.discriminator_model.train_on_batch(
                    real_samples, [-1.0] * batch_size)
                d_loss_gen = self.discriminator_model.train_on_batch(
                    generated_samples, [1.0] * batch_size)

                losses["discriminator_real"].append(d_loss_real)
                losses["discriminator_generated"].append(d_loss_gen)
                avg_d_loss_real += d_loss_real
                avg_d_loss_gen += d_loss_gen

            avg_d_loss_real /= n_critic_steps
            avg_d_loss_gen /= n_critic_steps

            # Train generator
            make_trainable(self.discriminator_model, False)

            generator_step += 1
            z = np.random.normal(size=(batch_size, self.prior_size))

            g_loss = self.gan_model.train_on_batch(
                z, -np.ones(batch_size))
            losses["generator"].append(g_loss)

            if self.model_dir is not None and (
                    losses["generator"][-1] <=
                    min(losses["generator"]) or
                    losses["discriminator"][-1] <=
                    min(losses["discriminator"])):
                self.generator_model.save_weights(
                    "generator_model.weights.h5")
                self.discriminator_model.save_weights(
                    "discriminator_model.weights.h5")

            print("Step {}/{} - Generator. Loss: {:f}|Discriminator Loss "
                  "(Real): {:f}|Discriminator Loss (Gen.): {:f}"
                  .format(iteration + 1, n_iterations, g_loss, avg_d_loss_real,
                          avg_d_loss_gen))

        return losses["generator"], \
            losses["discriminator_real"], losses["discriminator_generated"]
