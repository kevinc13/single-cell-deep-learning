import copy

import numpy as np
from keras.models import Model
from keras.layers import Input

from framework.keras.core import BaseModel
from framework.keras.parser import DefinitionParser


def make_trainable(model, value):
    model.trainable = value
    for l in model.layers: l.trainable = value


class GAN(BaseModel):
    def __init__(self, config, restore=False):
        self.prior_size = None

        self.generator_model = None
        self.discriminator_model = None
        self.gan_model = None

        super(GAN, self).__init__(config, restore=restore,
                                  setup_default_callbacks=False)

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
        # Generator
        generator_x = Input(shape=(self.prior_size,))
        generator_layers = []
        generator_layer_defs = self.config["generator_layers"]
        for index, layer_def in enumerate(generator_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            if index == 0:
                generator_layers.append(layer(generator_x))
            else:
                generator_layers.append(layer(generator_layers[-1]))

        self.generator_model = Model(inputs=generator_x,
                                     outputs=generator_layers[-1])

        # Discriminator

        # Input to discriminator is either a generated or real example
        discriminator_x = Input(shape=(self.config["input_size"],))
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

        output = self.discriminator_model(
            self.generator_model(generator_x))
        self.gan_model = Model(inputs=generator_x,
                                 outputs=output)

        g_optimizer = DefinitionParser.parse_optimizer_definition(
            self.config["generator_optimizer"])
        d_optimizer = DefinitionParser.parse_optimizer_definition(
            self.config["discriminator_optimizer"])

        metrics = [] if "metrics" not in self.config else \
            copy.copy(self.config["metrics"])
        self.generator_model.compile(
            optimizer=g_optimizer,
            loss=self.config["generator_loss"],
            metrics=metrics)
        self.discriminator_model.compile(
            optimizer=d_optimizer,
            loss=self.config["discriminator_loss"],
            metrics=metrics)
        make_trainable(self.discriminator_model, False)

        self.gan_model.compile(optimizer=g_optimizer,
                               loss=self.config["discriminator_loss"],
                               metrics=metrics)

    def train(self, train_dataset, steps,
              n_generator_steps=2, n_discriminator_steps=1,
              batch_size=50, validation_dataset=None, verbose=2):

        losses = {
            "generator_loss": [],
            "discriminator_loss": []
        }

        for step in range(steps):
            # Train discriminator
            make_trainable(self.discriminator_model, True)
            for _ in range(n_discriminator_steps):
                z = np.random.normal(size=(batch_size, self.prior_size))

                real_samples = train_dataset.next_batch(batch_size)
                generated_samples = self.generator_model.predict(z)

                x = np.concatenate((real_samples, generated_samples))
                y = [0.0] * batch_size + [1.0] * batch_size

                d_loss = self.discriminator_model.train_on_batch(x, y)
                losses["discriminator_loss"].append(d_loss)

                if self.model_dir is not None and \
                                step != 0 and step % 100 == 0 and \
                                d_loss < losses["discriminator_loss"][-1]:
                    self.discriminator_model.save_weights(
                        "generator_model.weights.h5")

            # Train generator
            make_trainable(self.discriminator_model, False)
            for _ in range(n_generator_steps):
                z = np.random.normal(size=(batch_size, self.prior_size))

                g_loss = self.gan_model.train_on_batch(
                    z, np.zeros(batch_size))
                losses["generator_loss"].append(g_loss)

                if self.model_dir is not None and \
                                step != 0 and step % 100 == 0 and \
                                g_loss < losses["generator_loss"][-1]:
                    self.generator_model.save_weights(
                        "generator_model.weights.h5")

        return losses["generator_loss"], losses["discriminator_loss"]

    def generate(self, n_samples=1):
        return self.generator_model.predict(
            np.random.normal(size=(n_samples, self.prior_size)))


class WassersteinGAN(GAN):
    def setup(self):
        pass

    def build(self):
        pass

    def train(self):
        pass
