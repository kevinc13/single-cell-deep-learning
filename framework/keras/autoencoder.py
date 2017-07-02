import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import (
    Dense, Activation, Input, Lambda
)
from keras.objectives import binary_crossentropy

from .core import BaseModel
from .parser import DefinitionParser

class DeepAutoencoder(BaseModel):

    def setup(self):
        pass

    def build(self):
        self.keras_model = Sequential()

        decoder_hidden_layers = list(reversed(
            self.config["encoder_hidden_layers"][:-1]))
        all_hidden_layers = self.config["encoder_hidden_layers"] + \
            decoder_hidden_layers

        if type(self.config["activations"]) == list:
            assert((len(all_hidden_layers) + 1) \
                == len(self.config["activations"]))
            self.activations = self.config["activations"]
        else:
            self.activations = [self.config["activations"] \
                for i in range(0, (len(all_hidden_layers) + 1))]

        for index, size in enumerate(all_hidden_layers):
            if index == 0:
                self.keras_model.add(Dense(
                    size, activation=self.activations[index],
                    input_shape=(self.config["input_size"],)))
            else:
                self.keras_model.add(Dense(
                    size, activation=self.activations[index]))

        self.keras_model.add(Dense(self.config["input_size"]))
        self.keras_model.add(Activation(self.activations[-1]))

        self.keras_model.compile(
            optimizer=DefinitionParser.parse_optimizer_definition(
                self.config["optimizer"]),
            loss=self.config["loss"],
            metrics=self.config["metrics"])

    def train(self, train_dataset,
              batch_size=100, epochs=100,
              validation_dataset=None):
        self.keras_model.fit(train_dataset.features, train_dataset.features,
                             batch_size=batch_size, epochs=epochs,
                             callbacks=self.callbacks, verbose=2,
                             validation_data=(
                                validation_dataset.features,
                                validation_dataset.features
                            ))

    def predict(self, features, batch_size=100):
        return self.keras_model.predict(
            features, batch_size=batch_size, verbose=1)

    def evaluate(self, test_dataset, batch_size=100):
        return self.keras_model.evaluate(
            test_dataset.features, test_dataset.features,
            batch_size=batch_size, verbose=1)


class VariationalAutoencoder(BaseModel):

    def setup(self):
        if "latent_size" in self.config:
            self.latent_size = self.config["latent_size"]
        else:
            raise Exception(
                "Must specify size of latent representation in the \
                variational autoencoder model configuration")

    def build(self):
        encoder_layer_defs = self.config["encoder_layers"]

        if "decoder_layers" in self.config:
            decoder_layer_defs = self.config["decoder_layers"]
        else:
            decoder_layer_defs = list(reversed(encoder_layer_defs))
        
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

        # Latent space
        self.z_mean = Dense(self.latent_size)(self.encoder_layers[-1])
        self.z_log_var = Dense(self.latent_size)(self.encoder_layers[-1])

        self.z = Lambda(self._sample)([self.z_mean, self.z_log_var])

        self.encoder_model = Model(self.x, self.z)

        # Decoder and Generator Models
        self.decoder_layers = []
        self.generator_layers = []

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

        self.reconstructions = Dense(
                self.config["input_size"], activation="sigmoid"
            )(self.decoder_layers[-1])
        self.generator_output = Dense(
                self.config["input_size"], activation="sigmoid"
            )(self.generator_layers[-1])

        self.keras_model = Model(self.x, self.reconstructions)
        self.generator_model = Model(self.generator_x, self.generator_output)

        self.keras_model.compile(
            optimizer=DefinitionParser.parse_optimizer_definition(
                self.config["optimizer"]),
            loss=self._loss,
            metrics=self.config["metrics"])

        print(self.keras_model.summary())

    def _sample(self, args):
        mean, log_var = args
        epsilon = K.random_normal(
            shape=(K.shape(self.x)[0], self.latent_size), mean=0., stddev=1)
        return mean + K.exp(self.z_log_var / 2) * epsilon

    def _loss(self, y_true, y_pred):
        reconstruction_error = binary_crossentropy(y_pred, y_true)
        kl_divergence = - 0.5 * K.mean(
            1 + self.z_log_var - K.exp(self.z_log_var) - K.square(self.z_mean),
            axis=-1)
        return reconstruction_error + kl_divergence

    def train(self, train_dataset, 
              epochs=50, batch_size=100,
              validation_dataset=None):
        self.keras_model.fit(train_dataset.features, train_dataset.features,
                             batch_size=batch_size, epochs=epochs,
                             shuffle=True, callbacks=self.callbacks,
                             verbose=2, validation_data=(
                                validation_dataset.features,
                                validation_dataset.features
                            ))

    def encode(self, x, batch_size=100):
        return self.encoder_model.predict(
            x, batch_size=batch_size, verbose=1)

    def generate(self, z, batch_size=100):
        return self.generator_model.predict(
            z, batch_size=batch_size, verbose=1)

