from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
import copy

import keras.backend as K
from keras.models import Sequential

from .core import BaseModel
from .parser import DefinitionParser

class DeepNeuralNetwork(BaseModel):

    def build(self):
        layer_defs = self.config["layers"]

        self.keras_model = Sequential()
        for index, layer_def in enumerate(layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            if index == 0 and \
                    "input_shape" not in kw_args and \
                    "input_shape" in self.config:
                kw_args["input_shape"] = self.config["input_shape"]
            layer = layer_ref(*args, **kw_args)

            self.keras_model.add(layer)

        metrics = [] if "metrics" not in self.config else \
            self.config["metrics"]
        self.keras_model.compile(
            optimizer=DefinitionParser.parse_optimizer_definition(
                self.config["optimizer"]),
            loss=self.config["loss"],
            metrics=metrics)

    def train(self, train_dataset,
              batch_size=100, epochs=100,
              validation_dataset=None):
        if validation_dataset is not None:
            self.keras_model.fit(train_dataset.features,
                                 train_dataset.labels,
                                 batch_size=batch_size, epochs=epochs,
                                 shuffle=True, callbacks=self.callbacks,
                                 verbose=2, validation_data=(
                                    validation_dataset.features,
                                    validation_dataset.labels
                                ))
        else:
            self.keras_model.fit(train_dataset.features,
                                 train_dataset.labels,
                                 batch_size=batch_size, epochs=epochs,
                                 shuffle=True, callbacks=self.callbacks,
                                 verbose=2)

    def predict(self, features, batch_size=100):
        return self.keras_model.predict(
            features, batch_size=batch_size, verbose=0)

    def evaluate(self, test_dataset, batch_size=100):
        return self.keras_model.evaluate(
            test_dataset.features, test_dataset.labels,
            batch_size=batch_size, verbose=0)