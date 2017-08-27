from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import copy
import json

import tensorflow as tf
from keras import backend as K
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard
)

from .callbacks import (
    TimeLogger, FileLogger,
    TerminateOnNaN
)


class BaseModel(object):
    """
    Base class for Keras-based model implementations
    """

    def __init__(self, config, restore=False, setup_default_callbacks=True):
        """
        Model constructor

        Args:
            config: Dictionary of model configuration parameters
        """
        self.config = copy.deepcopy(config)

        # Session management
        BaseModel.reset_session()

        # Define core model attributes
        self.name = self.config["name"] if "name" in self.config else "Model"
        self.model_dir = self.config["model_dir"] \
            if "model_dir" in self.config else None

        self.callbacks = []
        if setup_default_callbacks:
            self._setup_default_callbacks()

        # Run any extra setup steps
        self.setup()

        # Build model
        self.build()

        if restore:
            self.load_weights()
        else:
            if self.model_dir is not None:
                self.save_config()

    def setup(self):
        pass

    def _setup_default_callbacks(self):
        if "tensorboard" in self.config and self.config["tensorboard"]:
            self.callbacks.append(TensorBoard(
                log_dir=self.model_dir + "/tensorboard",
                histogram_freq=1,
                write_graph=True,
                write_grads=True))

        if self.model_dir is not None:
            if "checkpoint" in self.config and self.config["checkpoint"]:
                checkpoint_metric = "val_loss" \
                    if "checkpoint_metric" not in self.config else \
                    self.config["checkpoint_metric"]

                self.callbacks.append(ModelCheckpoint(
                    self.model_dir + "/keras_model.weights.h5",
                    monitor=checkpoint_metric, verbose=0,
                    save_best_only=True, save_weights_only=True))
            self.callbacks.append(TimeLogger())
            self.callbacks.append(FileLogger(
                self.model_dir + "/training.log", append=True))

        if "early_stopping_metric" in self.config \
                and "early_stopping_min_delta" in self.config \
                and "early_stopping_patience" in self.config:
            self.callbacks.append(EarlyStopping(
                monitor=self.config["early_stopping_metric"],
                min_delta=self.config["early_stopping_min_delta"],
                patience=self.config["early_stopping_patience"]))

        self.callbacks.append(TerminateOnNaN())

    def build(self):
        raise Exception("The model must implement the build method")

    def save_config(self):
        """
        Save model configuration to a JSON file
        """
        with open("{}/config.json".format(self.model_dir), "w") as f:
            json.dump(self.config, f)

    @classmethod
    def load_config(cls, model_dir):
        """
        Load model configuration from a JSON file
        """
        with open("{}/config.json".format(model_dir), "r") as f:
            return json.load(f)

    def load_weights(self):
        if hasattr(self, "keras_model"):
            self.keras_model.load_weights(
                self.model_dir + "/keras_model.weights.h5")

    @classmethod
    def restore(cls, model_dir):
        """
        Restore a previously saved version of the model
        """
        config = cls.load_config(model_dir)
        return cls(config, restore=True)

    @staticmethod
    def reset_session():
        tf.reset_default_graph()
        K.clear_session()
