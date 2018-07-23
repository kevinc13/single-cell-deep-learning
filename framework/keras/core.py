from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import copy
import json

import os
import tensorflow as tf
from keras import backend as K
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard
)

from .callbacks import (
    TimeLogger, FileLogger,
    TerminateOnNaN,
    TerminateOnExplodingMetric)


class BaseModel(object):
    """
    Base class for Keras-based model implementations
    """

    def __init__(self, config, restore=False):
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

        self.saveable_models = {}

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

    def setup_callbacks(self, callback_config):
        callbacks = []

        if self.model_dir is not None:
            if "tensorboard" in callback_config and \
                    callback_config["tensorboard"]:
                callbacks.append(TensorBoard(
                    log_dir=self.model_dir + "/tensorboard",
                    histogram_freq=1,
                    write_graph=True,
                    write_grads=True))

            if "checkpoint" in callback_config and \
                    callback_config["checkpoint"]:

                if isinstance(callback_config["checkpoint"], dict):
                    checkpoint_metric = callback_config["checkpoint"]["metric"]
                    checkpoint_file = callback_config["checkpoint"]["file"]
                else:
                    checkpoint_metric = "val_loss"
                    checkpoint_file = "keras_model.weights.h5"

                callbacks.append(ModelCheckpoint(
                    "{}/{}".format(self.model_dir, checkpoint_file),
                    monitor=checkpoint_metric, verbose=0,
                    save_best_only=True, save_weights_only=True))

            callbacks.append(TimeLogger())

            if "file_logger" in callback_config and \
                    callback_config["file_logger"]:

                if isinstance(callback_config["file_logger"], dict):
                    log_file = callback_config["file_logger"]["file"]
                else:
                    log_file = "training.log"

                callbacks.append(FileLogger("{}/{}".format(
                    self.model_dir, log_file), append=True))

        if "early_stopping" in callback_config and \
                isinstance(callback_config["early_stopping"], dict):
            callbacks.append(EarlyStopping(
                monitor=callback_config["early_stopping"]["metric"],
                min_delta=callback_config["early_stopping"]["min_delta"],
                patience=callback_config["early_stopping"]["patience"]))

        callbacks.append(TerminateOnNaN())
        callbacks.append(TerminateOnExplodingMetric())

        return callbacks

    def build(self):
        raise Exception("The model must implement the build method")

    def add_saveable_model(self, name, model):
        if self.model_dir is not None:
            self.saveable_models[name] = model

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
        for name, _ in self.saveable_models.items():
            weights_file = "{}/{}.weights.h5".format(self.model_dir, name)
            if hasattr(self, name) and os.path.isfile(weights_file):
                getattr(self, name).load_weights(weights_file)

    @classmethod
    def restore(cls, model_dir):
        """
        Restore a previously saved version of the model
        """
        config = cls.load_config(model_dir)
        config["model_dir"] = model_dir
        return cls(config, restore=True)

    @staticmethod
    def reset_session():
        tf.reset_default_graph()
        K.clear_session()
