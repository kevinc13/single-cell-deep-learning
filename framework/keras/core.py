from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import copy, json
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping
)
from .callbacks import (
    TimeLogger, FileLogger
)

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
        K.clear_session()
        self.sess = tf.Session()
        K.set_session(self.sess)

        # Define core model attributes
        self.name = self.config["name"] if "name" in self.config else "Model"
        self.model_dir = self.config["model_dir"] \
            if "model_dir" in self.config else None
        
        self.callbacks = []
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
        self.callbacks.append(TimeLogger())

        if self.model_dir is not None:
            self.callbacks.append(ModelCheckpoint(
                self.model_dir + "/keras_model.weights.h5",
                monitor="val_loss", verbose=0,
                save_best_only=True, save_weights_only=True))
            self.callbacks.append(FileLogger(
                self.model_dir + "/training.log", append=True))

        if "early_stopping_metric" in self.config \
                and "early_stopping_min_delta" in self.config \
                and "early_stopping_patience" in self.config:
            self.callbacks.append(EarlyStopping(
                monitor=self.config["early_stopping_metric"],
                min_delta=self.config["early_stopping_min_delta"],
                patience=self.config["early_stopping_patience"]))

    def build(self, graph):
        raise Exception("The model must implement the build method")

    def save_config(self):
        """
        Save model configuration to a JSON file
        """
        with open(self.model_dir + "/config.json", "w") as f:
            json.dump(self.config, f)

    @classmethod
    def load_config(cls, model_dir):
        """
        Load model configuration from a JSON file
        """
        with open(model_dir + "/config.json", "r") as f:
            return json.load(f)

    def load_weights(self):
        self.keras_model.load_weights(
            self.model_dir + "/keras_model.weights.h5")

    @classmethod
    def restore(cls, model_dir): 
        """
        Restore a previously saved version of the model
        """
        config = cls.load_config(model_dir)
        return cls(config, restore=True)

