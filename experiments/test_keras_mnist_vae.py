from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np

from keras.datasets import mnist
from framework.keras.autoencoder import VariationalAutoencoder as VAE
from framework.common.dataset import Dataset
from framework.common.experiment import BaseExperiment
import matplotlib.pyplot as plt

class Experiment(BaseExperiment):

    def __init__(self):
        self.experiment_name = "test_keras_mnist_vae"
        super(Experiment, self).__init__()

    def run(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        train_dataset = Dataset(x_train, y_train,
                                flatten=True, to_one_hot=False)
        test_dataset = Dataset(x_test, y_test,
                               flatten=True, to_one_hot=False)

        model_name = "MNIST_VAE"
        model_dir = self.get_model_dir(model_name)

        self.create_dir(model_dir)

        model_config = {
            "name": model_name,
            "model_dir": model_dir,

            "input_size": 784,
            "encoder_layers": [
                "Dense:128:activation='relu'",
                "Dense:64:activation='relu'"
            ],
            "latent_size": 10,

            "loss": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": []
        }
        
        vae = VAE(model_config)
        vae.train(train_dataset, epochs=1, batch_size=100,
                  validation_dataset=test_dataset)
