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

    def __init__(self, debug=False):
        self.experiment_name = "train_keras_mnist_vae"
        self.debug = debug
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
            "bernoulli": True,
            "encoder_layers": [
                "Dense:256:activation='relu'"
            ],
            "latent_size": 2,

            "n_warmup_epochs": 5,
            "optimizer": "adam"
        }

        if self.debug:
            epochs = 5
        else:
            epochs = 50
        
        vae = VAE(model_config)
        vae.train(train_dataset, epochs=epochs, batch_size=100,
                  validation_dataset=test_dataset)

        latent_reps = vae.encode(test_dataset.features)

        results = np.hstack((
            latent_reps, 
            np.expand_dims(test_dataset.labels, axis=1)
        ))

        header = []
        for l in range(1, model_config["latent_size"] + 1):
            header.append("dim{}".format(l))
        header.append("digit")
        header = np.array(header)

        results = np.vstack((header, results))

        self.logger.info("Saving results")
        self.save_data_table(
            results,
            model_config["model_dir"] + "/latent_representations.txt")

        plt.figure(figsize=(6, 6))
        plt.scatter(latent_reps[:, 0], latent_reps[:, 1], c=y_test)
        plt.colorbar()
        plt.show()
