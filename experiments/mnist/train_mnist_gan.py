from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np

from keras.datasets import mnist

from framework.common.util import create_dir
from framework.keras.gan import GAN
from framework.keras.gan import WassersteinGAN as WGAN
from framework.common.dataset import Dataset
from framework.common.experiment import BaseExperiment
import matplotlib.pyplot as plt


class TrainMNISTGAN(BaseExperiment):
    def __init__(self, debug=False):
        super(TrainMNISTGAN, self).__init__(debug)

        self.setup_dir()
        self.setup_logger()

    def run(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (x_train.astype("float32") - 127.5) / 127.5
        x_train = np.expand_dims(x_train, axis=3)

        x_test = (x_test.astype("float32") - 127.5) / 127.5
        x_test = np.expand_dims(x_test, axis=3)

        train_dataset = Dataset(x_train, y_train,
                                flatten=False, to_one_hot=False)
        # test_dataset = Dataset(x_test, y_test,
        #                        flatten=False, to_one_hot=False)

        model_name = "MNIST_GAN"
        model_dir = self.get_model_dir(model_name)

        create_dir(model_dir)

        model_config = {
            "name": model_name,
            "model_dir": model_dir,

            "input_shape": (28, 28, 1),
            "generator_layers": [
                "Dense:1024:activation='tanh'",

                "Dense:128*7*7",
                "BatchNormalization",
                "Activation:'tanh'",

                "Reshape:(7, 7, -1)",
                "UpSampling2D:size=(2,2)",
                "Conv2D:64:5:padding='same':activation='tanh'",

                "UpSampling2D:size=(2,2)",
                "Conv2D:1:5:padding='same'",
                "Activation:'tanh'"
            ],
            "discriminator_layers": [
                "Conv2D:64:5:padding='same':activation='tanh'",
                "MaxPooling2D:pool_size=(2,2)",
                "Conv2D:128:5:padding='same':activation='tanh'",
                "MaxPooling2D:pool_size=(2,2)",

                "Flatten",
                "Dense:1024:activation='tanh'",
                "Dense:1:activation='sigmoid'"
            ],
            "prior_size": 64,

            "discriminator_loss": "binary_crossentropy",

            "gan_optimizer": "adam:lr=1e-4",
            "discriminator_optimizer": "adam:lr=1e-3"
        }

        if self.debug:
            iterations = 3
        else:
            iterations = 5000

        gan = GAN(model_config)
        g_loss, d_loss_real, d_loss_gen = gan.train(
            train_dataset, iterations, batch_size=64)
        print("Generator Loss: ", g_loss)
        print("Discriminator Loss (Real): ", d_loss_real)
        print("Discriminator Loss (Generated): ", d_loss_gen)
        print("Finished training GAN")