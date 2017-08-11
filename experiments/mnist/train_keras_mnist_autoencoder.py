from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np

from keras.datasets import mnist
from framework.keras.autoencoder import DeepAutoencoder as AE
from framework.keras.pretraining import StackedAutoencoderPretraining as SAEP
from framework.common.dataset import Dataset
from framework.common.experiment import BaseExperiment
import matplotlib.pyplot as plt

class Experiment(BaseExperiment):

    def __init__(self, debug=False):
        self.experiment_name = "test_keras_mnist_ae"
        super(Experiment, self).__init__()

    def run(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        train_dataset = Dataset(x_train, y_train,
                                flatten=True, to_one_hot=False)
        test_dataset = Dataset(x_test, y_test,
                               flatten=True, to_one_hot=False)

        model_name = "MNIST_AE"
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

            "loss": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": []
        }
    
        ae = AE(model_config)
        ae.train(train_dataset,
                 epochs=10, batch_size=100,
                 validation_dataset=test_dataset)

        num_images = 10
        color = "magma"
        reconstructed_images = ae.predict(test_dataset.features)

        figure, a = plt.subplots(2, num_images, figsize=(40, 3))
        for i in range(num_images):
            a[0][i].imshow(np.reshape(test_dataset.features[i], (28, 28)),
                           cmap=color)
            a[1][i].imshow(np.reshape(reconstructed_images[i], (28, 28)),
                           cmap=color)
        figure.show()
        plt.draw()
        plt.waitforbuttonpress()

