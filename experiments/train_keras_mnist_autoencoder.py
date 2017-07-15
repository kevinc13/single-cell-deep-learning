from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np

from keras.datasets import mnist
from framework.keras.autoencoder import DeepAutoencoder
from framework.common.dataset import Dataset
import matplotlib.pyplot as plt

from . import *


def run():
    EXP_NAME = "test_keras_mnist_autoencoder"
    create_experiment_dir(EXP_NAME)

    (x_train, _), (x_test, _) = mnist.load_data()
    train_dataset = Dataset(x_train, x_train,
                            flatten=True, to_one_hot=False)
    test_dataset = Dataset(x_test, x_test,
                           flatten=True, to_one_hot=False)

    model_name = "MNIST_AE"
    model_dir = create_model_dir(EXP_NAME, model_name)
    model_config = {
        "name": model_name,
        "model_dir": model_dir,

        "input_size": 784,
        "encoder_hidden_layers": [256, 128],
        "activations": "sigmoid",
        "loss": "binary_crossentropy",
        "optimizer": "adam"
    }
    
    model = DeepAutoencoder(model_config)
    model.train(train_dataset, batch_size=100, epochs=2,
                validation_dataset=validation_dataset)

    num_images = 10
    color = "magma"
    reconstructed_images = model.sess.run(model.outputs[-1], feed_dict={
        model.x: test_dataset.features[:num_images],
        model.y: test_dataset.labels[:num_images]
    })

    figure, a = plt.subplots(2, num_images, figsize=(40, 3))
    for i in range(num_images):
        a[0][i].imshow(np.reshape(test_dataset.features[i], (28, 28)),
                       cmap=color)
        a[1][i].imshow(np.reshape(reconstructed_images[i], (28, 28)),
                       cmap=color)
    figure.show()
    plt.draw()
    plt.waitforbuttonpress()
