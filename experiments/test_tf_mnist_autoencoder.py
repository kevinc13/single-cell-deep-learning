from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from framework.tf.autoencoder import DeepAutoencoder as DAE
from framework.tf.dataset import DataSet
import matplotlib.pyplot as plt

from . import *


def run():
    EXP_NAME = "test_tf_mnist_autoencoder"
    create_experiment_dir(EXP_NAME)
    
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train_dataset = DataSet(mnist.train.images, mnist.train.images,
                            flatten=True, to_one_hot=False)
    validation_dataset = DataSet(mnist.validation.images,
                                 mnist.validation.images,
                                 flatten=True, to_one_hot=False)
    test_dataset = DataSet(mnist.test.images, mnist.test.images,
                           flatten=True, to_one_hot=False)

    model_name = "MNIST_AE"
    model_dir = create_model_dir(EXP_NAME, model_name)
    model_config = {
        "name": model_name,
        "model_dir": model_dir,

        "input_size": 784,
        "encoder_hidden_layers": [256, 128],
        "activations": "sigmoid",
        "loss": tf.losses.sigmoid_cross_entropy,
        "optimizer": tf.train.AdamOptimizer(learning_rate=0.01),
        "regularizer": None
    }
    
    model = DAE(model_config)
    model.train(train_dataset, validation_dataset,
                num_epochs=2, batch_size=100,
                epoch_log_verbosity=1, batch_log_verbosity=50)

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
