from keras.datasets import mnist
import numpy as np

from framework.common.dataset import Dataset
from framework.common.experiment import BaseExperiment
from framework.common.util import create_dir, save_data_table
from framework.keras.autoencoder import AdversarialAutoencoder as AAE
import matplotlib.pyplot as plt


class TrainMNISTAAE(BaseExperiment):
    def __init__(self, debug=False):
        super(TrainMNISTAAE, self).__init__(debug=debug)

        self.setup_dir()
        self.setup_logger()

    def run(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        train_dataset = Dataset(x_train, y_train,
                                flatten=True, to_one_hot=False)
        test_dataset = Dataset(x_test, y_test,
                               flatten=True, to_one_hot=False)

        model_name = "MNIST_AAE"
        model_dir = self.get_model_dir(model_name)

        create_dir(model_dir)

        model_config = {
            "name": model_name,
            "model_dir": model_dir,

            "input_size": 784,
            "encoder_layers": [
                "Dense:1000:activation='relu'",
                "Dense:1000:activation='relu'",
                "Dense:2:activation='relu'",
            ],
            "discriminator_layers": [
                "Dense:1000:activation='relu'",
                "Dense:1000:activation='relu'"
            ],

            "autoencoder_optimizer": "adam:lr=0.01",
            "discriminator_optimizer": "sgd:lr=0.1:momentum=0.9:nesterov=True",

            "autoencoder_callbacks": {
                "tensorboard": True,
                "checkpoint": {
                    "metric": "val_loss",
                    "file": "autoencoder_model.weights.h5"
                },
                "file_logger": {
                    "file": "autoencoder_model.training.log"
                }
            },
            "discriminator_callbacks": {
                "tensorboard": True,
                "checkpoint": {
                    "metric": "loss",
                    "file": "discriminator_model.weights.h5",
                },
                "file_logger": {"file": "discriminator_model.training.log"}
            }
        }

        if self.debug:
            epochs = 1
        else:
            epochs = 50

        aae = AAE(model_config)
        aae.train(train_dataset, epochs=epochs, batch_size=100,
                  validation_dataset=test_dataset)

        latent_reps = aae.encode(test_dataset.features)

        results = np.hstack((
            latent_reps,
            np.expand_dims(test_dataset.labels, axis=1)
        ))

        header = []
        for l in range(1, 3):
            header.append("dim{}".format(l))
        header.append("digit")
        header = np.array(header)

        results = np.vstack((header, results))

        self.logger.info("Saving results")
        save_data_table(
            results,
            model_config["model_dir"] + "/latent_representations.txt")

        plt.figure(figsize=(6, 6))
        plt.scatter(latent_reps[:, 0], latent_reps[:, 1],
                    c=y_test, cmap="rainbow")
        plt.colorbar()
        plt.show()