from keras.datasets import mnist
import numpy as np

from framework.common.dataset import Dataset
from framework.common.experiment import BaseExperiment
from framework.common.util import create_dir, save_data_table
import matplotlib.pyplot as plt

from framework.keras.autoencoder import KadurinAdversarialAutoencoder, \
    UnsupervisedClusteringAdversarialAutoencoder, AdversarialAutoencoder
from sklearn.metrics import adjusted_rand_score


class TrainMNISTAAE(BaseExperiment):
    def __init__(self, debug=False):
        super(TrainMNISTAAE, self).__init__(debug=debug)

        self.setup_dir()
        self.setup_logger()

    def run(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (x_train.astype("float32") - 127.5) / 255
        x_test = (x_test.astype("float32") - 127.5) / 255

        train_dataset = Dataset(x_train, y_train,
                                flatten=True, to_one_hot=False)
        test_dataset = Dataset(x_test, y_test,
                               flatten=True, to_one_hot=False)

        model_name = "MNIST_AAE"
        model_type = UnsupervisedClusteringAdversarialAutoencoder
        model_dir = self.get_model_dir(model_name)
        create_dir(model_dir)

        aae_model_config = {
            "name": model_name,
            "model_dir": model_dir,

            "input_shape": (784,),
            "encoder_layers": [
                "Dense:256:activation='relu'",
            ],
            "z_latent_distribution": "gaussian:2",
            "z_prior_distribution": "gaussian:2:mean=0.0:stddev=5.0",
            "output_distribution": "mean_gaussian:784",
            "z_discriminator_layers": [
                "Dense:50:activation='relu'",
                "Dense:25:activation='relu'"
            ],

            "autoencoder_optimizer": "adam:lr=0.001",
            "z_discriminator_optimizer": "adam:lr=0.001",

            "autoencoder_callbacks": {
                "file_logger": {"file": "autoencoder_model.training.log"}
            },
            "z_discriminator_callbacks": {
                "file_logger": {"file": "discriminator_model.training.log"}
            }
        }

        kadurin_aae_model_config = {
            "name": model_name,
            "model_dir": model_dir,

            "input_shape": (784,),
            "encoder_layers": [
                "Dense:256:activation='relu'",
            ],
            "z_latent_distribution": "gaussian:2",
            "z_prior_distribution": "gaussian:2:mean=0.0:stddev=5.0",
            "output_distribution": "mean_gaussian:784",
            "z_discriminator_layers": [
                "Dense:50:activation='relu'",
                "Dense:25:activation='relu'"
            ],
            "discriminative_power": 0.6,

            "autoencoder_optimizer": "adam:lr=0.0001",
            "z_discriminator_optimizer": "adam:lr=0.0001",

            "z_combined_callbacks": {
                "file_logger": {"file": "combined_model.training.log"}
            },
            "z_discriminator_callbacks": {
                "file_logger": {"file": "discriminator_model.training.log"}
            }
        }

        unsupervised_clustering_aae_config = {
            "name": model_name,
            "model_dir": model_dir,

            "input_shape": (784,),
            "encoder_layers": [
                "Dense:256:activation='relu'"
            ],

            "z_latent_distribution": "gaussian:2",
            "z_prior_distribution": "gaussian:2:mean=0.0:stddev=5.0",
            "n_clusters": 10,
            "output_distribution": "mean_gaussian:784",

            "z_discriminator_layers": [
                "Dense:50:activation='relu'",
                "Dense:25:activation='relu'"
            ],

            "y_discriminator_layers": [
                "Dense:50:activation='relu'",
                "Dense:25:activation='relu'"
            ],

            "autoencoder_optimizer": "adam:lr=0.0001",
            "z_discriminator_optimizer": "adam:lr=0.0001",
            "y_discriminator_optimizer": "adam:lr=0.0001",

            "autoencoder_callbacks": {
                "file_logger": {"file": "autoencoder_model.training.log"}
            }
            # "z_discriminator_callbacks": {},
            # "y_discriminator_callbacks": {},
            # "z_adversarial_callbacks": {},
            # "y_adversarial_callbacks": {}
        }

        if self.debug:
            epochs = 5
        else:
            epochs = 50

        aae = model_type(unsupervised_clustering_aae_config)
        aae.train(train_dataset, epochs=epochs, batch_size=100,
                  validation_dataset=test_dataset, verbose=2)

        latent_space = aae.encode(test_dataset.features)
        style, clusters = latent_space[0], latent_space[1]
        clusters = aae.cluster(test_dataset.features)

        # results = np.hstack((
        #     latent_space,
        #     np.expand_dims(test_dataset.labels, axis=1)
        # ))

        # header = []
        # for l in range(1, 3):
        #     header.append("dim{}".format(l))
        # header.append("digit")
        # header = np.array(header)
        #
        # results = np.vstack((header, results))
        #
        # self.logger.info("Saving results")
        # save_data_table(
        #     results,
        #     model_config["model_dir"] + "/latent_representations.txt")

        print("ARI: ", adjusted_rand_score(test_dataset.labels, clusters))
        print("Clusters:", np.unique(clusters, return_counts=True))

        plt.figure(figsize=(6, 6))
        plt.scatter(style[:, 0], style[:, 1],
                    c=y_test, cmap="rainbow")
        plt.colorbar()
        plt.show()