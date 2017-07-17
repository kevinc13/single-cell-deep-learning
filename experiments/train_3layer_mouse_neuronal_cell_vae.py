from __future__ import (
    absolute_import, division, print_function, unicode_literals
)
from sklearn.manifold import TSNE

import numpy as np

from framework.common.dataset import Dataset
from framework.common.experiment import BaseExperiment
from framework.common.sampling import stratified_kfold
from framework.common.util import from_one_hot
from framework.keras.autoencoder import VariationalAutoencoder as VAE

class Experiment(BaseExperiment):
    def __init__(self):
        self.experiment_name = "train_3layer_mouse_neuronal_cell_vae"
        super(Experiment, self).__init__()

    def load_data(self):
        df = np.array(self.read_data_table(
            "data/Usokin/processed/mouse_neuronal_cells.500g.txt"))
        features = df[1:, 1:-2]

        cell_ids = df[1:, 0]
        cell_types = df[1:, -2]
        cell_subtypes = df[1:, -1]

        return cell_ids, features, cell_types, cell_subtypes

    def generate_search_space(self, search_space_size):
        # Setup hyperparameter/architecture search space
        search_space = []

        base_config = {
            "input_size": 500,
            "loss": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": [],

            "early_stopping_metric": "val_loss",
            "early_stopping_min_delta": 0.01,
            "early_stopping_patience": 5
        }

        # 2 latent layer models
        for i in range(search_space_size):
            layer1 = int(np.random.choice(np.arange(300, 401, step=50)))
            layer2 = int(np.random.choice(np.arange(200, 251, step=50)))
            layer3 = int(np.random.choice(np.arange(100, 151, step=50)))
            latent_size = int(np.random.choice(np.arange(2, 21, step=2)))

            batch_norm = bool(np.random.choice([True, False]))

            if batch_norm:
                encoder_layers = [
                    "Dense:{}:activation='relu'".format(layer1),
                    "BatchNormalization",
                    "Dense:{}:activation='relu'".format(layer2),
                    "BatchNormalization"
                ]

                model_name = "{}_UsokinVAE_{}-{}_{}L_BN".format(
                    i, layer1, layer2, latent_size)
            else:
                encoder_layers = [
                    "Dense:{}:activation='relu'".format(layer1),
                    "Dense:{}:activation='relu'".format(layer2)
                ]
                model_name = "{}_UsokinVAE_{}-{}_{}L".format(
                    i, layer1, layer2, latent_size)

            model_dir = self.get_model_dir(model_name)

            self.create_dir(model_dir)

            model_config = dict(base_config)
            model_config.update({
                "name": model_name,
                "model_dir": model_dir,

                "encoder_layers": encoder_layers,
                "latent_size": latent_size
            })

            search_space.append(model_config)

        return search_space

    def run(self, debug=False):
        cell_ids, features, cell_types, cell_subtypes = self.load_data()
        datasets = stratified_kfold(
            features, cell_subtypes, n_folds=10, convert_labels_to_int=True)

        search_space = self.generate_search_space(30)

        experiment_results = [["model_name", "avg_validation_loss"]]
        for model_config in search_space:
            avg_valid_loss = 0.0
            for k in range(0, 10):
                train_dataset = Dataset.concatenate(
                    *(datasets[:k] + datasets[(k+1):]))
                valid_dataset = datasets[k]
                # Start training!
                vae = VAE(model_config)

                if debug:
                    epochs = 2
                else:
                    epochs = 50

                vae.train(train_dataset,
                          epochs=epochs, batch_size=100,
                          validation_dataset=valid_dataset)

                fold_valid_loss = vae.evaluate(valid_dataset)
                self.logger.info("{}|Fold #{} Loss = {:f}".format(
                    model_config["name"], k + 1, fold_valid_loss))

                avg_valid_loss += fold_valid_loss

                if debug:
                    break

            avg_valid_loss /= 10

            experiment_results.append([
                model_config["name"],
                avg_valid_loss
            ])

            self.logger.info("{}|Avg Loss = {:f}".format(
                model_config["name"], avg_valid_loss))

            # Save model's latent representations of full dataset
            full_dataset = Dataset.concatenate(*datasets)
            latent_reps = vae.encode(full_dataset.features)
            results = np.hstack((
                latent_reps, 
                np.expand_dims(from_one_hot(full_dataset.labels), axis=1)
            ))

            header = []
            for l in range(1, model_config["latent_size"] + 1):
                header.append("dim{}".format(l))
            header.append("cell_subtype")
            header = np.array(header)

            results = np.vstack((header, results))

            self.save_data_table(
                results, 
                model_config["model_dir"] + "/latent_representations.txt")

            if debug:
                break

        self.save_data_table(
            experiment_results,
            self.experiment_dir + "/experiment_results.txt")
