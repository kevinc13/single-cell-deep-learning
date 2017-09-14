from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from math import log

import numpy as np
from hyperopt import hp

from framework.common.experiment import HyperoptExperiment, \
    CrossValidationExperiment
from framework.common.sampling import stratified_kfold
from framework.common.util import read_data_table, save_data_table
from framework.keras.autoencoder import DeepAutoencoder as AE

N_EVALS = 100
N_FOLDS = 5
MAX_EPOCHS = 200


class TrainPBMCAE(CrossValidationExperiment, HyperoptExperiment):
    def __init__(self, n_genes, debug=False):
        super(TrainPBMCAE, self).__init__(debug=debug)

        self.input_size = n_genes

        self.setup_dir()
        self.setup_logger()
        self.setup_hyperopt(n_evals=N_EVALS)

        cell_ids, features, cell_types = self.load_data()
        self.datasets = stratified_kfold(
            features, cell_types, [cell_ids, cell_types],
            n_folds=N_FOLDS, convert_labels_to_int=True)
        self.logger.info(
            "Loaded {}g, standardized GSE94820 PBMC dataset".format(
                self.input_size))

        self.setup_cross_validation(n_folds=N_FOLDS,
                                    datasets=self.datasets,
                                    model_class=AE,
                                    epochs=MAX_EPOCHS)

    def load_data(self):
        filepath = "{}/data/GSE94820_PBMC/processed/pbmc.{}g.txt".format(
            self.root_dir, self.input_size)
        df = np.array(read_data_table(filepath))
        features = df[1:, 1:-1]

        cell_ids = df[1:, 0]
        cell_types = df[1:, -1]

        return cell_ids, features, cell_types

    def get_model_config(self, case_config):
        model_name = "{}_PBMCAE".format(self.case_counter)
        model_dir = self.get_model_dir(model_name)
        encoder_layers = []
        for size in case_config["encoder_layer_sizes"]:
            encoder_layers.append("Dense:{}:activation='{}'".format(
                size, case_config["activation"]))
            encoder_layers.append("BatchNormalization")

        latent_layer = "Dense:{}:activation='{}'".format(
            case_config["latent_layer_size"], case_config["activation"])
        optimizer = "adam:lr={}".format(case_config["optimizer_lr"])

        model_config = {
            "name": model_name,
            "model_dir": model_dir,

            "input_shape": (self.input_size,),
            "continuous": True,
            "encoder_layers": encoder_layers,
            "latent_layer": latent_layer,
            "optimizer": optimizer,
            "batch_size": case_config["batch_size"],

            "autoencoder_callbacks": {
                "early_stopping": {
                    "metric": "val_loss",
                    "min_delta": 0.1,
                    "patience": 5
                },
                "file_logger": True
            }
        }

        return model_config

    def train_final_ae(self, model_config):
        model_config["autoencoder_callbacks"]["tensorboard"] = True
        model_config["autoencoder_callbacks"]["checkpoint"] = {
            "metric": "loss",
            "file": "autoencoder_model.weights.h5"
        }
        model_config["autoencoder_callbacks"]["early_stopping"]["metric"] = \
            "loss"

        results = self.train_final_model(model_config)
        final_ae = results["model"]
        full_dataset = results["dataset"]

        self.logger.info("Encoding latent represenations...")
        latent_reps = final_ae.encode(full_dataset.features)

        results = np.hstack((
            np.expand_dims(full_dataset.sample_data[0], axis=1),
            latent_reps,
            np.expand_dims(full_dataset.sample_data[1], axis=1)
        ))

        latent_size = int(model_config["latent_layer"].split(":")[1])

        header = ["cell_ids"]
        for l in range(1, latent_size + 1):
            header.append("dim{}".format(l))
        header.append("cell_type")
        header = np.array(header)

        results = np.vstack((header, results))

        self.logger.info("Saving results")
        save_data_table(
            results, model_config["model_dir"] + "/latent_representations.txt")

    def run(self, debug=False):
        self.logger.info("EXPERIMENT START")

        trials, _, final_case_config = self.run_hyperopt(
            self.train_case_model)
        self.logger.info("Finished hyperopt optimization")

        # Save experiment results
        experiment_results = [[
            "model_name",
            "encoder_layers",
            "latent_layer",
            "optimizer",
            "batch_size",
            "cv_loss"
        ]]
        for result in trials.results:
            experiment_results.append([
                result["model_config"]["name"],
                result["model_config"]["encoder_layers"],
                result["model_config"]["latent_layer"],
                result["model_config"]["optimizer"],
                result["model_config"]["batch_size"],
                result["avg_valid_metrics"]["loss"]
            ])
        save_data_table(
            experiment_results,
            self.experiment_dir + "/experiment_results.txt")
        self.logger.info("Saved experiment results")

        # Train the final AE using the best model config
        final_model_config = self.get_model_config(final_case_config)
        final_model_config["name"] = "PBMCAE_FINAL"
        self.train_final_ae(final_model_config)

        self.logger.info("EXPERIMENT END")


class Train100g1LayerPBMCAE(TrainPBMCAE):
    def __init__(self, debug=False):
        super(Train100g1LayerPBMCAE, self).__init__(100, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", [100, 200]),
            ],
            "latent_layer_size": hp.choice(
                "latent_layer_size", [10, 15, 20, 25, 50]),
            "activation": hp.choice(
                "activation", ["elu", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer_lr": hp.loguniform(
                "optimizer_lr", -4 * log(10), -2 * log(10))
        }


class Train100g2LayerPBMCAE(TrainPBMCAE):
    def __init__(self, debug=False):
        super(Train100g2LayerPBMCAE, self).__init__(100, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", [50, 100, 200]),
                hp.choice("layer1", [25, 50, 100])
            ],
            "latent_layer_size": hp.choice(
                "latent_layer_size", [10, 15, 20, 25]),
            "activation": hp.choice(
                "activation", ["elu", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer_lr": hp.loguniform(
                "adam_lr", -4 * log(10), -2 * log(10))
        }
