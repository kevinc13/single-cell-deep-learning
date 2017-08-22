from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from math import log

import numpy as np
from hyperopt import hp

from framework.common.experiment import HyperoptExperiment, \
    CrossValidationExperiment
from framework.common.sampling import stratified_kfold
from framework.keras.autoencoder import VariationalAutoencoder as VAE


class Experiment(CrossValidationExperiment, HyperoptExperiment):
    def __init__(self, debug=False):
        super(Experiment, self).__init__(debug)

        self.experiment_name = "train_pbmc-100g-1layer-vae"
        if self.debug:
            self.experiment_name = "DEBUG_" + self.experiment_name

        self.setup_dir()
        self.setup_logger()
        self.setup_hyperopt(n_evals=50)

        self.input_size = 100
        cell_ids, features, cell_types = self.load_data()
        self.datasets = stratified_kfold(
            features, cell_types,
            [cell_ids, cell_types],
            n_folds=5, convert_labels_to_int=True)
        self.logger.info("Loaded 100g, standardized GSE94820 PBMC dataset")

        self.setup_cross_validation(n_folds=5,
                                    datasets=self.datasets,
                                    model_class=VAE)

    def load_data(self):
        df = np.array(self.read_data_table(
            "data/GSE94820_PBMC/processed/pbmc.100g.standardized.txt"))
        features = df[1:, 1:-1]

        cell_ids = df[1:, 0]
        cell_types = df[1:, -1]

        return cell_ids, features, cell_types

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", list(range(50, 110, step=10))),
                hp.choice("layer2", list(range(20, 60, step=10)))
            ],
            "latent_size": hp.choice(
                "latent_size", [10, 15, 20, 25, 50]),
            "activation": hp.choice(
                "activation", ["elu", "relu", "sigmoid", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer": hp.choice(
                "optimizer", [
                    {
                        "name": "adam",
                        "lr": hp.loguniform(
                            "adam_lr", -6 * log(10), -1 * log(10))
                    },
                    {
                        "name": "sgd",
                        "lr": hp.loguniform(
                            "sgd_lr", -8 * log(10), -4 * log(10)),
                        "momentum": hp.choice(
                            "sgd_momentum", [0.5, 0.9, 0.95, 0.99])
                    },
                    {
                        "name": "rmsprop",
                        "lr": hp.loguniform(
                            "rmsprop_lr", -6 * log(10), -1 * log(10))
                    }
                ])
        }

    def get_model_config(self, case_config):
        model_config = {
            "input_size": self.input_size,
            "bernoulli": False,
            "metrics": [],

            "early_stopping_metric": "val_loss",
            "early_stopping_min_delta": 0.1,
            "early_stopping_patience": 5
        }

        model_name = "{}_PBMCVAE".format(self.case_counter)
        model_dir = self.get_model_dir(model_name)
        encoder_layers = [
            "Dense:{}:activation='{}'".format(
                case_config["encoder_layer_sizes"][0],
                case_config["activation"]),
            "BatchNormalization",
            "Dense:{}:activation='{}'".format(
                case_config["encoder_layer_sizes"][1],
                case_config["activation"]),
            "BatchNormalization",
        ]

        if case_config["optimizer"]["name"] == "sgd":
            optimizer = "{}:lr={}:momentum={}:nesterov=True".format(
                case_config["optimizer"]["name"],
                case_config["optimizer"]["lr"],
                case_config["optimizer"]["momentum"])
        else:
            optimizer = "{}:lr={}".format(
                case_config["optimizer"]["name"],
                case_config["optimizer"]["lr"])

        model_config.update({
            "name": model_name,
            "model_dir": model_dir,
            "encoder_layers": encoder_layers,
            "latent_size": case_config["latent_size"],
            "optimizer": optimizer,
            "batch_size": case_config["batch_size"]
        })

        return model_config

    def train_final_vae(self, model_config):
        model_config["bernoulli"] = False
        model_config["tensorboard"] = True
        model_config["checkpoint"] = True

        results = self.train_final_model(model_config)
        final_vae = results["model"]
        full_dataset = results["dataset"]

        self.logger.info("Encoding latent represenations...")
        latent_reps = final_vae.encode(full_dataset.features)

        results = np.hstack((
            np.expand_dims(full_dataset.sample_data[0], axis=1),
            latent_reps,
            np.expand_dims(full_dataset.sample_data[1], axis=1)
        ))

        header = ["cell_ids"]
        for l in range(1, model_config["latent_size"] + 1):
            header.append("dim{}".format(l))
        header.append("cell_type")
        header = np.array(header)

        results = np.vstack((header, results))

        self.logger.info("Saving results")
        self.save_data_table(
            results,
            model_config["model_dir"] + "/latent_representations.txt")

    def run(self, debug=False):
        self.logger.info("EXPERIMENT START")

        trials, _, best_loss_case_config = self.run_hyperopt(
            self.train_case_model)
        self.logger.info("Finished hyperopt optimization")

        # Save experiment results
        losses = []
        experiment_results = [[
            "model_name",
            "encoder_layers",
            "latent_size",
            "optimizer",
            "batch_size",
            "cv_reconstruction_loss",
            "cv_kl_divergence_loss",
            "cv_total_loss"
        ]]
        for result in trials.results:
            losses.append((
                result["model_config"],
                result["avg_valid_metrics"]["reconstruction_loss"],
                result["avg_valid_metrics"]["kl_divergence_loss"],
                result["avg_valid_metrics"]["loss"]))
            experiment_results.append([
                result["model_config"]["name"],
                result["model_config"]["encoder_layers"],
                result["model_config"]["latent_size"],
                result["model_config"]["optimizer"],
                result["model_config"]["batch_size"],
                result["avg_valid_metrics"]["reconstruction_loss"],
                result["avg_valid_metrics"]["kl_divergence_loss"],
                result["avg_valid_metrics"]["loss"]
            ])
        self.save_data_table(
            experiment_results,
            self.experiment_dir + "/experiment_results.txt")
        self.logger.info("Saved experiment results")

        # Train the final VAE using the best model configs
        best_loss_model_config = self.get_model_config(best_loss_case_config)
        best_loss_model_config["name"] = "PBMCVAE_BestTotalLoss"

        best_recon_loss_model_config = sorted(losses, key=lambda x: x[1])[0][0]
        best_recon_loss_model_config["name"] = "PBMCVAE_BestReconLoss"

        best_kl_loss_model_config = sorted(losses, key=lambda x: x[2])[0][0]
        best_kl_loss_model_config["name"] = "PBMCVAE_BestKLDivergenceLoss"

        self.train_final_vae(best_loss_model_config)
        self.train_final_vae(best_recon_loss_model_config)
        self.train_final_vae(best_kl_loss_model_config)

        self.logger.info("EXPERIMENT END")
