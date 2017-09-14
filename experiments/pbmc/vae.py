from hyperopt import hp

from framework.common.experiment import (
    CrossValidationExperiment, HyperoptExperiment
)
from framework.common.sampling import stratified_kfold

import numpy as np
from math import log

from framework.common.util import read_data_table, save_data_table
from framework.keras.autoencoder import VariationalAutoencoder as VAE

N_EVALS = 100
N_FOLDS = 5
MAX_EPOCHS = 200


class TrainPBMCVAE(CrossValidationExperiment, HyperoptExperiment):
    def __init__(self, n_genes, debug=False):
        super(TrainPBMCVAE, self).__init__(debug=debug)

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
                                    model_class=VAE,
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
        model_name = "{}_PBMCVAE".format(self.case_counter)
        model_dir = self.get_model_dir(model_name)

        encoder_layers = []
        for size in case_config["encoder_layer_sizes"]:
            encoder_layers.append("Dense:{}:activation='{}'".format(
                size, case_config["activation"]))
            encoder_layers.append("BatchNormalization")

        if case_config["optimizer"]["name"] == "sgd":
            optimizer = "{}:lr={}:momentum={}:nesterov=True".format(
                case_config["optimizer"]["name"],
                case_config["optimizer"]["lr"],
                case_config["optimizer"]["momentum"])
        else:
            optimizer = "{}:lr={}".format(
                case_config["optimizer"]["name"],
                case_config["optimizer"]["lr"])

        model_config = {
            "name": model_name,
            "model_dir": model_dir,

            "input_shape": (self.input_size,),
            "continuous": True,
            "encoder_layers": encoder_layers,
            "latent_size": case_config["latent_size"],

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

    def train_final_vae(self, model_config):
        model_config["continuous"] = True
        model_config["tensorboard"] = True
        model_config["checkpoint"] = True
        model_config["early_stopping_metric"] = "loss"
        model_config["checkpoint_metric"] = "loss"

        results = self.train_final_model(model_config)
        final_vae = results["model"]
        full_dataset = results["dataset"]

        self.logger.info("Encoding latent represenations...")
        latent_reps = final_vae.encode(full_dataset.features)

        results = np.hstack((
            np.expand_dims(full_dataset.sample_data[0], axis=1),
            latent_reps,
            np.expand_dims(full_dataset.sample_data[1], axis=1),
            np.expand_dims(full_dataset.sample_data[2], axis=1)
        ))

        header = ["cell_ids"]
        for l in range(1, model_config["latent_size"] + 1):
            header.append("dim{}".format(l))
        header.append("cell_type")
        header.append("cell_subtype")
        header = np.array(header)

        results = np.vstack((header, results))

        self.logger.info("Saving results")
        save_data_table(
            results,
            model_config["model_dir"] + "/latent_representations.txt")

    def run(self):
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
            if None not in result["avg_valid_metrics"].values():
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
        save_data_table(
            experiment_results,
            self.experiment_dir + "/experiment_results.txt")
        self.logger.info("Saved experiment results")

        # Train the final VAE using the best model config
        best_loss_model_config = self.get_model_config(best_loss_case_config)
        best_loss_model_config["name"] = "PBMCVAE_Final"

        self.train_final_vae(best_loss_model_config)
        self.logger.info("EXPERIMENT END")


class Train100gPBMCVAE(TrainPBMCVAE):
    def __init__(self, debug=False):
        super(Train100gPBMCVAE, self).__init__(100, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", [])
            ],
            "latent_size": hp.choice(
                "latent_size", []),
            "activation": hp.choice(
                "activation", ["elu", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25]),
            "optimizer": hp.choice(
                "optimizer", [
                    {
                        "name": "adam",
                        "lr": hp.loguniform(
                            "adam_lr", -6 * log(10), -2 * log(10))
                    },
                    {
                        "name": "sgd",
                        "lr": hp.loguniform(
                            "sgd_lr", -8 * log(10), -4 * log(10)),
                        "momentum": hp.choice(
                            "sgd_momentum", [0.5, 0.9, 0.95, 0.99])
                    }
                ])
        }

