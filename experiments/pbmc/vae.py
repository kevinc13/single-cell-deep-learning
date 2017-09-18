from math import log
import numpy as np
import pandas as pd
from hyperopt import hp
from sklearn import preprocessing

from framework.common.util import save_data_table
from framework.common.experiment import (
    CrossValidationExperiment, HyperoptExperiment
)
from framework.common.sampling import stratified_kfold
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
        df = pd.read_csv(filepath, sep="\t", header=0, index_col=0)
        features = df.iloc[:, 0:-1].values.astype(dtype=np.float64)
        features_scaled = preprocessing.scale(features)

        cell_ids = df.index.values
        cell_types = df.iloc[:, -1].values

        return cell_ids, features_scaled, cell_types

    def get_model_config(self, case_config):
        model_name = "{}_PBMCVAE".format(self.case_counter)
        model_dir = self.get_model_dir(model_name)

        encoder_layers = []
        for size in case_config["encoder_layer_sizes"]:
            encoder_layers.append("Dense:{}:activation='{}'".format(
                size, case_config["activation"]))
            encoder_layers.append("BatchNormalization")
        optimizer = "adam:lr={}".format(case_config["optimizer_lr"])

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
        model_config["autoencoder_callbacks"]["tensorboard"] = True
        model_config["autoencoder_callbacks"]["checkpoint"] = {
            "metric": "loss",
            "file": "autoencoder_model.weights.h5"
        }
        model_config["autoencoder_callbacks"]["early_stopping"]["metric"] \
            = "loss"

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


class Train100g1LayerPBMCVAE(TrainPBMCVAE):
    def __init__(self, debug=False):
        super(Train100g1LayerPBMCVAE, self).__init__(100, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", [50, 100, 150, 200])
            ],
            "latent_size": hp.choice(
                "latent_size", [10, 15, 20, 25]),
            "activation": hp.choice(
                "activation", ["elu", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer_lr": hp.loguniform(
                "optimizer_lr", -4 * log(10), -2 * log(10))
        }


class Train100g2LayerPBMCVAE(TrainPBMCVAE):
    def __init__(self, debug=False):
        super(Train100g2LayerPBMCVAE, self).__init__(100, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", [100, 150, 200]),
                hp.choice("layer2", [25, 50, 100, 150, 200])
            ],
            "latent_size": hp.choice(
                "latent_size", [10, 15, 20, 25]),
            "activation": hp.choice(
                "activation", ["elu", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer_lr": hp.loguniform(
                "optimizer_lr", -4 * log(10), -2 * log(10))
        }


class Train200g1LayerPBMCVAE(TrainPBMCVAE):
    def __init__(self, debug=False):
        super(Train200g1LayerPBMCVAE, self).__init__(200, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", [50, 100, 150, 200, 250, 300])
            ],
            "latent_size": hp.choice(
                "latent_size", [10, 15, 20, 25]),
            "activation": hp.choice(
                "activation", ["elu", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer_lr": hp.loguniform(
                "optimizer_lr", -4 * log(10), -2 * log(10))
        }


class Train200g2LayerPBMCVAE(TrainPBMCVAE):
    def __init__(self, debug=False):
        super(Train200g2LayerPBMCVAE, self).__init__(200, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", [100, 150, 200, 250, 300]),
                hp.choice("layer2", [50, 100, 150, 200, 250, 300])
            ],
            "latent_size": hp.choice(
                "latent_size", [10, 15, 20, 25]),
            "activation": hp.choice(
                "activation", ["elu", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer_lr": hp.loguniform(
                "optimizer_lr", -4 * log(10), -2 * log(10))
        }


class Train500g1LayerPBMCVAE(TrainPBMCVAE):
    def __init__(self, debug=False):
        super(Train500g1LayerPBMCVAE, self).__init__(200, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", list(range(100, 550, 50)))
            ],
            "latent_size": hp.choice(
                "latent_size", [10, 15, 20, 25]),
            "activation": hp.choice(
                "activation", ["elu", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer_lr": hp.loguniform(
                "optimizer_lr", -4 * log(10), -2 * log(10))
        }


class Train500g2LayerPBMCVAE(TrainPBMCVAE):
    def __init__(self, debug=False):
        super(Train500g2LayerPBMCVAE, self).__init__(200, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", list(range(350, 550, 50))),
                hp.choice("layer2", list(range(100, 350, 50)))
            ],
            "latent_size": hp.choice(
                "latent_size", [10, 15, 20, 25]),
            "activation": hp.choice(
                "activation", ["elu", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer_lr": hp.loguniform(
                "optimizer_lr", -4 * log(10), -2 * log(10))
        }


class Train500g3LayerPBMCVAE(TrainPBMCVAE):
    def __init__(self, debug=False):
        super(Train500g3LayerPBMCVAE, self).__init__(200, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", [400, 450, 500]),
                hp.choice("layer2", [250, 300, 350]),
                hp.choice("layer3", [50, 100, 150, 200])
            ],
            "latent_size": hp.choice(
                "latent_size", [10, 15, 20, 25]),
            "activation": hp.choice(
                "activation", ["elu", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer_lr": hp.loguniform(
                "optimizer_lr", -4 * log(10), -2 * log(10))
        }
