from math import log
import numpy as np
import pandas as pd
from hyperopt import hp
from sklearn import preprocessing

from framework.common.util import save_data_table
from framework.common.experiment import (
    CrossValidationExperiment, HyperoptExperiment
)
from framework.common.sampling import kfold
from framework.keras.autoencoder import VariationalAutoencoder as VAE

N_EVALS = 100
N_FOLDS = 5
MAX_EPOCHS = 200


class TrainGBMVAE(CrossValidationExperiment, HyperoptExperiment):
    def __init__(self, n_genes, debug=False):
        super(TrainGBMVAE, self).__init__(debug=debug)

        self.input_size = n_genes

        self.setup_dir()
        self.setup_logger()
        self.setup_hyperopt(n_evals=N_EVALS)

        cell_ids, tumor_ids, features = self.load_data()
        self.datasets = kfold(features, [cell_ids, tumor_ids], n_folds=N_FOLDS)
        self.logger.info(
            "Loaded {}g GBM dataset".format(
                self.input_size))

        self.setup_cross_validation(n_folds=N_FOLDS,
                                    datasets=self.datasets,
                                    model_class=VAE,
                                    epochs=MAX_EPOCHS)

    def load_data(self):
        filepath = "{}/data/GSE57872_GBM/processed/gbm.{}g.centered.txt"\
            .format(self.root_dir, self.input_size)
        df = pd.read_csv(filepath, sep="\t", header=0, index_col=0)
        features = df.iloc[:, 1:].values.astype(dtype=np.float64)
        features_scaled = preprocessing.scale(features)
        tumor_ids = df.iloc[:, 0].values
        cell_ids = df.index.values

        return cell_ids, tumor_ids, features_scaled

    def get_model_config(self, case_config):
        model_name = "{}_GBMVAE".format(self.case_counter)
        model_dir = self.get_model_dir(model_name)

        encoder_layers = []
        for size in case_config["encoder_layer_sizes"]:
            encoder_layers.append("Dense:{}:activation='elu'".format(size))
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

        self.logger.info("Encoding cells in latent space...")
        latent_reps = final_vae.encode(full_dataset.features)

        results = np.hstack((
            np.expand_dims(full_dataset.sample_data[0], axis=1),
            latent_reps,
            np.expand_dims(full_dataset.sample_data[1], axis=1)
        ))

        header = ["cell_ids"]
        for l in range(1, model_config["latent_size"] + 1):
            header.append("dim{}".format(l))
        header.append("tumor_ids")
        header = np.array(header)

        results = np.vstack((header, results))

        self.logger.info("Saving latent representations")
        save_data_table(
            results,
            model_config["model_dir"] + "/latent_representations.txt")

        self.logger.info("Saving losses")
        metrics = final_vae.evaluate(full_dataset)
        save_data_table(
            [["metric", "value"],
             ["total_loss", metrics["loss"]],
             ["reconstruction_loss", metrics["reconstruction_loss"]],
             ["kl_divergence_loss", metrics["kl_divergence_loss"]]],
            model_config["model_dir"] + "/final_losses.txt"
        )

    def run(self):
        self.logger.info("EXPERIMENT START")

        trials, _, best_loss_case_config = self.run_hyperopt(
            self.train_case_model)
        self.logger.info("Finished hyperopt optimization")

        # Save experiment results
        losses = []
        experiment_results = [[
            "model_name",
            "n_layers",
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
                len(result["model_config"]["encoder_layers"]) / 2,
                "|".join(result["model_config"]["encoder_layers"]),
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
        best_loss_model_config["name"] = "GBMVAE_Final"

        self.train_final_vae(best_loss_model_config)
        self.logger.info("EXPERIMENT END")


class Train500gGBMVAE(TrainGBMVAE):
    def __init__(self, debug=False):
        super(Train500gGBMVAE, self).__init__(500, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": hp.choice("encoder_layer_sizes", [
                [
                    hp.choice("1_layer1", list(range(200, 1100, 100)))
                ],
                [
                    hp.choice("2_layer1", list(range(500, 1100, 100))),
                    hp.choice("2_layer2", list(range(100, 600, 100)))
                ],
                [
                    hp.choice("3_layer1", list(range(500, 1100, 100))),
                    hp.choice("3_layer2", list(range(200, 600, 100))),
                    hp.choice("3_layer3", [100, 200])
                ]
            ]),
            "latent_size": hp.choice(
                "latent_size", [10, 15, 20, 25]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer_lr": hp.loguniform(
                "optimizer_lr", -4 * log(10), -2 * log(10))
        }


class Train1000gGBMVAE(TrainGBMVAE):
    def __init__(self, debug=False):
        super(Train1000gGBMVAE, self).__init__(1000, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": hp.choice("encoder_layer_sizes", [
                [
                    hp.choice("1_layer1", list(range(500, 1600, 100)))
                ],
                [
                    hp.choice("2_layer1", list(range(900, 1600, 100))),
                    hp.choice("2_layer2", list(range(100, 1000, 100)))
                ],
                [
                    hp.choice("3_layer1", list(range(900, 1600, 100))),
                    hp.choice("3_layer2", list(range(500, 1000, 100))),
                    hp.choice("3_layer3", list(range(100, 600, 100)))
                ]
            ]),
            "latent_size": hp.choice(
                "latent_size", [10, 15, 20, 25]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer_lr": hp.loguniform(
                "optimizer_lr", -4 * log(10), -3 * log(10))
        }
