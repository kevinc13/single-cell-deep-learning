from hyperopt import hp

from framework.common.experiment import (
    CrossValidationExperiment, HyperoptExperiment
)
from framework.common.sampling import stratified_kfold

import numpy as np
from math import log

from framework.common.util import read_data_table, save_data_table
from framework.keras.gan import WassersteinGAN as WGAN


class TrainPBMCWGAN(CrossValidationExperiment, HyperoptExperiment):
    def __init__(self, n_genes, debug=False):
        super(TrainPBMCWGAN, self).__init__(debug=debug)

        self.input_size = n_genes

        self.setup_dir()
        self.setup_logger()
        self.setup_hyperopt(n_evals=100)

        cell_ids, features, cell_types = self.load_data()
        self.datasets = stratified_kfold(
            features, cell_types, [cell_ids, cell_types],
            n_folds=5, convert_labels_to_int=True)
        self.logger.info(
            "Loaded {}g, standardized GSE94820 PBMC dataset".format(
                self.input_size))

        self.setup_cross_validation(n_folds=5,
                                    datasets=self.datasets,
                                    model_class=WGAN,
                                    epochs=200)

    def load_data(self):
        filepath = "{}/data/GSE94820_PBMC/processed/pbmc.{}g.txt".format(
            self.root_dir, self.input_size)
        df = np.array(read_data_table(filepath))
        features = df[1:, 1:-1]

        cell_ids = df[1:, 0]
        cell_types = df[1:, -1]

        return cell_ids, features, cell_types

    def get_model_config(self, case_config):
        model_config = {

        }

        return model_config

    def train_final_wgan(self, model_config):
        pass

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

        # Train the final WGAN using the best model config
        best_loss_model_config = self.get_model_config(best_loss_case_config)
        best_loss_model_config["name"] = "PBMCWGAN_Final"

        self.train_final_wgan(best_loss_model_config)
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

