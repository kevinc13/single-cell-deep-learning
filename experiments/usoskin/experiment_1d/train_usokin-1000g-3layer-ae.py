from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from math import log

import numpy as np
from hyperopt import hp

from framework.common.experiment import HyperoptExperiment, \
    CrossValidationExperiment
from framework.common.sampling import stratified_kfold
from framework.keras.autoencoder import DeepAutoencoder as AE


class Experiment(CrossValidationExperiment, HyperoptExperiment):
    def __init__(self, debug=False):
        super(Experiment, self).__init__(debug)

        self.experiment_name = "train_usokin-1000g-3layer-ae"
        if self.debug:
            self.experiment_name = "DEBUG_" + self.experiment_name

        self.setup_dir()
        self.setup_logger()
        self.setup_hyperopt(n_evals=50)

        self.input_size = 1000
        cell_ids, features, cell_types, cell_subtypes = self.load_data()
        self.datasets = stratified_kfold(
            features, cell_subtypes,
            [cell_ids, cell_types, cell_subtypes],
            n_folds=5, convert_labels_to_int=True)
        self.logger.info("Loaded 1000g, standardized Usokin dataset")

        self.setup_cross_validation(n_folds=5,
                                    datasets=self.datasets,
                                    model_class=AE,
                                    epochs=200)

    def load_data(self):
        df = np.array(self.read_data_table(
            "data/Usokin/processed/usokin.1000g.standardized.txt"))
        features = df[1:, 1:-2]

        cell_ids = df[1:, 0]
        cell_types = df[1:, -2]
        cell_subtypes = df[1:, -1]

        return cell_ids, features, cell_types, cell_subtypes

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", [800, 900, 1000]),
                hp.choice("layer2", [500, 600, 700]),
                hp.choice("layer3", [200, 300, 400]),
                hp.choice("latent_layer", [10, 15, 20, 25, 50, 100, 200])
            ],
            "activation": hp.choice(
                "activation", ["elu", "relu", "sigmoid", "tanh"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "optimizer": hp.choice(
                "optimizer", [
                    {
                        "name": "adam",
                        "lr": hp.loguniform(
                            "adam_lr", -5 * log(10), -2 * log(10))
                    },
                    {
                        "name": "sgd",
                        "lr": hp.loguniform(
                            "sgd_lr", -7 * log(10), -3 * log(10)),
                        "momentum": hp.choice(
                            "sgd_momentum", [0.5, 0.9, 0.95, 0.99])
                    }
                ])
        }

    def get_model_config(self, case_config):
        model_config = {
            "input_size": self.input_size,
            "metrics": [],
            "bernoulli": False,
            "loss": "mse",

            "early_stopping_metric": "val_loss",
            "early_stopping_min_delta": 0.1,
            "early_stopping_patience": 5
        }

        model_name = "{}_UsokinAE".format(self.case_counter)
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
            "Dense:{}:activation='{}'".format(
                case_config["encoder_layer_sizes"][2],
                case_config["activation"]),
            "BatchNormalization",
            "Dense:{}:activation='{}'".format(
                case_config["encoder_layer_sizes"][3],
                case_config["activation"])
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
            "optimizer": optimizer,
            "batch_size": case_config["batch_size"]
        })

        return model_config

    def train_final_ae(self, model_config):
        model_config["tensorboard"] = True
        model_config["checkpoint"] = True
        model_config["early_stopping_metric"] = "loss"
        model_config["checkpoint_metric"] = "loss"

        results = self.train_final_model(model_config)
        final_ae = results["model"]
        full_dataset = results["dataset"]

        self.logger.info("Encoding latent represenations...")
        latent_reps = final_ae.encode(full_dataset.features)

        results = np.hstack((
            np.expand_dims(full_dataset.sample_data[0], axis=1),
            latent_reps,
            np.expand_dims(full_dataset.sample_data[1], axis=1),
            np.expand_dims(full_dataset.sample_data[2], axis=1)
        ))

        latent_size = int(model_config["encoder_layers"][-1].split(":")[1])

        header = ["cell_ids"]
        for l in range(1, latent_size + 1):
            header.append("dim{}".format(l))
        header.append("cell_type")
        header.append("cell_subtype")
        header = np.array(header)

        results = np.vstack((header, results))

        self.logger.info("Saving results")
        save_data_table(
            results,
            model_config["model_dir"] + "/latent_representations.txt")

    def run(self, debug=False):
        self.logger.info("EXPERIMENT START")

        trials, _, best_loss_case_config = self.run_hyperopt(
            self.train_case_model)
        self.logger.info("Finished hyperopt optimization")

        # Save experiment results
        experiment_results = [[
            "model_name",
            "encoder_layers",
            "optimizer",
            "batch_size",
            "cv_loss"
        ]]
        for result in trials.results:
            experiment_results.append([
                result["model_config"]["name"],
                result["model_config"]["encoder_layers"],
                result["model_config"]["optimizer"],
                result["model_config"]["batch_size"],
                result["avg_valid_metrics"]["loss"]
            ])
        save_data_table(
            experiment_results,
            self.experiment_dir + "/experiment_results.txt")
        self.logger.info("Saved experiment results")

        # Train the final AE using the best model configs
        best_loss_model_config = self.get_model_config(best_loss_case_config)
        best_loss_model_config["name"] = "UsokinAE_BestLoss"
        self.train_final_ae(best_loss_model_config)

        self.logger.info("EXPERIMENT END")
