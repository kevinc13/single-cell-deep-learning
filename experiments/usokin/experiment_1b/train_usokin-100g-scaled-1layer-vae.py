from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import sys
import numpy as np
from math import log
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval

from framework.common.dataset import Dataset
from framework.common.experiment import BaseExperiment
from framework.common.sampling import stratified_kfold
from framework.keras.autoencoder import VariationalAutoencoder as VAE

class Experiment(BaseExperiment):
    def __init__(self, debug=False):
        self.debug = debug
        self.n_evals = 50

        self.experiment_name = "train_usokin-100g-scaled-1layer-vae"
        if self.debug:
            self.experiment_name = "DEBUG_" + self.experiment_name

        self.datasets = []
        self.case_counter = 0
        super(Experiment, self).__init__()

    def load_data(self):
        df = np.array(self.read_data_table(
            "data/Usokin/processed/usokin.100g.scaled.txt"))
        features = df[1:, 1:-2]

        cell_ids = df[1:, 0]
        cell_types = df[1:, -2]
        cell_subtypes = df[1:, -1]

        return cell_ids, features, cell_types, cell_subtypes

    def generate_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", list(np.arange(50, 201, step=50)))
            ],
            "latent_size": hp.choice(
                "latent_size", [4, 11, 20, 50]),
            "activation": hp.choice(
                "activation", ["elu", "relu", "sigmoid"]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
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
                    },
                    {
                        "name": "rmsprop",
                        "lr": hp.loguniform(
                            "rmsprop_lr", -6 * log(10), -2 * log(10))
                    }
                ])
        }

    def get_model_config(self, case_config):
        model_config = {
            "input_size": 100,
            "loss": "binary_crossentropy",
            "metrics": [],

            "early_stopping_metric": "val_loss",
            "early_stopping_min_delta": 0.1,
            "early_stopping_patience": 5
        }

        model_name = "{}_UsokinVAE".format(self.case_counter)
        model_dir = self.get_model_dir(model_name)
        encoder_layers = [
            "Dense:{}:activation='{}'".format(
                case_config["encoder_layer_sizes"][0],
                case_config["activation"]),
            "BatchNormalization"
        ]

        if case_config["optimizer"]["name"] == "sgd":
            optimizer = "{}:lr={}:momentum={}:nesterov=True".format(
                case_config["optimizer"]["name"],
                case_config["optimizer"]["lr"],
                case_config["optimizer"]["momentum"])
        elif case_config["optimizer"]["name"] in ["adam", "rmsprop"]:
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

    def train_vae(self, case_config):
        model_config = self.get_model_config(case_config)
        self.create_dir(model_config["model_dir"])

        self.logger.info("Training {}: {}|{}|{}".format(
            model_config["name"],
            case_config["activation"],
            model_config["optimizer"],
            model_config["batch_size"]))

        avg_valid_loss = 0.0
        for k in range(0, 10):
            train_dataset = Dataset.concatenate(
                *(self.datasets[:k] + self.datasets[(k+1):]))
            valid_dataset = self.datasets[k]
            # Start training!
            vae = VAE(model_config)

            if self.debug:
                epochs = 2
            else:
                epochs = 100

            vae.train(train_dataset,
                      epochs=epochs, batch_size=case_config["batch_size"],
                      validation_dataset=valid_dataset)

            fold_valid_loss = vae.evaluate(valid_dataset)
            self.logger.info("{}|Fold #{} Loss = {:f}".format(
                model_config["name"], k + 1, fold_valid_loss))

            avg_valid_loss += fold_valid_loss

            if self.debug:
                break

        avg_valid_loss /= 10
        self.logger.info("{}|Avg Validation Loss = {:f}".format(
            model_config["name"], avg_valid_loss))

        self.case_counter += 1

        return {
            "status": STATUS_OK,
            "loss": avg_valid_loss,
            "name": model_config["name"],
            "model_config": model_config
        }

    def train_final_vae(self, model_config):
        model_config["name"] = model_config["name"] + "_FINAL"
        model_dir = self.get_model_dir(model_config["name"])
        self.create_dir(model_dir)
        model_config["model_dir"] = model_dir
        model_config["tensorboard"] = True
        model_config["checkpoint"] = True

        n_epochs = 2 if self.debug else 100
        full_dataset = Dataset.concatenate(*self.datasets)

        self.logger.info("Training Final VAE: " + model_config["name"])
        final_vae = VAE(model_config)
        final_vae.train(full_dataset,
                        epochs=n_epochs, batch_size=50,
                        validation_dataset=full_dataset)        
        loss = final_vae.evaluate(full_dataset)
        self.logger.info("{}|Loss = {:f}".format(model_config["name"], loss))

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
        self.save_data_table(
            results,
            model_config["model_dir"] + "/latent_representations.txt")        

    def run(self, debug=False):
        # Load Usokin dataset
        cell_ids, features, cell_types, cell_subtypes = self.load_data()
        self.datasets = stratified_kfold(
            features, cell_subtypes,
            [cell_ids, cell_types, cell_subtypes],
            n_folds=10, convert_labels_to_int=True)

        # Setup Hyperopt
        trials = Trials()
        search_space = self.generate_search_space()
        n_evals = 1 if self.debug else self.n_evals
        best = fmin(self.train_vae,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=n_evals,
                    trials=trials)
        self.logger.info("Finished hyperopt optimization")

        # Train the final VAE using the best model config from Hyperopt
        best_model_config = self.get_model_config(
            space_eval(search_space, best))
        self.train_final_vae(best_model_config)

        # Write experiment results
        experiment_results = [[
            "model_name",
            "encoder_layers",
            "latent_size",
            "optimizer",
            "batch_size",
            "10foldcv_loss"
        ]]
        for result in trials.results:
            experiment_results.append([
                result["name"],
                str(result["model_config"]["encoder_layers"]),
                result["model_config"]["latent_size"],
                result["model_config"]["optimizer"],
                result["model_config"]["batch_size"],
                result["loss"]
            ])
        self.save_data_table(
            experiment_results,
            self.experiment_dir + "/experiment_results.txt")