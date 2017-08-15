from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from math import log
import numpy as np
from hyperopt import STATUS_OK, STATUS_FAIL
from hyperopt import hp

from framework.common.dataset import Dataset
from framework.common.experiment import HyperoptExperiment
from framework.common.sampling import stratified_kfold
from framework.keras.autoencoder import VariationalAutoencoder as VAE


class Experiment(HyperoptExperiment):
    def __init__(self, debug=False):
        super(Experiment, self).__init__()
        self.debug = debug
        self.n_evals = 50

        self.experiment_name = "train_usokin-100g-2layer-vae"
        if self.debug:
            self.experiment_name = "DEBUG_" + self.experiment_name

        self.datasets = []
        self.case_counter = 0

        # Create experiment directory, setup logger
        self.setup_dir()
        self.setup_logger()

    def load_data(self):
        df = np.array(self.read_data_table(
            "data/Usokin/processed/usokin.100g.standardized.txt"))
        features = df[1:, 1:-1]

        cell_ids = df[1:, 0]
        cell_types = df[1:, -1]

        return cell_ids, features, cell_types

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": [
                hp.choice("layer1", [50, 100]),
                hp.choice("layer2", [25, 50])
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
                        "name": "rmsprop",
                        "lr": hp.loguniform(
                            "rmsprop_lr", -6 * log(10), -1 * log(10))
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

    def get_model_config(self, case_config):
        model_config = {
            "input_size": 100,
            "bernoulli": False,
            "metrics": [],

            "early_stopping_metric": "val_loss",
            "early_stopping_min_delta": 0.1,
            "early_stopping_patience": 5
        }

        model_name = "Case{}_UsokinVAE".format(self.case_counter)
        model_dir = self.get_model_dir(model_name)
        encoder_layers = [
            "Dense:{}:activation='{}'".format(
                case_config["encoder_layer_sizes"][0],
                case_config["activation"]),
            "BatchNormalization",
            "Dense:{}:activation='{}'".format(
                case_config["encoder_layer_sizes"][1],
                case_config["activation"]),
            "BatchNormalization"
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

    def train_case_vae(self, case_config):
        model_config = self.get_model_config(case_config)
        self.create_dir(model_config["model_dir"])

        self.logger.info("Training {}: {}|{}|batch_size={}".format(
            model_config["name"],
            case_config["activation"],
            model_config["optimizer"],
            model_config["batch_size"]))

        status = STATUS_OK

        avg_valid_loss = 0.0
        avg_valid_recon_loss = 0.0
        avg_valid_kl_loss = 0.0
        for k in range(0, 10):
            train_dataset = Dataset.concatenate(
                *(self.datasets[:k] + self.datasets[(k + 1):]))
            valid_dataset = self.datasets[k]
            # Start training!
            vae = VAE(model_config)

            if self.debug:
                epochs = 2
            else:
                epochs = 100

            vae.train(train_dataset,
                      epochs=epochs, batch_size=model_config["batch_size"],
                      validation_dataset=valid_dataset)

            fold_valid_loss, fold_valid_recon_loss, \
                fold_valid_kl_loss = vae.evaluate(valid_dataset)
            if np.isnan(fold_valid_loss) or np.isnan(
                    fold_valid_recon_loss) or np.isnan(fold_valid_kl_loss):
                avg_valid_loss = None
                avg_valid_recon_loss = None
                avg_valid_kl_loss = None

                status = STATUS_FAIL
                break
            else:
                self.logger.info(
                    "{}|Fold #{} Loss={:f}; Recon. Loss={:f}; KL Loss={:f}"
                        .format(model_config["name"], k + 1,
                                fold_valid_loss, fold_valid_recon_loss,
                                fold_valid_kl_loss))
                avg_valid_loss += fold_valid_loss
                avg_valid_recon_loss += fold_valid_recon_loss
                avg_valid_kl_loss += fold_valid_kl_loss

            if self.debug:
                break

        if status != STATUS_FAIL:
            avg_valid_loss /= 10
            avg_valid_recon_loss /= 10
            avg_valid_kl_loss /= 10
            self.logger.info("{}|Avg Valid Loss = {:f}".format(
                model_config["name"], avg_valid_loss))
            self.logger.info("{}|Avg Valid Recon Loss = {:f}".format(
                model_config["name"], avg_valid_recon_loss))
            self.logger.info("{}|Avg Valid KL Divergence Loss = {:f}".format(
                model_config["name"], avg_valid_kl_loss))

        self.case_counter += 1

        return {
            "status": status,
            "loss": avg_valid_loss,
            "recon_loss": avg_valid_recon_loss,
            "kl_loss": avg_valid_kl_loss,
            "name": model_config["name"],
            "model_config": model_config
        }

    def train_final_vae(self, model_config):
        model_dir = self.get_model_dir(model_config["name"])
        self.create_dir(model_dir)
        model_config["model_dir"] = model_dir
        model_config["bernoulli"] = False
        model_config["tensorboard"] = True
        model_config["checkpoint"] = True

        n_epochs = 2 if self.debug else 100
        full_dataset = Dataset.concatenate(*self.datasets)

        self.logger.info("Training Final VAE: {}".format(model_config["name"]))
        final_vae = VAE(model_config)
        final_vae.train(full_dataset, epochs=n_epochs,
                        batch_size=model_config["batch_size"],
                        validation_dataset=full_dataset)
        loss = final_vae.evaluate(full_dataset)[0]
        self.logger.info("{}|Loss = {:f}".format(model_config["name"], loss))

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

        cell_ids, features, cell_types = self.load_data()
        self.datasets = stratified_kfold(
            features, cell_types,
            [cell_ids, cell_types],
            n_folds=10, convert_labels_to_int=True)
        self.logger.info("Loaded 100g, standardized Usokin dataset")

        trials, _, best_loss_case_config = self.run_hyperopt(
            self.train_case_vae, self.n_evals)
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
                result["recon_loss"],
                result["kl_loss"],
                result["loss"]))
            experiment_results.append([
                result["name"],
                str(result["model_config"]["encoder_layers"]),
                result["model_config"]["latent_size"],
                result["model_config"]["optimizer"],
                result["model_config"]["batch_size"],
                result["recon_loss"],
                result["kl_loss"],
                result["loss"]
            ])
        self.save_data_table(
            experiment_results,
            self.experiment_dir + "/experiment_results.txt")
        self.logger.info("Saved experiment results")

        # Train the final VAE using the best model configs
        best_loss_model_config = self.get_model_config(best_loss_case_config)
        best_loss_model_config["name"] = "UsokinVAE_BestTotalLoss"

        best_recon_loss_model_config = sorted(losses, key=lambda x: x[1])[0][0]
        best_recon_loss_model_config["name"] = "UsokinVAE_BestReconLoss"

        best_kl_loss_model_config = sorted(losses, key=lambda x: x[2])[0][0]
        best_kl_loss_model_config["name"] = "UsokinVAE_BestKLDivergenceLoss"

        self.train_final_vae(best_loss_model_config)
        self.train_final_vae(best_recon_loss_model_config)
        self.train_final_vae(best_kl_loss_model_config)

        self.logger.info("EXPERIMENT END")
