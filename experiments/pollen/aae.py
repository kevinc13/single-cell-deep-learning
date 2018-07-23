from math import log
import numpy as np
import pandas as pd
from hyperopt import hp
from sklearn import preprocessing

from framework.common.experiment import CrossValidationExperiment, \
    HyperoptExperiment
from framework.common.sampling import stratified_kfold
from framework.common.util import save_data_table
from framework.keras.autoencoder import AdversarialAutoencoder as AAE

N_EVALS = 100
N_FOLDS = 5
MAX_EPOCHS = 200


class TrainPollenAAE(CrossValidationExperiment, HyperoptExperiment):
    def __init__(self, n_genes, debug=False):
        super(TrainPollenAAE, self).__init__(debug=debug)

        self.input_size = n_genes

        self.setup_dir()
        self.setup_logger()
        self.setup_hyperopt(n_evals=N_EVALS)

        cell_ids, features, cell_types, cell_subtypes = self.load_data()
        self.datasets = stratified_kfold(
            features, cell_subtypes, [cell_ids, cell_types, cell_subtypes],
            n_folds=N_FOLDS, convert_labels_to_int=True)
        self.logger.info(
            "Loaded {}g Pollen dataset".format(
                self.input_size))

        self.setup_cross_validation(n_folds=N_FOLDS,
                                    datasets=self.datasets,
                                    model_class=AAE,
                                    epochs=MAX_EPOCHS)

    def load_data(self):
        filepath = "{}/data/Pollen/processed/pollen.{}g.txt".format(
            self.root_dir, self.input_size)
        df = pd.read_csv(filepath, sep="\t", header=0, index_col=0)
        features = df.iloc[:, 0:-2].values.astype(dtype=np.float64)
        features_scaled = preprocessing.scale(features)

        cell_ids = df.index.values
        cell_types = df.iloc[:, -2].values
        cell_subtypes = df.iloc[:, -1].values

        return cell_ids, features_scaled, cell_types, cell_subtypes

    def get_model_config(self, case_config):
        model_name = "{}_PollenAAE".format(self.case_counter)
        model_dir = self.get_model_dir(model_name)

        encoder_layers = []
        for size in case_config["encoder_layer_sizes"]:
            encoder_layers.append("Dense:{}:activation='elu'".format(size))
            if case_config["enc_batch_normalization"]:
                encoder_layers.append("BatchNormalization")

        discriminator_layers = []
        for index, size in enumerate(case_config["discriminator_layer_sizes"]):
            discriminator_layers.append(
                "Dense:{}:activation='elu'".format(size))
            if case_config["disc_batch_normalization"] and \
                    index != len(case_config["discriminator_layer_sizes"]):
                discriminator_layers.append("BatchNormalization")

        autoencoder_optimizer = "{}:lr={}".format(
            case_config["autoencoder_optimizer"]["name"],
            case_config["autoencoder_optimizer"]["lr"])
        if "momentum" in case_config["autoencoder_optimizer"]:
            autoencoder_optimizer += ":momentum={}:nesterov=True".format(
                case_config["autoencoder_optimizer"]["momentum"])

        if "discriminator_optimizer" in case_config:
            discriminator_optimizer = "{}:lr={}".format(
                case_config["discriminator_optimizer"]["name"],
                case_config["discriminator_optimizer"]["lr"])
            if "momentum" in case_config["discriminator_optimizer"]:
                discriminator_optimizer += ":momentum={}:nesterov=True".format(
                    case_config["discriminator_optimizer"]["momentum"])
        else:
            discriminator_optimizer = autoencoder_optimizer

        latent_dist = "gaussian:{}".format(case_config["latent_size"])
        prior_dist = "gaussian:{}:mean=0.0:stddev=1.0".format(
            case_config["latent_size"])
        output_dist = "mean_gaussian:{}".format(self.input_size)

        model_config = {
            "name": model_name,
            "model_dir": model_dir,

            "input_shape": (self.input_size,),
            "stochastic": True,
            "encoder_layers": encoder_layers,
            "discriminator_layers": discriminator_layers,
            "latent_distribution": latent_dist,
            "prior_distribution": prior_dist,
            "output_distribution": output_dist,

            "autoencoder_optimizer": autoencoder_optimizer,
            "discriminator_optimizer": discriminator_optimizer,
            "batch_size": case_config["batch_size"],

            "autoencoder_callbacks": {
                "file_logger": {"file": "autoencoder_model.training.log"},
                "early_stopping": {
                    "metric": "val_loss",
                    "min_delta": 0.1,
                    "patience": 5
                },
            },
            "discriminator_callbacks": {
                "file_logger": {"file": "discriminator_model.training.log"},
                "early_stopping": {
                    "metric": "val_loss",
                    "min_delta": 0.1,
                    "patience": 10
                },
            },
            "adversarial_callbacks": {
                "file_logger": {"file": "adversarial_model.training.log"}
            }
        }

        return model_config

    def train_final_aae(self, model_config):
        model_config["autoencoder_callbacks"]["checkpoint"] = {
            "metric": "loss",
            "file": "autoencoder_model.weights.h5"
        }
        model_config["discriminator_callbacks"]["checkpoint"] = {
            "metric": "loss",
            "file": "discriminator_model.weights.h5"
        }
        model_config["autoencoder_callbacks"]["early_stopping"]["metric"] = \
            "loss"
        model_config["discriminator_callbacks"]["early_stopping"]["metric"] = \
            "loss"

        results = self.train_final_model(model_config)
        final_aae = results["model"]
        full_dataset = results["dataset"]

        self.logger.info("Encoding latent represenations...")
        latent_reps = final_aae.encode(full_dataset.features)

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
            "discriminator_layers",
            "ae_optimizer",
            "disc_optimizer",
            "batch_size",
            "cv_ae_loss",
            "cv_disc_loss_prior",
            "cv_disc_loss_posterior",
            "cv_disc_loss",
            "cv_adv_loss",
            "cv_total_loss"
        ]]
        for result in trials.results:
            if None not in result["avg_valid_metrics"].values():
                losses.append((
                    result["model_config"],
                    result["avg_valid_metrics"]["loss"]))
            experiment_results.append([
                result["model_config"]["name"],
                # TODO: Format the encoder layers better
                result["model_config"]["encoder_layers"],
                result["model_config"]["latent_size"],
                result["model_config"]["discriminator_layers"],
                result["model_config"]["autoencoder_optimizer"],
                result["model_config"]["discriminator_optimizer"],
                result["model_config"]["batch_size"],
                result["avg_valid_metrics"]["ae_loss"],
                result["avg_valid_metrics"]["disc_loss_prior"],
                result["avg_valid_metrics"]["disc_loss_posterior"],
                result["avg_valid_metrics"]["disc_loss"],
                result["avg_valid_metrics"]["adv_loss"],
                result["avg_valid_metrics"]["loss"]
            ])
        save_data_table(
            experiment_results,
            self.experiment_dir + "/experiment_results.txt")
        self.logger.info("Saved experiment results")

        # Train the final AAE using the best model config
        best_loss_model_config = self.get_model_config(best_loss_case_config)
        best_loss_model_config["name"] = "PollenAAE_Final"

        self.train_final_aae(best_loss_model_config)
        self.logger.info("EXPERIMENT END")


class Train500gPrelimPollenAAE(TrainPollenAAE):
    def __init__(self, debug=False):
        super(Train500gPrelimPollenAAE, self).__init__(500, debug=debug)

    def hyperopt_search_space(self):
        return {
            "encoder_layer_sizes": hp.choice(
                "encoder_layer_sizes", [
                    [hp.choice("ae1_layer1", list(range(100, 1100, 100)))],
                    [
                        hp.choice("ae2_layer1", list(range(400, 1100, 100))),
                        hp.choice("ae2_layer2", list(range(100, 500, 100)))
                    ],
                    [
                        hp.choice("ae3_layer1", list(range(500, 1100, 100))),
                        hp.choice("ae3_layer2", [200, 300, 400, 500]),
                        hp.choice("ae3_layer3", [100, 200])
                    ]
                ]),
            "discriminator_layer_sizes": hp.choice(
                "discriminator_layer_sizes", [
                    [hp.choice("disc1_layer1", [10, 25, 50, 75, 100])],
                    [
                        hp.choice("disc2_layer1", [50, 75, 100]),
                        hp.choice("disc2_layer2", [10, 25, 50])
                    ],
                    [
                        hp.choice("disc3_layer1", [50, 75, 100]),
                        hp.choice("disc3_layer2", [25, 50]),
                        hp.choice("disc3_layer3", [10, 25])
                    ]
                ]),
            "latent_size": hp.choice(
                "latent_size", [10, 15, 20, 25, 50, 100]),
            "batch_size": hp.choice(
                "batch_size", [10, 25, 50]),
            "autoencoder_optimizer": hp.choice(
                "autoencoder_optimizer", [
                    {
                        "name": "sgd",
                        "lr": hp.loguniform(
                            "sgd_lr", -6 * log(10), -4 * log(10)),
                        "momentum": hp.choice(
                            "sgd_momentum", [0.9, 0.95, 0.99])
                    },
                    {
                        "name": "adam",
                        "lr": hp.loguniform(
                            "adam_lr", -6 * log(10), -2 * log(10))
                    }
                ]),
            "optimizer_lr": hp.loguniform(
                "optimizer_lr", -4 * log(10), -2 * log(10)),
            "enc_batch_normalization": hp.choice("batch_norm", [True, False]),
            "disc_batch_normalization": hp.choice("batch_norm", [True, False])
        }
