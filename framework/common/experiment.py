from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import csv
import logging
import os
import sys
from time import gmtime, strftime

import six
import numpy as np
from collections import Iterable
from hyperopt import fmin, tpe, Trials, space_eval, STATUS_OK, STATUS_FAIL

from framework.common.dataset import Dataset


class BaseExperiment(object):

    def __init__(self, debug=False):
        self.debug = debug

        self.experiment_name = None
        self.root_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__))))
        self.results_dir = None
        self.experiment_dir = None

        self.logger = None

    def setup_dir(self):
        if not hasattr(self, "experiment_name") or \
                self.experiment_name is None:
            raise Exception("The experiment must specify an experiment name")

        self.results_dir = "{}/results".format(self.root_dir)
        self.experiment_dir = "{}/{}".format(
            self.results_dir, self.experiment_name)

        self.create_dir(self.experiment_dir)

    def setup_logger(self):
        # Setup logger
        self.logger = logging.getLogger(self.experiment_name)
        if len(self.logger.handlers) == 0:
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(name)s] %(asctime)s - %(message)s")

            if self.experiment_dir is not None:
                file_handler = logging.FileHandler(
                    "{0}/experiment.log".format(self.experiment_dir))
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)

    def run(self):
        raise Exception("The experiment must implement the run method")

    def create_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def get_model_dir(self, model_name):
        return "{}/{}".format(self.experiment_dir, model_name)

    def read_data_table(self, filepath, delimiter="\t"):
        with open("{}/{}".format(self.root_dir, filepath), "r") as f:
            data = []
            for line in f.readlines():
                data.append(line.replace("\n", "").split(delimiter))

            return data

    def save_data_table(self, data, filepath, root=None, delimiter="\t"):
        if root is not None:
            filepath = root + "/" + filepath

        delimiter = str(delimiter) if six.PY2 else delimiter

        with open(filepath, "w") as f:
            writer = csv.writer(
                f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
            for r in data:
                writer.writerow(r)

    def current_time(self):
        return strftime("%m-%d-%y_%H:%M:%S", gmtime())


class HyperoptExperiment(BaseExperiment):
    def __init__(self, debug=False):
        super(HyperoptExperiment, self).__init__(debug)
        self.n_evals = 2

    def setup_hyperopt(self, n_evals):
        if not self.debug:
            self.n_evals = n_evals

    def hyperopt_search_space(self):
        raise Exception(
            "The hyperopt experiment must implement \
            the 'hyperopt_search_space' method")

    def run_hyperopt(self, objective_func):
        trials = Trials()
        search_space = self.hyperopt_search_space()
        best = fmin(objective_func,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=self.n_evals,
                    trials=trials)
        return trials, best, space_eval(search_space, best)


class CrossValidationExperiment(BaseExperiment):
    def __init__(self, debug):
        super(CrossValidationExperiment, self).__init__(debug)
        self.case_counter = 0
        self.n_folds = 0
        self.model_class = None
        self.datasets = []

    def setup_cross_validation(self, n_folds, datasets, model_class):
        self.n_folds = n_folds
        self.datasets = datasets
        self.model_class = model_class

    def get_model_config(self, case_config):
        raise Exception("Experiment must implement the \
                        'get_model_config' method")

    def train_case_model(self, case_config, epochs=100,
                         batch_size=None, loss_metric="loss"):
        model_config = self.get_model_config(case_config)
        self.create_dir(model_config["model_dir"])

        if self.logger is not None:
            self.logger.info("Training %s..." % model_config["name"])

        status = STATUS_OK
        avg_valid_metrics = {}
        for k in range(0, self.n_folds):
            train_dataset = Dataset.concatenate(
                *(self.datasets[:k] + self.datasets[(k + 1):])
            )
            valid_dataset = self.datasets[k]

            model = self.model_class(model_config)

            if batch_size is None:
                if "batch_size" in model_config:
                    batch_size = model_config["batch_size"]
                elif "batch_size" in case_config:
                    batch_size = case_config["batch_size"]
                else:
                    raise Exception("No batch size specified \
                                    for model training")

            if self.debug: epochs = 2
            model.train(train_dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_dataset=valid_dataset)

            eval_metrics = model.evaluate(valid_dataset)
            if not isinstance(eval_metrics, Iterable):
                eval_metrics = [eval_metrics]

            fold_valid_metrics = dict(zip(model.keras_model.metrics_names,
                                          eval_metrics))

            if np.any(np.isnan(list(fold_valid_metrics.values()))) or \
                    np.any(np.isinf(list(fold_valid_metrics.values()))):
                for key in fold_valid_metrics.keys():
                    avg_valid_metrics[key] = None
                status = STATUS_FAIL
                break
            else:
                for name, value in fold_valid_metrics.items():
                    if name in avg_valid_metrics:
                        avg_valid_metrics[name] += value
                    else:
                        avg_valid_metrics[name] = value

                    if self.logger is not None:
                        self.logger.info("{}|Fold #{}|{} = {:f}".format(
                            model_config["name"], k + 1, name, value))

            if self.debug:
                break

        if status != STATUS_FAIL:
            for name, metric in avg_valid_metrics.items():
                metric /= self.n_folds
                if self.logger is not None:
                    self.logger.info("{}|Avg {} = {:f}".format(
                        model_config["name"], name, metric))

        self.case_counter += 1

        return {
            "status": status,
            "model_config": model_config,
            "loss": avg_valid_metrics[loss_metric],
            "avg_valid_metrics": avg_valid_metrics
        }

    def train_final_model(self, model_config, epochs=100, batch_size=None):
        model_dir = self.get_model_dir(model_config["name"])
        self.create_dir(model_dir)
        model_config["model_dir"] = model_dir

        if batch_size is None:
            if "batch_size" in model_config:
                batch_size = model_config["batch_size"]
            else:
                raise Exception("No batch size specified \
                                for model training")

        full_dataset = Dataset.concatenate(*self.datasets)

        if self.logger is not None:
            self.logger.info("Training Final Model: {}".format(
                model_config["name"]))

        model = self.model_class(model_config)
        if self.debug: epochs = 2
        model.train(full_dataset, epochs=epochs,
                    batch_size=batch_size,
                    validation_dataset=full_dataset)

        eval_metrics = model.evaluate(full_dataset)
        if not isinstance(eval_metrics, Iterable):
            eval_metrics = [eval_metrics]

        metrics = dict(zip(model.keras_model.metrics_names,
                           eval_metrics))

        if self.logger is not None:
            for k, v in metrics.items():
                self.logger.info("{}|{} = {:f}".format(
                    model_config["name"], k, v
                ))

        return {
            "model": model,
            "dataset": full_dataset,
            "metrics": metrics
        }
