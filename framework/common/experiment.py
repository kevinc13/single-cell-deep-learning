from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import csv
import logging
import os
import sys
from time import gmtime, strftime

import six
from hyperopt import fmin, tpe, Trials, space_eval


class BaseExperiment(object):

    def __init__(self):
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

    def run(self, debug=False):
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
    def hyperopt_search_space(self):
        raise Exception(
            "The hyperopt experiment must implement \
            the 'hyperopt_search_space' method")

    def run_hyperopt(self, objective_func, n_evals):
        trials = Trials()
        if self.debug:
            n_evals = 2
        search_space = self.hyperopt_search_space()
        best = fmin(objective_func,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=n_evals,
                    trials=trials)
        return trials, best, space_eval(search_space, best)
