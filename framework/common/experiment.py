from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import os
import sys
import six
import csv
import logging
import numpy as np
from time import gmtime, strftime

from .dataset import Dataset

class BaseExperiment(object):

    def __init__(self):
        if not hasattr(self, "experiment_name") or \
                self.experiment_name is None:
            raise Exception("The experiment must specify an experiment name")

        self.root_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__))))
        self.results_dir = "{}/results".format(self.root_dir)
        self.experiment_dir = "{}/{}".format(
            self.results_dir, self.experiment_name)

        self.create_dir(self.experiment_dir)

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