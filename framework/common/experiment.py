import os
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

    def current_time(self):
        return strftime("%m-%d-%y_%H:%M:%S", gmtime())