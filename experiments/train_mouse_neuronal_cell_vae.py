import numpy as np
import math

from framework.common.dataset import Dataset
from framework.common.experiment import BaseExperiment
from framework.common.sampling import stratified_sample
from framework.keras.autoencoder import VariationalAutoencoder as VAE

class Experiment(BaseExperiment):
    def __init__(self):
        self.experiment_name = "train_mouse_neuronal_cell_vae"
        super(Experiment, self).__init__()

    def load_data(self):
        df = np.array(self.read_data_table(
            "{}/data/Usokin/processed/mouse_neuronal_cells.500g.txt"
            .format(self.root_dir)))
        features = df[1:,1:-2]

        cell_types = df[1:,-2]
        cell_subtypes = df[1:, -1]

        return stratified_sample(
            features, cell_subtypes, convert_labels_to_int=True)

    def run(self):
        train_dataset, valid_dataset = self.load_data()

        # Setup hyperparameter/architecture search space

        # Start training!