from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import sys
import numpy as np

from framework.common.dataset import Dataset
from framework.common.experiment import BaseExperiment
from framework.common.sampling import stratified_sample
from framework.keras.supervised import DeepNeuralNetwork as DNN

class Experiment(BaseExperiment):
    def __init__(self, debug=False):
        self.debug = debug
        self.n_evals = 50

        self.experiment_name = "train_usokin-100g-dnn"
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
        return None     

    def run(self, debug=False):
        # Load Usokin dataset
        cell_ids, features, cell_types, cell_subtypes = self.load_data()
        train_dataset, valid_dataset, test_dataset = tuple(stratified_sample(
            features, cell_types,
            [cell_ids, cell_types, cell_subtypes],
            proportions=[0.6, 0.2, 0.2],
            convert_labels_to_int=True))

        model_name = "UsokinDNN"
        model_dir = self.get_model_dir(model_name)

        create_dir(model_dir)

        model_config = {
            "name": model_name,
            "model_dir": model_dir,

            "layers": [
                "Dense:200:activation='relu':input_shape=(100,)",
                "Dense:4:activation='softmax'"
            ],

            "loss": "categorical_crossentropy",
            "optimizer": "adam",
            "metrics": ["accuracy"]
        }
    
        epochs = 5 if self.debug else 50
        dnn = DNN(model_config)
        dnn.train(train_dataset,
                  epochs=epochs, batch_size=50,
                  validation_dataset=valid_dataset)

        loss, accuracy = tuple(dnn.evaluate(test_dataset))
        print("Test Accuracy: ", accuracy)

        print("Dataset Verification:")
        for i in np.random.choice(np.arange(test_dataset.num_examples), 5):
            print("Sample ID: ", test_dataset.sample_data[0][i], end="|")
            print("Sample Label: ", format(test_dataset.labels[i]))

