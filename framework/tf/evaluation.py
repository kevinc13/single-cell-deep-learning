from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import tensorflow as tf
import numpy as np
import numpy.random as npr
#from sklearn.metrics import roc_auc_score, precision_score, recall_score


class ClassificationMetrics:

    def __init__(self, true, pred):
        self.true = true
        self.pred = pred

    def accuracy(self):
        return np.mean(np.equal(self.true, self.pred).astype(int))

    # def auROC(self):
    #     return roc_auc_score(self.true, self.pred)
    #
    # def precision(self):
    #     return precision_score(self.true, self.pred)
    #
    # def recall(self):
    #     return recall_score(self.true, self.pred)

    def __str__(self):
        return ("Accuracy: " + "{:.3f}".format(self.accuracy()))  # + " | "
#                "AUROC: " + "{:.3f}".format(self.auROC()) + "\n"
#                "Precision: " + "{:.3f}".format(self.precision()) + " | "
#                "Recall: " + "{:.3f}".format(self.recall()))

    def as_dict(self):
        metrics = {}
        metrics["accuracy"] = self.accuracy()

        return metrics
