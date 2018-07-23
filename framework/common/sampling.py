from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from math import log10, sqrt
import numpy as np

from .dataset import Dataset
from .util import shuffle_in_unison, percentage_split, unpack_tuple


def log_uniform_sample(a, b):
    x = np.random.uniform(low=0.0, high=1.0)
    return 10**((log10(b) - log10(a))*x + log10(a))


def sqrt_uniform_sample(a, b):
    x = np.random.uniform(low=0.0, high=1.0)
    return (b - a) * sqrt(x) + a


def stratified_sample(features,
                      labels,
                      sample_data=None,
                      proportions=None,
                      convert_labels_to_int=False):
    if sample_data is None:
        sample_data = []
    if proportions is None:
        proportions = []

    # Randomize dataset
    features, labels, sample_data = \
        unpack_tuple(shuffle_in_unison(features, labels, *sample_data), 2)

    if convert_labels_to_int:
        strata, int_labels = np.unique(labels, return_inverse=True)
    else:
        strata = np.unique(labels)

    folds = []
    for i in range(len(proportions)):
        folds.append({
                "features": [],
                "labels": []
            })
        if len(sample_data) > 0:
            folds[-1]["sample_data"] = []
            for d in range(len(sample_data)):
                folds[-1]["sample_data"].append([])

    for s in strata:
        stratum_idx = labels == s
        stratum_features = features[stratum_idx,]

        if convert_labels_to_int:
            stratum_labels = int_labels[stratum_idx,]
        else:
            stratum_labels = labels[stratum_idx,]

        stratum_sample_data = [d[stratum_idx,] for d in sample_data]

        fold_indexes = percentage_split(
            np.arange(stratum_features.shape[0]), proportions)

        for i in range(len(proportions)):
            fold_idx = fold_indexes[i]

            folds[i]["features"] += stratum_features[fold_idx].tolist()
            folds[i]["labels"] += stratum_labels[fold_idx].tolist()

            for d_idx, d in enumerate(stratum_sample_data):
                folds[i]["sample_data"][d_idx] += d[fold_idx].tolist()

    datasets = []
    for i, fold in enumerate(folds):
        dataset = Dataset(
            np.array(fold["features"]), 
            np.array(fold["labels"]),
            sample_data=fold["sample_data"],
            to_one_hot=True)
        datasets.append(dataset)

    return datasets


def stratified_kfold(
        features, 
        labels,
        sample_data=None,
        n_folds=10,
        convert_labels_to_int=False):
    proportions = [1.0 / n_folds] * n_folds
    return stratified_sample(
        features, labels, sample_data=sample_data,
        proportions=proportions, convert_labels_to_int=convert_labels_to_int)


def kfold(features, sample_data=None, n_folds=10):
    if sample_data is None:
        sample_data = []

    proportions = [1.0 / n_folds] * n_folds

    # Randomize dataset
    features, sample_data = \
        unpack_tuple(shuffle_in_unison(features, *sample_data), 1)

    fold_indexes = percentage_split(np.arange(features.shape[0]), proportions)
    folds = []
    for i in range(len(proportions)):
        folds.append({"features": []})
        if len(sample_data) > 0:
            folds[-1]["sample_data"] = []

    for i in range(len(proportions)):
        fold_idx = fold_indexes[i]

        folds[i]["features"] += features[fold_idx].tolist()

        for d in sample_data:
            folds[i]["sample_data"].append(d[fold_idx].tolist())

    datasets = []
    for i, fold in enumerate(folds):
        dataset = Dataset(
            np.array(fold["features"]),
            np.array(fold["features"]),
            sample_data=fold["sample_data"])
        datasets.append(dataset)

    return datasets
