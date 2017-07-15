import numpy as np

from .dataset import Dataset
from .util import shuffle_in_unison, percentage_split


def log_uniform_sample(a, b):
    x = np.random.uniform(low=0.0, high=1.0)
    return 10**((log10(b) - log10(a))*x + log10(a))


def sqrt_uniform_sample(a, b):
    x = np.random.uniform(low=0.0, high=1.0)
    return (b - a) * sqrt(x) + a


def stratified_sample(features,
                      labels,
                      proportions=[],
                      convert_labels_to_int=False):
    # Randomize dataset
    features, labels = shuffle_in_unison(features, labels)

    if convert_labels_to_int:
        strata, int_labels = np.unique(labels, return_inverse=True)
    else:
        strata = np.unique(labels)

    folds = [] # [(features1, labels1), (features2, labels2), ...]

    for i in range(len(proportions)):
        folds.append({
                "features": [],
                "labels": []
            })

    for s in strata:
        stratum_idx = labels == s
        stratum_features = features[stratum_idx,]

        if convert_labels_to_int:
            stratum_labels = int_labels[stratum_idx,]
        else:
            stratum_labels = labels[stratum_idx,]

        fold_indexes = percentage_split(
            np.arange(stratum_features.shape[0]), proportions)

        for i in range(len(proportions)):
            fold_idx = fold_indexes[i]

            folds[i]["features"] = folds[i]["features"] + \
                stratum_features[fold_idx,].tolist()
            folds[i]["labels"] = folds[i]["labels"] + \
                stratum_labels[fold_idx,].tolist()

    datasets = []
    for fold in folds:
        dataset = Dataset(
            np.array(fold["features"]), 
            np.array(fold["labels"]), 
            to_one_hot=True)
        dataset.shuffle()
        datasets.append(dataset)

    return datasets

def stratified_kfold(
        features, labels, n_folds=10, convert_labels_to_int=False):
    proportions = [1.0 / n_folds] * n_folds
    print(proportions)
    return stratified_sample(
        features, labels,
        proportions, convert_labels_to_int)
