import numpy as np

from .dataset import Dataset

def stratified_sample(features,
                      labels,
                      prop_train=0.8,
                      convert_labels_to_int=False):
    if convert_labels_to_int:
        strata, int_labels = np.unique(labels, return_inverse=True)
    else:
        strata = np.unique(labels)

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for s in strata:
        stratum_idx = labels == s
        stratum_features = features[stratum_idx,]

        if convert_labels_to_int:
            stratum_labels = int_labels[stratum_idx,]
        else:
            stratum_labels = labels[stratum_idx,]

        n_examples = stratum_features.shape[0]
        train_idx = np.random.choice(
            n_examples,
            size=math.ceil(prop_train * n_examples),
            replace=False)
        test_idx = [x for x in range(n_examples) if x not in train_idx]

        train_features = train_features + \
            stratum_features[train_idx,].tolist()
        train_labels = train_labels + stratum_labels[train_idx,].tolist()

        test_features = test_features + stratum_features[test_idx,].tolist()
        test_labels = test_labels + stratum_labels[test_idx,].tolist()

    train_dataset = Dataset(
        np.array(train_features), np.array(train_labels), to_one_hot=True)
    test_dataset = Dataset(
        np.array(test_features), np.array(test_labels), to_one_hot=True)

    train_dataset.shuffle()
    test_dataset.shuffle()

    return train_dataset, test_dataset