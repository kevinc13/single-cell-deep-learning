from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from .util import shuffle_in_unison
from .util import to_one_hot as convert_to_one_hot
import numpy as np


class Dataset:
    """ Defines a dataset for use in a ML model"""
    def __init__(self, 
                 features,
                 labels,
                 sample_data=[],
                 flatten=False,
                 to_one_hot=False):
        """
        Initialize a dataset

        Args:
            features: Numpy array of 'm' example features
            labels: Numpy array of 'm' example labels
            sample_data...: Extra Numpy arrays that contain information
                         about the samples
            flatten: Whether or not to flatten features into [m x n] array
            to_one_hot: Whether or not to convert labels into array of
                        one-hot vectors
            

        Usage:
            import numpy as np

            # Create random array of 10, 28x28x1 "images"
            images = np.rand(10, 28, 28, 1)

            # Create labels array of image classes (0=car, 1=person, 2=tree)
            labels = np.array([0, 1, 2, 1, 2, 1, 0, 0, 0, 1])

            # Create data set of images and convert
            # image labels to one-hot vectors
            image_dataset = DataSet(features, labels, to_one_hot=True)

            # Get next batch of 5 images
            (batch_features, batch_labels) = image_dataset.next_batch(5)
        """

        assert(features.shape[0] == labels.shape[0])
        self._features = features if not flatten else features.reshape(
            (features.shape[0], -1), order="F")
        self._labels = labels if not to_one_hot else convert_to_one_hot(
            np.squeeze(labels).astype(int))
        self._sample_data = sample_data

        self._num_examples = features.shape[0]
        self._epoch_count = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def sample_data(self):
        return self._sample_data

    def shuffle(self):
        self._features, self._labels, self._sample_data = unpack_tuple(
            shuffle_in_unison(
                self._features, self._labels, *self._sample_data), 2)

    def next_batch(self, batch_size):
        """
        Gets next batch of examples given a batch size

        Adapted from TF MNIST DataSet class code

        Args:
            batch_size: Batch size to use when dividing data

        Returns:
            Tuple of one batch's features and labels
        """
        assert(batch_size < self._num_examples)
        start = self._index_in_epoch

        if start == self._num_examples:
            # Finished 1 epoch
            self._epoch_count += 1

            # Shuffle dataset for next epoch
            self.shuffle()

            # Start next epoch
            start = 0
            self._index_in_epoch = 0

        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._index_in_epoch = self._num_examples

        end = self._index_in_epoch

        return (self._features[start:end], self._labels[start:end])

    @staticmethod
    def concatenate(*datasets, **kwargs):
        all_features = np.vstack(tuple([d.features for d in datasets]))
        all_labels = np.vstack(tuple([d.labels for d in datasets]))
        all_sample_data = np.column_stack(
            tuple([d.sample_data for d in datasets]))
        return Dataset(
            all_features, all_labels,
            sample_data=all_sample_data, **kwargs)
