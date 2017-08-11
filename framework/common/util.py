from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np
import math


def shuffle_in_unison(*arrays):
    assert(len(arrays) > 0)

    permutation = np.arange(arrays[0].shape[0])
    np.random.shuffle(permutation)
    
    result = []
    for a in arrays:
        result.append(a[permutation])
    return tuple(result)


def percentage_split(data, percentages=[]):
    prev_idx = 0
    size = len(data)
    cum_percentage = 0

    splits = []
    for p in percentages:
        cum_percentage += p
        next_idx = int(math.ceil(cum_percentage * size))
        splits.append(data[prev_idx:next_idx])
        prev_idx = next_idx
    
    return splits


def unpack_tuple(data, n=2):
    for i in range(n):
        yield data[i]
    yield data[n:]

def to_one_hot(vector):
        """
        Converts vector of values into one-hot array

        Args:
            vector: Numpy array/vector

        Returns:
            Numpy array of one-hot encoded values
        """

        # Get number of classes from max value in vector
        num_classes = np.max(vector) + 1

        # Create array of zeros
        result = np.zeros(shape=(vector.shape[0], num_classes))

        # Set appropriate values to 1
        result[np.arange(vector.shape[0]), vector] = 1

        # Return as integer NumPy array
        return result.astype(int)


def from_one_hot(vector):
        """
        Converts one-hot array into vector of numbers
        """
        return np.array([np.where(r == 1)[0][0] for r in vector])
