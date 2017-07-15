import numpy as np


def shuffle_in_unison(a, b):
    permutation = np.arange(a.shape[0])
    np.random.shuffle(permutation)
    a = a[permutation]
    b = b[permutation]
    return a, b


def percentage_split(data, percentages=[]):
    prev_idx = 0
    size = len(data)
    cum_percentage = 0

    splits = []
    for p in percentages:
        cum_percentage += p
        next_idx = int(cum_percentage * size)
        splits.append(data[prev_idx:next_idx])
        prev_idx = next_idx
    
    return splits

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
