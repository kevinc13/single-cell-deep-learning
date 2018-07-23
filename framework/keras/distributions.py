from math import log, exp
import numpy as np

import keras.backend as K
from keras.layers import Dense, Lambda
from keras.losses import mean_squared_error


class Distribution:
    def __init__(self, size):
        self._size = size
        self.built = False

    @property
    def size(self):
        return self._size


class Gaussian(Distribution):
    def __init__(self, size, mean=None, stddev=None, name="z"):
        super(Gaussian, self).__init__(size)
        self.name = name

        if mean is not None and stddev is not None:
            self.fixed = True
            self.mean = mean
            self.log_var = log(stddev ** 2)
        else:
            self.fixed = False
            self.mean = None
            self.log_var = None
            self.mean_layer = None
            self.log_var_layer = None

    def sample(self, args):
        if self.fixed:
            if not isinstance(args, int):
                raise Exception("Sampling from a fixed Gaussian distribution "
                                "requires specification of batch size")
            else:
                batch_size = args
                epsilon = np.random.standard_normal((batch_size, self._size))
                return self.mean + epsilon * exp(self.log_var / 2)
        else:
            mean, log_var = args
            epsilon = K.random_normal(shape=(K.shape(mean)[0], self._size),
                                      mean=0., stddev=1.0)
            return mean + epsilon * K.exp(log_var / 2)

    def negative_log_likelihood(self, y_true):
        with K.name_scope("negative_log_likelihood"):
            return - K.sum(
                -0.5 * np.log(2 * np.pi)
                - 0.5 * self.log_var
                - 0.5 * K.square(y_true - self.mean)
                / K.exp(self.log_var),
                axis=-1)

    def __call__(self, x, name=None):
        if not self.built:
            self.mean_layer = Dense(self._size, name=self.name + "_mean")
            self.log_var_layer = Dense(self._size, name=self.name + "_log_var")
            self.built = True
        self.mean = self.mean_layer(x)
        self.log_var = self.log_var_layer(x)
        output = Lambda(self.sample, name=name)(
            [self.mean, self.log_var])
        return output


class MeanGaussian(Gaussian):
    def __call__(self, x, name=None):
        if not self.built:
            self.mean_layer = Dense(self._size, name=name)
            self.log_var_layer = Dense(self._size)
            self.built = True
        self.mean = self.mean_layer(x)
        self.log_var = self.log_var_layer(x)
        return self.mean


class Bernoulli(Distribution):
    def __init__(self, size, p=None):
        super(Bernoulli, self).__init__(size)
        if p is None:
            self.fixed = False
            self.p = None
        else:
            self.fixed = True
            self.p = p
            self.p_layer = None

    def sample(self, batch_size):
        return K.cast(K.less(K.random_uniform(
                shape=(batch_size, self._size)), self.p), dtype="float32")

    def negative_log_likelihood(self, y_true):
        return K.sum(K.binary_crossentropy(y_true, self.p), axis=-1)

    def __call__(self, x, name=None):
        if not self.built:
            self.p_layer = Dense(self._size, activation="sigmoid", name=name)
            self.built = True

        self.p = self.p_layer(x)
        return self.p


class Categorical(Distribution):
    def __init__(self, size):
        super(Categorical, self).__init__(size)
        self.output = None
        self.output_layer = None

    def sample(self, batch_size):
        p = [1.0/self.size] * self.size
        return np.random.multinomial(1, p, size=batch_size)

    def __call__(self, x, name=None):
        if not self.built:
            self.output_layer = Dense(
                self._size, activation="softmax", name=name)
            self.built = True

        self.output = self.output_layer(x)
        return self.output


class Deterministic(Distribution):
    def __init__(self, size, continuous=False):
        super(Deterministic, self).__init__(size)
        self.continuous = continuous
        self.output = None
        self.output_layer = None

    def negative_log_likelihood(self, y_true):
        if self.continuous:
            return mean_squared_error(y_true, self.output)
        else:
            return K.sum(K.binary_crossentropy(y_true, self.output), axis=-1)

    def __call__(self, x, name=None):
        if not self.built:
            activation = "linear" if self.continuous else "sigmoid"
            self.output_layer = Dense(
                self._size, activation=activation, name=name)
            self.built = True
        self.output = self.output_layer(x)
        return self.output
