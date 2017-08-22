from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import importlib
import sys
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Name of the experiment to run",
                        type=str)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true")
    parser.add_argument("-s", "--seed",
                        help="Seed to use for reproducible results",
                        type=int)
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
        import tensorflow as tf
        tf.set_random_seed(args.seed)

    try:
        experiment_module = importlib.import_module(
            "experiments.{}".format(args.experiment))
    except ImportError as err:
        sys.exit("Error importing experiment: {}".format(err))
    else:
        experiment = experiment_module.Experiment(debug=args.debug)
        experiment.run()


if __name__ == '__main__':
    main()
