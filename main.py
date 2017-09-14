from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import argparse
import inspect
import importlib
import sys

import numpy as np
import tensorflow as tf

from framework.common.experiment import BaseExperiment


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
        tf.set_random_seed(args.seed)

    try:
        parts = args.experiment.split(".")
        module_path = "experiments.{}".format(".".join(parts[:-1]))
        class_name = parts[-1]

        experiment_module = importlib.import_module(module_path)
    except ImportError as err:
        sys.exit("Error importing experiment: {}".format(err))
    else:
        experiment_class = getattr(experiment_module, class_name)
        experiment = experiment_class(debug=args.debug)

        if isinstance(experiment, BaseExperiment):
            experiment.run()
        else:
            sys.exit(
                "Error: specified experiment is not of type BaseExperiment")


if __name__ == '__main__':
    main()
