from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import importlib
import sys
import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_string("e", "", "Name of the experiment to run")
flags.DEFINE_boolean("d", False, "Debug mode")

def main(_):
    run_config = flags.FLAGS.__flags.copy()

    if run_config["e"] == "":
        sys.exit("Error: Must provide an experiment name")

    try:
        module = importlib.import_module(
            "experiments.{}".format(run_config["e"]))
    except ImportError as err:
        sys.exit("Error importing experiment: {}".format(err))
    else:
        experiment = module.Experiment(debug=run_config["d"])
        experiment.run()

if __name__ == '__main__':
    tf.app.run()
