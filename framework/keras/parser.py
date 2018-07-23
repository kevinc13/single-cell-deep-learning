from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from keras import optimizers
from keras import layers
from . import distributions

optimizer_mappings = {
    "sgd": optimizers.SGD,
    "adam": optimizers.Adam,
    "rmsprop": optimizers.RMSprop,
    "adagrad": optimizers.Adagrad,
    "adadelta": optimizers.Adadelta,
    "adamax": optimizers.Adamax,
    "nadam": optimizers.Nadam
}

dist_mappings = {
    "gaussian": distributions.Gaussian,
    "mean_gaussian": distributions.MeanGaussian,
    "bernoulli": distributions.Bernoulli
}


class DefinitionParser:
    @staticmethod
    def parse_arguments(params):
        args = []
        named_args = dict()
        for p in params:
            if "=" in p:
                k, v = p.split("=")
                named_args[k] = eval(v)
            else:
                args.append(eval(p))

        return args, named_args

    @staticmethod
    def parse_optimizer_definition(optimizer_definition):
        params = optimizer_definition.split(":")
        if len(params) == 0:
            return None
        
        name = params.pop(0)
        params = dict(
            (k, eval(v)) for k, v in (p.split("=") for p in params))
        
        return optimizer_mappings[name](**params)

    @staticmethod
    def parse_layer_definition(layer_definition):
        all_params = layer_definition.split(":")
        if len(all_params) == 0:
            return None

        layer_name = all_params.pop(0)
        layer_ref = eval("layers." + layer_name)
        args, named_args = DefinitionParser.parse_arguments(all_params)

        return layer_ref, args, named_args

    @staticmethod
    def parse_distribution_definition(dist_definition):
        all_params = dist_definition.split(":")
        if len(all_params) == 0:
            return None

        dist_name = all_params.pop(0)
        args, named_args = DefinitionParser.parse_arguments(all_params)

        return dist_mappings[dist_name](*args, **named_args)
