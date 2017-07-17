from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

from keras import optimizers
from keras import layers

optimizer_mappings = {
    "sgd": optimizers.SGD,
    "adam": optimizers.Adam,
    "rmsprop": optimizers.RMSprop,
    "adagrad": optimizers.Adagrad,
    "adadelta": optimizers.Adadelta,
    "adamax": optimizers.Adamax,
    "nadam": optimizers.Nadam
}

class DefinitionParser:

    @staticmethod
    def parse_optimizer_definition(optimizer_definition):
        params = optimizer_definition.split(":")
        if len(params) == 0:
            return None
        
        name = params.pop(0)
        params = dict(
            (k, float(v)) for k, v in (p.split("=") for p in params))
        
        return optimizer_mappings[name](**params)

    @staticmethod
    def parse_layer_definition(layer_definition):
        all_params = layer_definition.split(":")
        if len(all_params) == 0:
            return None

        layer_name = all_params.pop(0)
        layer_ref = eval("layers." + layer_name)

        args = []
        named_args = dict()
        for p in all_params:
            if "=" in p:
                k, v = p.split("=")
                named_args[k] = eval(v)
            else:
                args.append(eval(p))

        return layer_ref, args, named_args
