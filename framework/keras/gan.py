from keras.layers import Input

from framework.keras.core import BaseModel
from framework.keras.parser import DefinitionParser


class GAN(BaseModel):
    def __init__(self, config, restore=False):
        self.latent_size = None

        self.generator_layers = None
        self.discriminator_layers = None

        self.generator_model = None
        self.discriminator_model = None
        self.keras_model = None

        super(GAN, self).__init__(config, restore=False)

    def setup(self):
        if "latent_size" in self.config:
            self.latent_size = self.config["latent_size"]
        else:
            raise Exception("Must specify the size of the "
                            "generator's noise prior")

    def build(self):
        generator_x = Input(shape=(self.latent_size,))
        generator_layer_defs = self.config["generator_layers"]

        # Generator
        for index, layer_def in enumerate(generator_layer_defs):
            layer_ref, args, kw_args = \
                DefinitionParser.parse_layer_definition(layer_def)
            layer = layer_ref(*args, **kw_args)

            if index == 0:
                self.generator_layers.append(layer(self.x))
            else:
                self.generator_layers.append(layer(self.generator_layers[-1]))

        discriminator_x = Input(shape=(self.config["input_size"],))
        discriminator_layer_defs = self.config["discriminator_layers"]

        pass


class WassersteinGAN(GAN):
    def setup(self):
        pass

    def build(self):
        pass

    def train(self):
        pass
