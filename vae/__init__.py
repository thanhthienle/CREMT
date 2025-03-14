from .base import *
from .vanilla_vae import *
from .cvae import *

# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE

vae_models = {
    'VanillaVAE':VanillaVAE,
    'ConditionalVAE':ConditionalVAE,
}