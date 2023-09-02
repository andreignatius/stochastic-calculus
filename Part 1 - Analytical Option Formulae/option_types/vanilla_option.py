from .abstract_option_type import AbstractOption

from .option_models.black_scholes_model import *
from .option_models.bachelier_model import *
from .option_models.black_76_model import *
from .option_models.displaced_diffusion_model import *


class VanillaOption(AbstractOption):
    def black_scholes_model(
        self, S: float, K: float, r: float, sigma: float, T: float
    ) -> AbstractBlackScholesModel:
        return VanillaBlackScholesModel(S, K, r, sigma, T)  # TODO: Verify params

    def bachelier_model(self) -> AbstractBachelierModel:
        return VanillaBachelierModel()  # TODO: Add params

    def black_76_model(self) -> AbstractBlack76Model:
        return VanillaBlack76Model()  # TODO: Add params

    def displaced_diffusion_model(self) -> AbstractDisplacedDiffusionModel:
        return VanillaDisplacedDiffusionModel()  # TODO: Add params
