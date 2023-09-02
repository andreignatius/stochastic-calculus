from .abstract_option_type import AbstractOption
from .option_models.bachelier_model import *
from .option_models.black_76_model import *
from .option_models.black_scholes_model import *
from .option_models.displaced_diffusion_model import *


class DigitalAssetOrNothingOption(AbstractOption):
    def black_scholes_model(self) -> AbstractBlackScholesModel:
        return DigitalAssetOrNothingBlackScholesModel()

    def bachelier_model(self) -> AbstractBachelierModel:
        return DigitalAssetOrNothingBachelierModel()

    def black_76_model(self) -> AbstractBlack76Model:
        return DigitalAssetOrNothingBlack76Model()

    def displaced_diffusion_model(self) -> AbstractDisplacedDiffusionModel:
        return DigitalAssetOrNothingDisplacedDiffusionModel()
