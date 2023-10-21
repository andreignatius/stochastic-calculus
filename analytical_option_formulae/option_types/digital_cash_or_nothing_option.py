from .abstract_option_type import AbstractOption
from .option_models.bachelier_model import *
from .option_models.black_76_model import *
from .option_models.black_scholes_model import *
from .option_models.displaced_diffusion_model import *


class DigitalCashOrNothingOption(AbstractOption):
    def black_scholes_model(self) -> AbstractBlackScholesModel:
        return DigitalCashOrNothingBlackScholesModel()

    def bachelier_model(self) -> AbstractBachelierModel:
        return DigitalCashOrNothingBachelierModel()

    def black_76_model(self) -> AbstractBlack76Model:
        return DigitalCashOrNothingBlack76Model()

    def displaced_diffusion_model(self) -> AbstractDisplacedDiffusionModel:
        return DigitalCashOrNothingBlack76Model()
