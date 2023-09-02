"""
Authors     : Andre Lim & Joseph Adhika
Tests       : option_types.test_option_models.test_black_76_model
Description : uses lognormal distribution of returns, strictly prices non-zero values
"""
from .abstract_option_model import AbstractOptionModel


class AbstractBlackScholesModel(AbstractOptionModel):
    def __init__(self, S: float, K: float, r: float, sigma: float, T: float):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T


class VanillaBlackScholesModel(AbstractBlackScholesModel):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae


class DigitalCashOrNothingBlackScholesModel(AbstractBlackScholesModel):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae


class DigitalAssetOrNothingBlackScholesModel(AbstractBlackScholesModel):
    def calculate_call_price(self) -> float:
        pass  # TODO: add formulae

    def calculate_put_price(self) -> float:
        pass  # TODO: add formulae
