"""
Authors     : Andre Lim & Joseph Adhika
Tests       : option_types.test_option_models.test_black_76_model
Description : uses lognormal distribution of returns, strictly prices non-zero values
"""
from .abstract_option_model import AbstractOptionModel
import numpy as np
from scipy.stats import norm

class AbstractBlackScholesModel(AbstractOptionModel):
    def __init__(self, S: float, K: float, r: float, sigma: float, T: float):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T


class VanillaBlackScholesModel(AbstractBlackScholesModel):
    def calculate_call_price(self) -> float:
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def calculate_put_price(self) -> float:
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)


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
