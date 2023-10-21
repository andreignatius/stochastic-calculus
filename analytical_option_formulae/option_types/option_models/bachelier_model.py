"""
Authors     : Dylan Loo & Sarah Yin
Tests       : option_types.test_option_models.test_bachelier_model
Description : uses normal distribution of returns, allows for pricing of negative values
"""

from scipy.stats import norm
import math
from .abstract_option_model import AbstractOptionModel

class AbstractBachelierModel(AbstractOptionModel):
    def __init__(self, S:float, K:float, T:float, r:float, sigma:float):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma*self.S

        self.d1 = self._calculate_d1()

    def _calculate_d1(self):
        return (self.S - self.K) / (self.sigma * math.sqrt(self.T))

class VanillaBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        call_price = math.exp(-self.r * self.T) * \
            (((self.S - self.K) * norm.cdf(self.d1)) + \
            (math.sqrt(self.T) * self.sigma * norm.pdf(self.d1)))
        return call_price

    def calculate_put_price(self) -> float:
        put_price = math.exp(-self.r * self.T) * \
            (((self.K - self.S) * norm.cdf(-self.d1)) + \
            (math.sqrt(self.T) * self.sigma * norm.pdf(-self.d1)))
        return put_price

class DigitalCashOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        call_price = math.exp(-self.r * self.T) * norm.cdf(self.d1)
        return call_price

    def calculate_put_price(self) -> float:
        put_price = math.exp(-self.r * self.T) * norm.cdf(-self.d1)
        return put_price


class DigitalAssetOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        call_price = math.exp(-self.r * self.T) * \
            (((self.S) * norm.cdf(self.d1) + self.sigma * \
              math.sqrt(self.T) * norm.pdf(self.d1)))
        return call_price

    def calculate_put_price(self) -> float:
        put_price = math.exp(-self.r * self.T) * \
            (((self.S) * norm.cdf(-self.d1) + self.sigma * \
              math.sqrt(self.T) * norm.pdf(-self.d1)))
        return put_price