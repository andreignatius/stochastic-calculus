"""
Authors     : Dylan Loo & Sarah Yin
Tests       : option_types.test_option_models.test_bachelier_model
Description : uses normal distribution of returns, allows for pricing of negative values
"""

from scipy.stats import norm
import math
from .abstract_option_model import AbstractOptionModel

class AbstractBachelierModel(AbstractOptionModel):
    def __init__(self, S:float, K:float, T:float, r:float, sigma:float, cash:float):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.cash = cash
        self.sigma = sigma
        
        self.d1 = self._calculate_d1()
        self.d2 = self._calculate_d2()

        self.N_d1_cdf = self._calculate_cdf(self.d1)
        self.N_d2_cdf = self._calculate_cdf(self.d2)

        self.N_d1_pdf = self._calculate_pdf(self.d1)
        self.N_d2_pdf = self._calculate_pdf(self.d2)

        self.N_minus_d1 = self._calculate_cdf(-self.d1)
        self.N_minus_d2 = self._calculate_cdf(-self.d2)

    def _calculate_d1(self):
        return (self.S - self.K) \
               / (self.sigma * math.sqrt(self.T)) \
               + (self.sigma*(self.T**0.5))/2
    
    def _calculate_d2(self):
        return self.d1 - self.sigma * math.sqrt(self.T)
    
    def _calculate_cdf(self, dist):
        return norm.cdf(dist)
    
    def _calculate_pdf(self, dist):
        return norm.pdf(dist)
    

class VanillaBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        call_price = (math.exp(-self.r * self.T) \
            * (self.S - self.K)) \
            * self.N_d1_cdf\
            + (self.sigma * self.T**0.5 * self.N_d1_pdf)
        return call_price

    def calculate_put_price(self) -> float:
        put_price = (math.exp(-self.r * self.T) \
            * (self.K - self.S)) \
            * self.N_d2_cdf \
            + (self.sigma * self.T**0.5 * self.N_d2_pdf)
        return put_price


class DigitalCashOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        call_price = math.exp(-self.r * self.T) * (self.cash * self.N_d1)
        return call_price

    def calculate_put_price(self) -> float:
        put_price = math.exp(-self.r * self.T) * (self.cash * self.N_minus_d1)
        return put_price


class DigitalAssetOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        call_price = math.exp(-self.r * self.T) * self.N_d1
        return call_price

    def calculate_put_price(self) -> float:
        put_price = math.exp(-self.r * self.T) * self.N_minus_d1
        return put_price