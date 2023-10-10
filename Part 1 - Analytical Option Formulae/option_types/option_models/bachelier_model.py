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
        self.N_d1_cdf = self._calculate_cdf(self.d1)
        self.N_d1_pdf = self._calculate_pdf(self.d1)

    def _calculate_d1(self):
        return (self.S - self.K) \
               / (self.sigma * math.sqrt(self.T))
    
    # define CDF and PDF functions
    def _calculate_cdf(self, dist):
        return norm.cdf(dist)
    def _calculate_pdf(self, dist):
        return norm.pdf(dist)
    

class VanillaBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        # Define call price as per notes
        call_price = math.exp(-self.r * self.T) * \
            (((self.S - self.K) * self.N_d1_cdf) + \
            (math.sqrt(self.T) * self.N_d1_pdf))
        return call_price

    def calculate_put_price(self) -> float:
        # Define put price via put-call parity
        call_price = self.calculate_call_price()
        put_price = call_price - self.S + math.exp(-self.r * self.T) * self.K
        return put_price


class DigitalCashOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        call_price = math.exp(-self.r * self.T) * (self.cash * self.N_d1)
        return call_price

    def calculate_put_price(self) -> float:
        # Define put price via put-call parity
        call_price = self.calculate_call_price()
        put_price = call_price - self.S + math.exp(-self.r * self.T) * self.K
        return put_price


class DigitalAssetOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        call_price = math.exp(-self.r * self.T) * \
            ((self.S - self.K) * self.N_d1_cdf)
        return call_price

    def calculate_put_price(self) -> float:
        # Define put price via put-call parity
        call_price = self.calculate_call_price()
        put_price = call_price - self.S + math.exp(-self.r * self.T) * self.K
        return put_price