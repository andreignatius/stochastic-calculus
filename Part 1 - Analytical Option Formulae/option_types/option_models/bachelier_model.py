"""
Authors     : Dylan Loo & Sarah Yin
Tests       : option_types.test_option_models.test_bachelier_model
Description : uses normal distribution of returns, allows for pricing of negative values
"""
from .abstract_option_model import AbstractOptionModel


class AbstractOptionModel:
    def __init__(self, parameters):
        self.parameters = parameters

class AbstractBachelierModel(AbstractOptionModel):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.calculate_d1_d2()

    def calculate_d1_d2(self):
        S = self.parameters['S']
        K = self.parameters['K']
        T = self.parameters['T']
        r = self.parameters['r']
        σ = self.parameters['σ']
        self.d1 = (S - K) / (σ * math.sqrt(T)) + 0.5 * σ * math.sqrt(T)
        self.d2 = self.d1 - σ * math.sqrt(T)
        self.N_d1 = norm.cdf(self.d1)
        self.N_d2 = norm.cdf(self.d2)
        self.N_minus_d1 = norm.cdf(-self.d1)
        self.N_minus_d2 = norm.cdf(-self.d2)
        return S, K, r, T, σ, self.N_d1, self.N_d2, self.N_minus_d1, self.N_minus_d2

    
class VanillaBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        S, K, r, T, _, self.N_d1, self.N_d2, _, _, = self.calculate_d1_d2()
        call_price = math.exp(-r * T) * \
            (S * self.N_d1 - K * self.N_d2)
        return call_price

    def calculate_put_price(self) -> float:
        S, K, r, T, _, _, _, self.N_minus_d1, _ = self.calculate_d1_d2()
        put_price = math.exp(-r * T) * \
            (K * self.N_minus_d2 - S * self.N_minus_d1)
        return put_price


class DigitalCashOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        _, _, r, T, _, self.N_d1, _, _, _ = self.calculate_d1_d2()
        cash_amount = self.parameters['cash_amount']
        call_price = math.exp(-r * T) * (cash_amount * self.N_d1)
        return call_price

    def calculate_put_price(self) -> float:
        _, _, r, T, _, _, _,self.N_minus_d1, _ = self.calculate_d1_d2()
        cash_amount = self.parameters['cash_amount']
        put_price = math.exp(-r * T) * (cash_amount * self.N_minus_d1)
        return put_price


class DigitalAssetOrNothingBachelierModel(AbstractBachelierModel):
    def calculate_call_price(self) -> float:
        S, _, r, T, _, self.N_d1, _, _, _ = self.calculate_d1_d2()
        call_price = math.exp(-r * T) * self.N_d1
        return call_price

    def calculate_put_price(self) -> float:
        S, _, r, T, _, _, self.N_minus_d1, _, _ = self.calculate_d1_d2()
        put_price = math.exp(-r * T) * self.N_minus_d1
        return put_price