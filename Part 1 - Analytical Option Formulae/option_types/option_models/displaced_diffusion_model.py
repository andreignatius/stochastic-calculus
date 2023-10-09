"""
Authors     : Andre Lim & Joseph Adhika
Tests       : option_types.test_option_models.test_displaced_diffusion_model.py
Description : Mark Rubinstein's Displaced Diffusion Model following lecture notes
"""
import numpy as np
from scipy.stats import norm
from .abstract_option_model import AbstractOptionModel


class AbstractDisplacedDiffusionModel(AbstractOptionModel):
    """
    Displaced diffusion is extension of Black-Scholes with an additional parameter beta
    ...
    Parameters
    ----------
    S : float
        The current price of the underlying asset
    K : float
        The strike price of the options
    r : float
        Risk free interest rate (decimal)
    sigma : float
        Volatility
    T : float
        Maturity period (years)
    beta : float
        Displaced diffusion model parameter (0,1], but lecture notes say [0,1]
        https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=6976&context=lkcsb_research
    """

    def __init__(self, S: float, K: float, r: float, sigma: float, T: float, beta: float):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.beta = beta

        self.adjusted_S = self.S/self.beta
        self.adjusted_K = self.K + ((1-self.beta)/self.beta)*self.S
        self.adjusted_sigma = self.sigma * self.beta
        
        self.d1 = self._calculate_d1()
        self.d2 = self._calculate_d2()

    def _calculate_d1(self) -> float:
        return (np.log(self.adjusted_S / self.adjusted_K) + (self.r + 0.5 * self.adjusted_sigma**2) * self.T) / (
            self.adjusted_sigma * np.sqrt(self.T)
        )

    def _calculate_d2(self) -> float:
            return self.d1 - self.adjusted_sigma * np.sqrt(self.T)
            
    def _calculate_d2_digital_cash_or_nothing(self) -> float:
        return (np.log(self.adjusted_S / self.adjusted_K) + (self.r - 0.5 * self.adjusted_sigma**2) * self.T) / (self.adjusted_sigma * np.sqrt(self.T))

class VanillaDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        return self.adjusted_S * norm.cdf(self.d1) - self.adjusted_K * np.exp(
            -self.r * self.T
        ) * norm.cdf(self.d2)

    def calculate_put_price(self) -> float:
        return self.adjusted_K * np.exp(-self.r * self.T) * norm.cdf(
            -self.d2
        ) - self.adjusted_S * norm.cdf(-self.d1)

class DigitalCashOrNothingDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        return np.exp(-self.r * self.T) * norm.cdf(self._calculate_d2_digital_cash_or_nothing())

    def calculate_put_price(self) -> float:
        return np.exp(-self.r * self.T) * (1 - norm.cdf(self._calculate_d2_digital_cash_or_nothing()))


class DigitalAssetOrNothingDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        return self.adjusted_S * norm.cdf(self.d1)

    def calculate_put_price(self) -> float:
        return self.adjusted_S * (1 - norm.cdf(self.d1))

