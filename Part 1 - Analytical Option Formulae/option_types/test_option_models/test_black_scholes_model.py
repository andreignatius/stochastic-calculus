import pytest
from option_types.option_models.black_scholes_model import *

# Example:
S = 100          # Stock price today
K = 100          # Strike price
T = 1            # Time until expiry (in years)
r = 0.05         # Risk-free rate
sigma = 0.2      # Volatility


vanillaBSM = VanillaBlackScholesModel(S, K, r, sigma, T)

# This will give you the price of European call and put options using the Black-Scholes formula.
def test_vanilla_call_price():
	EST_OPTION_VALUE = 10.45058
	tolerance = 0.00001
	call_price = vanillaBSM.calculate_call_price()
	assert call_price > 0
	assert call_price == pytest.approx(EST_OPTION_VALUE, tolerance)

def test_vanilla_put_price():
	EST_OPTION_VALUE = 5.57353
	tolerance = 0.00001
	put_price = vanillaBSM.calculate_put_price()
	assert put_price > 0
	assert put_price == pytest.approx(EST_OPTION_VALUE, tolerance)
