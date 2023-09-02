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
	call_price = vanillaBSM.calculate_call_price()
	print(f"Call Option Price: ${call_price:.2f}")
	assert call_price > 0

def test_vanilla_put_price():
	put_price = vanillaBSM.calculate_put_price()
	print(f"Put Option Price: ${put_price:.2f}")
	assert put_price > 0

