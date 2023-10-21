import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

from analytical_option_formulae.option_types.option_models.bachelier_model import VanillaBachelierModel

# Example:
S = 100  # Stock price today
K = 110  # Strike price
T = 1  # Time until expiry (in years)
r = 0.05  # Risk-free rate
sigma = 3  # Volatility


vanillaBachelier = VanillaBachelierModel(S, K, T, r, sigma*S)

def callintegrand_Bachelier():
    price = vanillaBachelier.calculate_call_price()
    return price / K**2

def putintegrand_Bachelier():
    price = vanillaBachelier.calculate_put_price()
    return price / K**2

F = S + r * T

# vanillaBachelier = VanillaBachelierModel(S, K, r, sigma, T)

I_put = quad(lambda x: putintegrand_Bachelier(), 0.0, F)
I_call = quad(lambda x: callintegrand_Bachelier(), F, 5000)
E_var = 2 * (I_put[0] + I_call[0])

print("Bachelier call price: ", vanillaBachelier.calculate_call_price())
print("Bachelier put price: ", vanillaBachelier.calculate_put_price())
print('The expected integrated variance is: %.9f' % E_var)

