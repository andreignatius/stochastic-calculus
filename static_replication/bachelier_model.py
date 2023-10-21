import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import math

from analytical_option_formulae.option_types.option_models.bachelier_model import VanillaBachelierModel

# Example:
S = 100  # Stock price today
K = 110  # Strike price
T = 1  # Time until expiry (in years)
r = 0.05  # Risk-free rate
sigma = 0.4  # Volatility


vanillaBachelier = VanillaBachelierModel(S, K, T, r, sigma*S)
print("Bachelier call price: ", vanillaBachelier.calculate_call_price())
print("Bachelier put price: ", vanillaBachelier.calculate_put_price())

def callintegrand_Bachelier(K):
    # vanillaBachelier.K = K
    vanillaBachelier = VanillaBachelierModel(S, K, T, r, sigma*S)
    price = vanillaBachelier.calculate_call_price()
    return price / K**2

def putintegrand_Bachelier(K):
    # vanillaBachelier.K = K
    vanillaBachelier = VanillaBachelierModel(S, K, T, r, sigma*S)
    price = vanillaBachelier.calculate_put_price()
    return price / K**2

F = S + np.exp(r*T)

I_put = quad(lambda x: putintegrand_Bachelier(x), 0.0, F)
I_call = quad(lambda x: callintegrand_Bachelier(x), F, 5000)
E_var = 2 * (I_put[0] + I_call[0])


print('The expected integrated variance is 000: %.9f' % E_var)

sigma = math.sqrt(E_var)
print('What sigma should we use: %.9f' % sigma)

vanillaBachelier = VanillaBachelierModel(S, K, T, r, sigma*S)
print("Bachelier call price: ", vanillaBachelier.calculate_call_price())
print("Bachelier put price: ", vanillaBachelier.calculate_put_price())
