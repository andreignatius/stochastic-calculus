import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import math

from analytical_option_formulae.option_types.option_models.bachelier_model import VanillaBachelierModel

# Start Date: 1-Dec-2020
# End Date: 15-Jan-2021
# The time difference is 45 days (considering 31 days in December and 14 days in January).

# In a year with 365 days, the time to maturity T is: 45 / 365

# Example:
S = 100.0  # Stock price today
K = 110.0  # Strike price
T = 45/365  # Time until expiry (in years)
# T = 1
r = 0.05  # Risk-free rate
sigma = 0.4  # Volatility


vanillaBachelier = VanillaBachelierModel(S, K, T, r, sigma*S)
print("Bachelier call price: ", vanillaBachelier.calculate_call_price())
print("Bachelier put price: ", vanillaBachelier.calculate_put_price())


F = S * np.exp(r*T)

def callintegrand_Bachelier(K):
    vanillaBachelier = VanillaBachelierModel(S, K, T, r, sigma*S)
    price = vanillaBachelier.calculate_call_price()
    return price / K**2

def putintegrand_Bachelier(K):
    vanillaBachelier = VanillaBachelierModel(S, K, T, r, sigma*S)
    price = vanillaBachelier.calculate_put_price()
    return price / K**2

I_put = quad(putintegrand_Bachelier, 0, F)
I_call = quad(callintegrand_Bachelier, F, 5000)

E_var = 2*np.exp(r*T)*(I_put[0] + I_call[0])

print('The expected integrated variance is: %.9f' % E_var)

sigma = math.sqrt( E_var / T)
print('What sigma should we use: ', sigma)

def exotic_payoff_replication(S_T):
    return S_T**(1/3) + 1.5 * np.log(S_T) + 10.0

def exotic_option_price():
    # Let's consider a dynamic range for integration.
    # Setting the range based on multiple standard deviations around the forward price.
    S_min = F - 10 * sigma * np.sqrt(T)
    S_max = F + 10 * sigma * np.sqrt(T)
    
    integrand = lambda S_T: exotic_payoff_replication(S_T) * norm.pdf((S_T - F) / (sigma * np.sqrt(T)))
    price, _ = quad(integrand, S_min, S_max, epsabs=1e-8, epsrel=1e-8)
    
    return price * np.exp(-r * T)

print('The exotic option price using static replication is: %.2f' % exotic_option_price())
