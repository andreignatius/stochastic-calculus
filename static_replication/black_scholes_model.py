import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import math

from analytical_option_formulae.option_types.option_models.black_scholes_model import VanillaBlackScholesModel

# Start Date: 1-Dec-2020
# End Date: 15-Jan-2021
# The time difference is 45 days (considering 31 days in December and 14 days in January).

# In a year with 365 days, the time to maturity T is: 45 / 365

# Example:
S = 100  # Stock price today
K = 110  # Strike price
# T = 1  # Time until expiry (in years)
T = 45/365  # Time until expiry (in years)
r = 0.05  # Risk-free rate
sigma = 0.4  # Volatility
F = S * np.exp(r*T)

vanillaBSM = VanillaBlackScholesModel(S, K, r, sigma, T)
print("BSM call price: ", vanillaBSM.calculate_call_price())
print("BSM put price: ", vanillaBSM.calculate_put_price())

def callintegrand_BSM(K):
    vanillaBSM = VanillaBlackScholesModel(S, K, r, sigma, T)
    price = vanillaBSM.calculate_call_price()
    return price / K**2

def putintegrand_BSM(K):
    vanillaBSM = VanillaBlackScholesModel(S, K, r, sigma, T)
    price = vanillaBSM.calculate_put_price()
    return price / K**2

I_put = quad(lambda x: putintegrand_BSM(x), 0.0, F)
I_call = quad(lambda x: callintegrand_BSM(x), F, 5000)
E_var = 2*np.exp(r*T) * (I_put[0] + I_call[0])


print('The expected integrated variance is 000: %.9f' % E_var)

def callintegrand_BSM(K):
    vanillaBSM = VanillaBlackScholesModel(S, K, r, sigma, T)
    price = vanillaBSM.calculate_call_price()
    return price / K**2

def putintegrand_BSM(K):
    vanillaBSM = VanillaBlackScholesModel(S, K, r, sigma, T)
    price = vanillaBSM.calculate_put_price()
    return price / K**2



I_put = quad(lambda x: putintegrand_BSM(x), 0.0, F)
I_call = quad(lambda x: callintegrand_BSM(x), F, 5000)
E_var = 2*np.exp(r*T)*(I_put[0] + I_call[0])
print('The expected integrated variance is 111: %.9f' % E_var)

vanillaBSM = VanillaBlackScholesModel(S, K, r, sigma, T)
print("BSM call price: ", vanillaBSM.calculate_call_price())
print("BSM put price: ", vanillaBSM.calculate_put_price())

sigma = math.sqrt( E_var / T)
print('What sigma should we use: ', sigma)

def exotic_payoff_replication(S_T):
    return S_T**(1/3) + 1.5 * np.log(S_T) + 10.0

def exotic_option_price():
    # Define the Black-Scholes density function
    def BS_density(S_T):
        term1 = np.log(S_T/S)
        term2 = (r - 0.5 * sigma**2) * T
        numerator = np.exp(- (term1 - term2)**2 / (2 * sigma**2 * T))
        denominator = S_T * sigma * np.sqrt(2 * np.pi * T)
        return numerator / denominator

    # We consider a dynamic range for integration.
    # Setting the range based on multiple standard deviations around the forward price.
    F = S * np.exp(r * T)
    S_min = S * np.exp((r - sigma**2 / 2) * T - 6 * sigma * np.sqrt(T))
    S_max = S * np.exp((r - sigma**2 / 2) * T + 6 * sigma * np.sqrt(T))
    
    integrand = lambda S_T: exotic_payoff_replication(S_T) * BS_density(S_T)
    price, _ = quad(integrand, S_min, S_max, epsabs=1e-8, epsrel=1e-8)
    
    return price * np.exp(-r * T)

print('The exotic option price using static replication is: %.2f' % exotic_option_price())


# Prof Tee method for comparison

def BlackScholesCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def BlackScholesPut(S, K, r, sigma, T):
    return BlackScholesCall(S, K, r, sigma, T) - S + K*np.exp(-r*T)


def callintegrand(K, S, r, T, sigma):
    # print("check K0: ", K)
    price = BlackScholesCall(S, K, r, sigma, T) / K**2
    return price


def putintegrand(K, S, r, T, sigma):
    # print("check K1: ", K)
    price = BlackScholesPut(S, K, r, sigma, T) / K**2
    return price


S = 100.0
K = 110.0
r = 0.05
# T = 1.0
T = 45/365  # Time until expiry (in years)
sigma = 0.4
F = S * np.exp(r*T)
I_put = quad(lambda x: putintegrand(x, S, r, T, sigma), 0.0, F)
I_call = quad(lambda x: callintegrand(x, S, r, T, sigma), F, 5000)
E_var = 2*np.exp(r*T)*(I_put[0] + I_call[0])
print("BSM call price (Prof Tee): ", BlackScholesCall(S, K, r, sigma, T))
print("BSM put price (Prof Tee): ", BlackScholesPut(S, K, r, sigma, T))
print('The expected integrated variance (Prof Tee) is: %.9f' % E_var)
