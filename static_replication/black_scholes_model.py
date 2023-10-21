import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

from analytical_option_formulae.option_types.option_models.black_scholes_model import VanillaBlackScholesModel

# Example:
S = 100  # Stock price today
K = 110  # Strike price
T = 1  # Time until expiry (in years)
r = 0.05  # Risk-free rate
sigma = 0.4  # Volatility

vanillaBSM = VanillaBlackScholesModel(S, K, r, sigma, T)
print("BSM call price: ", vanillaBSM.calculate_call_price())
print("BSM put price: ", vanillaBSM.calculate_put_price())

F = S + np.exp(r*T)

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

# S = 100.0
# r = 0.05
# T = 1.0
# sigma = 0.4
F = S * np.exp(r*T)
I_put = quad(lambda x: putintegrand_BSM(x), 0.0, F)
I_call = quad(lambda x: callintegrand_BSM(x), F, 5000)
E_var = 2*np.exp(r*T)*(I_put[0] + I_call[0])
print('The expected integrated variance is 111: %.9f' % E_var)

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
T = 1.0
sigma = 0.4
F = S * np.exp(r*T)
I_put = quad(lambda x: putintegrand(x, S, r, T, sigma), 0.0, F)
I_call = quad(lambda x: callintegrand(x, S, r, T, sigma), F, 5000)
E_var = 2*np.exp(r*T)*(I_put[0] + I_call[0])
print("BSM call price (Prof Tee): ", BlackScholesCall(S, K, r, sigma, T))
print("BSM put price (Prof Tee): ", BlackScholesPut(S, K, r, sigma, T))
print('The expected integrated variance (Prof Tee) is: %.9f' % E_var)
