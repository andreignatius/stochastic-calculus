import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, least_squares, fsolve
import matplotlib.pylab as plt

from analytical_option_formulae.option_types.option_models.black_scholes_model import (
    VanillaBlackScholesModel,
)
from analytical_option_formulae.option_types.option_models.displaced_diffusion_model import (
    VanillaDisplacedDiffusionModel,
)


# Black-Scholes with Displaced Diffusion
def BlackScholesLognormalCall_displaced(S, K, r, sigma, T, beta):
    S_displaced = beta * S + (1 - beta) * K
    d1 = (np.log(S_displaced / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S_displaced * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def BlackScholesLognormalPut_displaced(S, K, r, sigma, T, beta):
    S_displaced = beta * S + (1 - beta) * K
    d1 = (np.log(S_displaced / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S_displaced * norm.cdf(-d1)


def impliedVolatility_displaced(S, K, r, price, T, payoff, beta):
    try:
        if payoff.lower() == "call":
            impliedVol = brentq(
                lambda x: price
                - BlackScholesLognormalCall_displaced(S, K, r, x, T, beta),
                1e-12,
                10.0,
            )
        elif payoff.lower() == "put":
            impliedVol = brentq(
                lambda x: price
                - BlackScholesLognormalPut_displaced(S, K, r, x, T, beta),
                1e-12,
                10.0,
            )
        else:
            raise NameError("Payoff type not recognized")
    except Exception as e:
        print(f"Error for Strike {K}, Payoff {payoff}: {e}")
        impliedVol = np.nan

    return impliedVol


def BlackScholes(F, K, r, vol, T, payoff):
    """
    Wrapper around VanillaBlackScholesModel to return call or put price.
    """
    bs_model = VanillaBlackScholesModel(F, K, r, vol, T)
    if payoff == "call":
        return bs_model.calculate_call_price()
    else:
        return bs_model.calculate_put_price()


def DDModel(F, K, r, sigma, T, beta):
    """
    Displaced Diffusion Model to compute implied volatility for each strike.
    """
    # Create an instance of VanillaDisplacedDiffusionModel
    dd_model = VanillaDisplacedDiffusionModel(F, K, r, sigma, T, beta)

    # Determine if it's a call or put based on moneyness
    if K > F:
        price = dd_model.calculate_call_price()
        payoff = "call"
    else:
        price = dd_model.calculate_put_price()
        payoff = "put"

    # Define error function for root-finding
    def error_function(vol):
        bs_price = BlackScholes(F, K, r, vol, T, payoff)
        return bs_price - price

    try:
        # Inverse calculation using brentq
        implied_vol = brentq(error_function, 1e-12, 10.0)
    except Exception as e:
        print(f"Error for Strike {K}, Payoff {payoff}: {e}")
        implied_vol = np.nan

    return implied_vol

    # def error_function(vol):
    #     bs_price = BlackScholes(F, K, r, vol, T, payoff)
    #     return bs_price - price

    # try:
    #     # Inverse calculation using fsolve
    #     implied_vol = fsolve(error_function, 0.2)  # starting guess of 0.2
    # except Exception as e:
    #     print(f"Error for Strike {K}, Payoff {payoff}: {e}")
    #     implied_vol = np.nan

    # return implied_vol


# def displacement_calibration(x, strikes, market_prices, S, r, T):
#     beta, sigma = x
#     err = 0.0
#     for i, K in enumerate(strikes):
#         if K > S:
#             model_price = BlackScholesLognormalCall_displaced(S, K, r, sigma, T, beta)
#         else:
#             model_price = BlackScholesLognormalPut_displaced(S, K, r, sigma, T, beta)

#         err += (model_price - market_prices[i])**2
#     return err


def displacement_calibration(x, S, strikes, rate, sigma, T, market_vols):
    err = 0.0
    adjusted_summary_list = list(zip(strikes, market_vols))
    for strike, vol in adjusted_summary_list:
        if strike > S:
            dd_price_call = displaced_diffusion_call(F, strike, rate, sigma, T, x[0])
            imp_vol = implied_volatility(S, strike, rate, dd_price_call, T, "call")
        else:
            dd_price_put = displaced_diffusion_put(F, strike, rate, sigma, T, x[0])
            imp_vol = implied_volatility(S, strike, rate, dd_price_put, T, "put")
        err += (vol - imp_vol) ** 2
    return err


#####
# Here, load DataFrame with strike and implied volatility information into "df"
#####

# Set a constant value for beta
beta_constant = 0.8  # you can adjust this value as needed

df = pd.read_csv("../data/SPX_options.csv")
print("check df0: ", df)
df["mid"] = 0.5 * (df["best_bid"] + df["best_offer"])
df["strike"] = df["strike_price"] * 0.001
df["payoff"] = df["cp_flag"].map(lambda x: "call" if x == "C" else "put")
exdate = sorted(df["exdate"].unique())[0]
df = df[df["exdate"] == exdate]
days_to_expiry = (pd.Timestamp(str(exdate)) - pd.Timestamp("2020-12-01")).days
T = days_to_expiry / 365
S = 3662.45
r = 0.14 / 100.0
F = S * np.exp(r * T)

# Adjusting the lambda function to pass beta
df["vols"] = df.apply(
    lambda x: impliedVolatility_displaced(
        S, x["strike"], r, x["mid"], T, x["payoff"], beta_constant
    ),  # passing beta here
    axis=1,
)

df.dropna(inplace=True)
call_df = df[df["payoff"] == "call"]
put_df = df[df["payoff"] == "put"]
strikes = put_df["strike"].values
impliedvols = []
for K in strikes:
    if K > S:
        impliedvols.append(call_df[call_df["strike"] == K]["vols"].values[0])
    else:
        impliedvols.append(put_df[put_df["strike"] == K]["vols"].values[0])

# populate "df" with the dataframe containing strikes and market implied volatilities
df = pd.DataFrame({"strike": strikes, "impliedvol": impliedvols})

print("check df: ", df)

# Displacement Diffusion Calibration
# initialGuess_displaced = [0.7, 0.2]
initialGuess_displaced = [0.1, 0.1]
res_displaced = least_squares(
    lambda x: displacement_calibration(x, df["strike"], df["impliedvol"], S, r, T),
    initialGuess_displaced,
)
beta, sigma = res_displaced.x


print("beta: ", beta)
print("sigma: ", sigma)

# Computing the implied volatilities with the displaced diffusion model
displaced_prices = [
    BlackScholesLognormalCall_displaced(S, K, r, sigma, T, beta)
    if K > S
    else BlackScholesLognormalPut_displaced(S, K, r, sigma, T, beta)
    for K in strikes
]
# displaced_vols = [impliedVolatility_displaced(S, K, r, price, T, 'call' if K > S else 'put', beta) for K, price in zip(strikes, displaced_prices)]

# Computing implied volatilities for each strike
displaced_vols = [DDModel(F, K, r, sigma, T, beta) for K in strikes]


# print("displaced_vols: ", displaced_vols)

plt.figure(tight_layout=True)
plt.plot(strikes, df["impliedvol"], "gs", label="Market Vols")
plt.plot(strikes, displaced_vols, "b-.", label="Displaced Diffusion Vols")
plt.legend()
plt.show()
