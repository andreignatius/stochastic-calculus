import datetime as dt
import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import sys

sys.path.append("..")

from analytical_option_formulae.option_types.vanilla_option import VanillaOption

# Common Functions

vanilla_option = VanillaOption()


def implied_volatility(
    S: float, K: float, r: float, price: float, T: float, options_type: str
) -> float:
    try:
        bs_model = lambda x: vanilla_option.black_scholes_model(S, K, r, x, T)
        if options_type.lower() == "call":
            implied_vol = brentq(
                lambda x: price - bs_model(x).calculate_call_price(), 1e-12, 10.0
            )
        elif options_type.lower() == "put":
            implied_vol = brentq(
                lambda x: price - bs_model(x).calculate_put_price(), 1e-12, 10.0
            )
        else:
            raise NameError("Payoff type not recognized")
    except Exception:
        implied_vol = np.nan

    return implied_vol


def payoff_function(S: float) -> float:
    return S ** (1 / 3) + 1.5 * np.log(S) + 10


def payoff_function_second_derivative(S: float) -> float:
    return -2 / (9 * S**5 / 3) - 1.5 / S**2


# BS Model


def black_scholes_pricing(S: float, r: float, sigma: float, T: float) -> float:
    discount_factor = np.exp(-r * T)
    first_term = (S * np.exp((r - 1 / 3 * sigma**2) * T)) ** (1 / 3)
    second_term = 1.5 * (np.log(S) + (r - 0.5 * sigma**2) * T)
    third_term = 10
    return discount_factor * (first_term + second_term + third_term)


# Bachelier Model


def bachelier_pricing(S: float, r: float, sigma: float, T: float) -> float:
    discount_factor = np.exp(-r * T)

    def first_term_func(x):
        return (
            1
            / np.sqrt(2 * np.pi)
            * (S + S * sigma * np.sqrt(T) * x) ** (1 / 3)
            * np.exp(-0.5 * x**2)
        )

    first_term_lower_bound = -1 / (sigma * np.sqrt(T))
    first_term = quad(first_term_func, first_term_lower_bound, float("inf"))[0]

    def second_term_func(x):
        return (
            1.5
            / np.sqrt(2 * np.pi)
            * np.log(S + S * sigma * np.sqrt(T) * x)
            * np.exp(-0.5 * x**2)
        )

    second_term_lower_bound = -1 / (sigma * np.sqrt(T))
    second_term = quad(second_term_func, second_term_lower_bound, float("inf"))[0]
    third_term = 10

    return discount_factor * (first_term + second_term + third_term)


# SABR Model


def SABR(
    F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float
) -> float:
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta) ** 2) / 24) * alpha * alpha / (F ** (2 - 2 * beta))
        numer2 = 0.25 * rho * beta * nu * alpha / (F ** (1 - beta))
        numer3 = ((2 - 3 * rho * rho) / 24) * nu * nu
        VolAtm = alpha * (1 + (numer1 + numer2 + numer3) * T) / (F ** (1 - beta))
        sabrsigma = VolAtm
    else:
        z = (nu / alpha) * ((F * X) ** (0.5 * (1 - beta))) * np.log(F / X)
        zhi = np.log((((1 - 2 * rho * z + z * z) ** 0.5) + z - rho) / (1 - rho))
        numer1 = (((1 - beta) ** 2) / 24) * ((alpha * alpha) / ((F * X) ** (1 - beta)))
        numer2 = 0.25 * rho * beta * nu * alpha / ((F * X) ** ((1 - beta) / 2))
        numer3 = ((2 - 3 * rho * rho) / 24) * nu * nu
        numer = alpha * (1 + (numer1 + numer2 + numer3) * T) * z
        denom1 = ((1 - beta) ** 2 / 24) * (np.log(F / X)) ** 2
        denom2 = (((1 - beta) ** 4) / 1920) * ((np.log(F / X)) ** 4)
        denom = ((F * X) ** ((1 - beta) / 2)) * (1 + denom1 + denom2) * zhi
        sabrsigma = numer / denom

    return sabrsigma


def SABRCall(
    S: float,
    K: float,
    r: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    T: float,
) -> float:
    sabr_vol = SABR(S * np.exp(r * T), K, T, alpha, beta, rho, nu)
    return vanilla_option.black_scholes_model(
        S, K, r, sabr_vol, T
    ).calculate_call_price()


def SABRPut(
    S: float,
    K: float,
    r: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    T: float,
) -> float:
    sabr_vol = SABR(S * np.exp(r * T), K, T, alpha, beta, rho, nu)
    return vanilla_option.black_scholes_model(
        S, K, r, sabr_vol, T
    ).calculate_put_price()


def sabrcallintegrand(
    K: float,
    S: float,
    r: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    price = SABRCall(
        S, K, r, alpha, beta, rho, nu, T
    ) * payoff_function_second_derivative(K)
    return price


def sabrputintegrand(
    K: float,
    S: float,
    r: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    price = SABRPut(
        S, K, r, alpha, beta, rho, nu, T
    ) * payoff_function_second_derivative(K)
    return price


def static_replication_sabr(
    r: float,
    S: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    F = S * np.exp(r * T)
    first_term = np.exp(-r * T) * payoff_function(F)
    second_term = quad(
        lambda x: sabrputintegrand(x, S, r, T, alpha, beta, rho, nu), 1e-6, F
    )[0]
    third_term = quad(
        lambda x: sabrcallintegrand(x, S, r, T, alpha, beta, rho, nu), F, float("inf")
    )[0]
    return first_term + second_term + third_term


# Data Importing & Processing
df_spx = pd.read_csv("../data/SPX_options.csv")
df_spx = df_spx[df_spx["exdate"] == 20210115]
df_spx["mid"] = 0.5 * (df_spx["best_bid"] + df_spx["best_offer"])
df_spx["strike_price"] = df_spx["strike_price"] / 1000

df_spy = pd.read_csv("../data/SPY_options.csv")
df_spy = df_spy[df_spy["exdate"] == 20210115]
df_spy["mid"] = 0.5 * (df_spy["best_bid"] + df_spy["best_offer"])
df_spy["strike_price"] = df_spy["strike_price"] / 1000

df_rates = pd.read_csv("../data/zero_rates_20201201.csv")
df_rates["rate_decimal"] = df_rates["rate"] / 100
df_rates = df_rates.drop(["date"], axis=1)
df_rates.set_index("days", inplace=True)
df_rates = df_rates.reindex(np.arange(df_rates.index.min(), df_rates.index.max() + 1))
df_rates = df_rates.interpolate(method="linear")

# Data Input
S_spx = 3662.45
S_spy = 366.02
start_date = dt.date(2020, 12, 1)
expiry_date = dt.date(2021, 1, 15)
days_to_expiry = (expiry_date - start_date).days
T = days_to_expiry / 365
r = df_rates.loc[days_to_expiry]["rate_decimal"]
F_spx = S_spx * np.exp(r * T)
F_spy = S_spy * np.exp(r * T)
K_spx = min(df_spx["strike_price"], key=lambda x: abs(x - F_spx))
K_spy = min(df_spy["strike_price"], key=lambda x: abs(x - F_spy))

# From part 2
sigma_spx = 0.1849096526276905  # ATM sigma
sigma_spy = 0.1972176434869465  # ATM sigma
spx_sabr = {
    "alpha": 1.8165044370781172,
    "beta": 0.7,
    "rho": -0.4043017672449347,
    "nu": 2.790158312103804,
}
spy_sabr = {
    "alpha": 0.9081326337814014,
    "beta": 0.7,
    "rho": -0.4887794457550238,
    "nu": 2.7285163417661487,
}

# Results
## SPX
spx_results = {
    "sigma SPX": sigma_spx,
    "BS price SPX": black_scholes_pricing(S_spx, r, sigma_spx, T),
    "Bachelier price SPX": bachelier_pricing(S_spx, r, sigma_spx, T),
    "Static replication SABR SPX": static_replication_sabr(
        r,
        S_spx,
        K_spx,
        T,
        spx_sabr["alpha"],
        spx_sabr["beta"],
        spx_sabr["rho"],
        spx_sabr["nu"],
    ),
}
print(spx_results)

## SPY
spy_results = {
    "sigma SPY": sigma_spy,
    "BS price SPY": black_scholes_pricing(S_spy, r, sigma_spy, T),
    "Bachelier price SPY": bachelier_pricing(S_spy, r, sigma_spy, T),
    "Static replication SABR SPY": static_replication_sabr(
        r,
        S_spy,
        K_spy,
        T,
        spy_sabr["alpha"],
        spy_sabr["beta"],
        spy_sabr["rho"],
        spy_sabr["nu"],
    ),
}
print(spy_results)
