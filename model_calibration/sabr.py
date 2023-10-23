import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pylab as plt


def SABR(F, K, T, alpha, beta, rho, nu):
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


from scipy.optimize import least_squares

beta = 0.7


def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T, x[0], beta, x[1], x[2])) ** 2

    return err


def impliedVolatility(S, K, r, price, T, payoff):
    try:
        if payoff.lower() == "call":
            impliedVol = brentq(
                lambda x: price - BlackScholesLognormalCall(S, K, r, x, T), 1e-12, 10.0
            )
        elif payoff.lower() == "put":
            impliedVol = brentq(
                lambda x: price - BlackScholesLognormalPut(S, K, r, x, T), 1e-12, 10.0
            )
        else:
            raise NameError("Payoff type not recognized")
    except Exception as e:
        print(f"Error for Strike {K}, Payoff {payoff}: {e}")
        impliedVol = np.nan

    return impliedVol


def BlackScholesLognormalCall(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def BlackScholesLognormalPut(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


#####
# Here, load DataFrame with strike and implied volatility information into "df"
#####
df = pd.read_csv("../data/SPX_options.csv")
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

df["vols"] = df.apply(
    lambda x: impliedVolatility(S, x["strike"], r, x["mid"], T, x["payoff"]), axis=1
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

initialGuess = [0.02, 0.2, 0.1]
res = least_squares(
    lambda x: sabrcalibration(x, df["strike"], df["impliedvol"], F, T), initialGuess
)
alpha = res.x[0]
rho = res.x[1]
nu = res.x[2]

print(
    "Calibrated SABR model parameters: alpha = %.3f, beta = %.1f, rho = %.3f, nu = %.3f"
    % (alpha, beta, rho, nu)
)

sabrvols = []
for K in strikes:
    sabrvols.append(SABR(F, K, T, alpha, beta, rho, nu))

plt.figure(tight_layout=True)
plt.plot(strikes, df["impliedvol"], "gs", label="Market Vols")
plt.plot(strikes, sabrvols, "m--", label="SABR Vols")
plt.legend()
plt.show()
