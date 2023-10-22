import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pylab as plt

from analytical_option_formulae.option_types.option_models.displaced_diffusion_model import VanillaDisplacedDiffusionModel


from scipy.optimize import least_squares

def dd_calibration(x, strikes, market_vols, F, S, r, T):
    # sigma, beta = x
    sigma = x
    beta = 1.0 # work on assumption that beta is 1
    err = 0.0
    # print("check strikes: ", strikes)
    for i, K in enumerate(strikes):
        # print("check i: ", i)
        # print("check params : ", F, K, r, sigma, T, beta)
        dd_model = VanillaDisplacedDiffusionModel(F, K, r, sigma, T, beta)

        # Determine if it's a call or put based on moneyness
        if K > F:
            price = dd_model.calculate_call_price()
            payoff = 'call'
        else:
            price = dd_model.calculate_put_price()
            payoff = 'put'

        # print("price: ", price)
        # print("payoff: ", payoff)
        implied_vol = impliedVolatility(S, K, r, price, T, 
                                                  'call' if K > S else 'put')
        # print("implied_vol: ", implied_vol)
        # print("market_vols[i]: ", market_vols[i])
        err += (implied_vol - market_vols[i])**2
        # print("individual err: ", (implied_vol - market_vols[i])**2)
        # print("err: ", err)
    return err

def impliedVolatility_dd(F, K, r, S, sigma, T, beta):
    dd_model = VanillaDisplacedDiffusionModel(F, K, r, sigma, T, beta)

    # Determine if it's a call or put based on moneyness
    if K > F:
        price = dd_model.calculate_call_price()
        payoff = 'call'
    else:
        price = dd_model.calculate_put_price()
        payoff = 'put'

    print("price: ", price)
    print("payoff: ", payoff)
    print("check params: ", S, K, r, price, T, 'call' if K > S else 'put')
    implied_vol = impliedVolatility(S, K, r, price, T, 
                                                'call' if K > S else 'put')
    print("check implied_vol: ", implied_vol)
    return implied_vol

def impliedVolatility(S, K, r, price, T, payoff):
    try:
        if (payoff.lower() == 'call'):
            impliedVol = brentq(lambda x: price -
                                BlackScholesLognormalCall(S, K, r, x, T),
                                1e-12, 10.0)
        elif (payoff.lower() == 'put'):
            impliedVol = brentq(lambda x: price -
                                BlackScholesLognormalPut(S, K, r, x, T),
                                1e-12, 10.0)
        else:
            raise NameError('Payoff type not recognized')
    except Exception as e:
        # print(f"Error for Strike {K}, Payoff {payoff}: {e}")
        impliedVol = np.nan

    return impliedVol



def BlackScholesLognormalCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def BlackScholesLognormalPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


#####
# Here, load DataFrame with strike and implied volatility information into "df"
#####
df = pd.read_csv('SPX_options.csv')
df['mid'] = 0.5*(df['best_bid'] + df['best_offer'])
df['strike'] = df['strike_price']*0.001
df['payoff'] = df['cp_flag'].map(lambda x: 'call' if x == 'C' else 'put')
exdate = sorted(df['exdate'].unique())[0]
df = df[df['exdate'] == exdate]
days_to_expiry = (pd.Timestamp(str(exdate)) - pd.Timestamp('2020-12-01')).days
T = days_to_expiry/365
S = 3662.45
r = 0.14/100.0
F = S*np.exp(r*T)

df['vols'] = df.apply(lambda x: impliedVolatility(S,
                                                  x['strike'],
                                                  r,
                                                  x['mid'],
                                                  T,
                                                  x['payoff']),
                      axis=1)
df.dropna(inplace=True)
call_df = df[df['payoff'] == 'call']
put_df = df[df['payoff'] == 'put']
strikes = put_df['strike'].values
impliedvols = []
for K in strikes:    
    if K > S:
        impliedvols.append(call_df[call_df['strike'] == K]['vols'].values[0])
    else:
        impliedvols.append(put_df[put_df['strike'] == K]['vols'].values[0])

# populate "df" with the dataframe containing strikes and market implied volatilities
df = pd.DataFrame({'strike': strikes, 'impliedvol': impliedvols})

# initialGuess = [0.5, 0.5]
# res = least_squares(lambda x: dd_calibration(x, df['strike'], df['impliedvol'], F, S, 0.001255, 0.046575), initialGuess,bounds=([0.0,0.0],[2.0,2.0]))
# beta, sigma = res.x

# what if we work on premise that beta is 1 and try to find sigma?
initialGuess = [0.5]
res = least_squares(lambda x: dd_calibration(x, df['strike'], df['impliedvol'], F, S, 0.001255, 0.046575), initialGuess,bounds=(0.0, 2.0))
# beta, sigma = res.x
sigma = res.x[0]
beta = 1.0
print("beta: ", beta)
print("sigma: ", sigma)

displaced_vols = [impliedVolatility_dd(F, K, r, S, sigma, T, beta) for K in strikes]
# print('Calibrated SABR model parameters: alpha = %.3f, beta = %.1f, rho = %.3f, nu = %.3f' % (alpha, beta, rho, nu))
print("check strikes: ", strikes)
print("check displaced_vols: ", displaced_vols)
# sabrvols = []
# for K in strikes:
#     sabrvols.append(SABR(F, K, T, alpha, beta, rho, nu))

plt.figure(tight_layout=True)
plt.plot(strikes, df['impliedvol'], 'gs', label='Market Vols')
plt.plot(strikes, displaced_vols, 'm--', label='DD Vols')
plt.legend()
plt.show()