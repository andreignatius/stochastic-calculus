import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import sys
import time

sys.path.append("..")
from analytical_option_formulae.option_types.vanilla_option import VanillaOption


# Define common functions and models
vanilla_option = VanillaOption()


def calculate_stock_price(
    S: float, r: float, sigma: float, t: float, W: float
) -> float:
    return S * np.exp((r * t - 0.5 * sigma**2 * t) + sigma * W)


def calculate_phi(S_t: float, K: float, r: float, sigma: float, T: float, t: float):
    d1 = (np.log(S_t / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    return norm.cdf(d1)


def compute_psi(paths, K, r, sigma, t, T):
    dt = T / paths.shape[0]
    d1 = (np.log(paths[t] / K) + (r + 0.5 * sigma**2) * (T - t * dt)) / (
        sigma * np.sqrt(T - t * dt)
    )
    d2 = d1 - sigma * np.sqrt(T - t * dt)
    return -K * np.exp(-r * T) * norm.cdf(d2)


def simulate_paths(S0, sigma, r, T, N, n_paths):
    dt = T / N
    dW = np.random.randn(N, n_paths) * np.sqrt(dt)  # Array of Wiener increments
    paths = np.zeros((N + 1, n_paths))
    paths[0] = S0
    for t in range(1, N + 1):
        paths[t] = calculate_stock_price(paths[t - 1], r, sigma, dt, dW[t - 1])
    return paths


# Record the start time
start_time = time.time()

S_0 = 100
K = 100
sigma = 0.2
r = 0.05
T = 1 / 12

# Parameters
num_paths = 50000
hedging_intervals = [21, 84]
errors = {N: [] for N in hedging_intervals}


black_scholes_model = vanilla_option.black_scholes_model(S_0, K, r, sigma, T)
black_scholes_call_price = black_scholes_model.calculate_call_price()


for N in hedging_intervals:
    dt = T / N

    paths = simulate_paths(S_0, sigma, r, T, N, num_paths)
    B = np.exp(r * np.linspace(0, T, N + 1))  # Bond value at each time step

    portfolio_values = (
        black_scholes_call_price  # Initial portfolio value when selling the call
    )

    cash = (
        portfolio_values
        - calculate_phi(paths[0], K, r, sigma, T, 0 * T / paths.shape[0]) * paths[0]
        + compute_psi(paths, K, r, sigma, 0, T)
    )  # Adjusted cash position

    for t in range(1, N):  # Start from 1 as we've already initialized at t=0
        delta_prev = calculate_phi(
            paths[t], K, r, sigma, T, (t - 1) * T / paths.shape[0]
        )
        delta_now = calculate_phi(paths[t], K, r, sigma, T, t * T / paths.shape[0])

        # Adjust portfolio for change in stock and bond positions
        cash -= (delta_now - delta_prev) * paths[t]
        cash *= np.exp(r * dt)  # Account for risk-free interest on cash

        portfolio_values = cash + delta_now * paths[t]  # Update portfolio value

    option_payoffs = np.maximum(paths[-1] - K, 0)

    # Total portfolio value = stock position + bond position
    portfolio_values = stock_position + bond_position
    hedging_errors = portfolio_values - option_payoffs
    errors[N] = hedging_errors

    plt.hist(hedging_errors, bins=50, alpha=0.5, label=f"N={N}")
    plt.legend()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time} seconds")
plt.xlabel(r"Hedging Error: ( Portfolio Value - Option Payoff ) (\$)")
plt.ylabel(r"Frequency Count")
plt.title(r"Frequency Distribution of Hedging Errors for Different Hedging Intervals")
plt.show()
