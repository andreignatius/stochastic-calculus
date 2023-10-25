import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

def simulate_paths(S0, sigma, r, T, N, n_paths):
    dt = T/N
    dW = np.random.randn(N, n_paths) * np.sqrt(dt)  # Array of Wiener increments
    paths = np.zeros((N+1, n_paths))
    paths[0] = S0
    for t in range(1, N+1):
        paths[t] = paths[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*dW[t-1])
    return paths

def compute_phi(paths, K, r, sigma, t, T):
    dt = T/paths.shape[0]
    d1 = (np.log(paths[t]/K) + (r + 0.5*sigma**2)*(T-t*dt)) / (sigma*np.sqrt(T-t*dt))
    return norm.cdf(d1)

def compute_psi(paths, K, r, sigma, t, T):
    dt = T/paths.shape[0]
    d1 = (np.log(paths[t]/K) + (r + 0.5*sigma**2)*(T-t*dt)) / (sigma*np.sqrt(T-t*dt))
    d2 = d1 - sigma*np.sqrt(T-t*dt)
    return -K * np.exp(-r*(T-t*dt)) * norm.cdf(d2)

# Record the start time
start_time = time.time()

S_0 = 100
K = 100
sigma = 0.2
r = 0.05
T = 1/12

# Parameters
num_paths = 50000
hedging_intervals = [21, 84]  
errors = {N: [] for N in hedging_intervals}  

def bs_call_price(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

for N in hedging_intervals:
    dt = T/N  
    
    paths = simulate_paths(S_0, sigma, r, T, N, num_paths)
    
    portfolio_values = bs_call_price(S_0, K, r, sigma, T)  # Initial portfolio value when selling the call
    delta_initial = compute_phi(paths, K, r, sigma, 0, T)
    
    # Initial positions in stock and bond
    stock_position = delta_initial * paths[0]
    bond_position = portfolio_values - stock_position  # Everything not invested in stock is invested in bond
    
    for t in range(1, N):  
        delta_now = compute_phi(paths, K, r, sigma, t, T)
        
        # Calculate change in stock position based on new delta
        change_in_stock = (delta_now - delta_initial) * paths[t]
        
        # Update bond position: account for interest and change in stock position
        bond_position = bond_position * np.exp(r*dt) - change_in_stock
        
        # Update stock position
        stock_position = delta_now * paths[t]
        
        delta_initial = delta_now
    
    option_payoffs = np.maximum(paths[-1] - K, 0)
    
    # Total portfolio value = stock position + bond position
    portfolio_values = stock_position + bond_position
    hedging_errors = portfolio_values - option_payoffs
    errors[N] = hedging_errors
    
    plt.hist(hedging_errors, bins=50, alpha=0.5, label=f'N={N}')
    plt.legend()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time} seconds")
plt.xlabel(r'Hedging Error: ( Portfolio Value - Option Payoff ) (\$)')
plt.ylabel(r'Frequency Count')
plt.title(r'Frequency Distribution of Hedging Errors for Different Hedging Intervals')
plt.show()
