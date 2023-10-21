import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

def compute_phi_vectorized(S, K, r, sigma, t, T):
    # For vectorized conditions
    phi_values = np.zeros_like(S)
    
    # Mask for conditions
    mask = t >= T
    phi_values[mask] = (S[mask] > K).astype(float)
    
    # For non-masked values
    d1 = (np.log(S[~mask]/K) + (r + 0.5*sigma**2)*(T-t)) / (sigma*np.sqrt(T-t))
    phi_values[~mask] = norm.cdf(d1)
    
    return phi_values

def compute_psi_vectorized(S, K, r, sigma, t, T):
    psi_values = np.zeros_like(S)
    
    # Mask for conditions
    mask = t >= T
    psi_values[mask] = (S[mask] <= K).astype(float)
    
    # For non-masked values
    d1 = (np.log(S[~mask]/K) + (r + 0.5*sigma**2)*(T-t)) / (sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    psi_values[~mask] = -K * np.exp(-r*T) * norm.cdf(d2)
    
    return psi_values

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

for N in hedging_intervals:
    dt = T/N
    
    # Initialization
    S = np.full(num_paths, S_0, dtype=np.float64) # Create an array filled with S_0
    B = np.ones(num_paths) 
    portfolio_values = np.zeros(num_paths)
    
    Z = np.random.randn(N, num_paths)  # Generate all random numbers at once

    for t in np.linspace(0, T, N):  # We can include the final time now
        phi = compute_phi_vectorized(S, K, r, sigma, t, T)
        psi = compute_psi_vectorized(S, K, r, sigma, t, T)

        portfolio_values = phi * S - psi * B
        
        # Update stock and bond values for all paths simultaneously
        dS = r * S * dt + sigma * S * Z[int(t*N)]
        S += dS
        S = np.maximum(S, 1e-8)  # Ensure stock prices don't go too close to 0
        
        B *= np.exp(r * dt)

    # Compute option payoffs and hedging errors for all paths simultaneously
    option_payoffs = np.maximum(S - K, 0)
    hedging_errors = portfolio_values - option_payoffs

    # Store the hedging errors
    errors[N] = hedging_errors.tolist()

    # Plot the histogram for this N value
    plt.hist(errors[N], bins=50, alpha=0.5, label=f'N={N}')
    plt.legend()

print("READY!")
# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time} seconds")
plt.show()
