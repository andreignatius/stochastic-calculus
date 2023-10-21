import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

def simulate_paths(S0, sigma, r, T, N, n_paths):
    dt = T/N
    paths = np.zeros((N+1, n_paths))
    paths[0] = S0
    for t in range(1, N+1):
        Z = np.random.randn(n_paths)
        paths[t] = paths[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return paths

def compute_phi(St, K, r, sigma, t, T):
    if t >= T:
        return 1.0 if St > K else 0.0
    d1 = (np.log(St/K) + (r + 0.5*sigma**2)*(T-t)) / (sigma*np.sqrt(T-t))
    return norm.cdf(d1)

def compute_psi(St, K, r, sigma, t, T):
    if t >= T:
        return 1.0 if St <= K else 0.0
    
    d1 = (np.log(St/K) + (r + 0.5*sigma**2)*(T-t)) / (sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    return -K * np.exp(-r*T) * norm.cdf(d2)

# Record the start time
start_time = time.time()

S_0 = 100
K = 100
sigma = 0.2
r = 0.05
T = 1/12

# Parameters
num_paths = 50000
hedging_intervals = [21, 84]  # N values
errors = {N: [] for N in hedging_intervals}  # Initialize errors for each N value

for N in hedging_intervals:
    dt = T/N  # time step
    
    for path in range(num_paths):
        if path % 1000 == 0 and path > 0:
            print("paths explored: ", path)
        # Initialization
        S = S_0
        B = 1.0  # Starting value for risk-free bond
        portfolio_value = 0.0  # Reset the portfolio value for each path
        
        for t in np.linspace(0, T, N+1)[:-1]:  # Excluding the final time
            phi = compute_phi(S, K, r, sigma, t, T)
            psi = compute_psi(S, K, r, sigma, t, T)
            
            # Update the portfolio value for this step based on phi and psi
            portfolio_value = phi * S - psi * B
            
            # Simulate the next stock price
            dS = r * S * dt + sigma * S * np.random.normal(0, np.sqrt(dt))
            S += dS
            
            # Update bond value (assuming continuous compounding)
            B = B * np.exp(r * dt)

        # At the end of the path (maturity), compute the option payoff
        option_payoff = max(S - K, 0)
        
        # Compute the hedging error for this path
        hedging_error = portfolio_value - option_payoff
        
        # Store the hedging error
        errors[N].append(hedging_error)
    
    # Now, within the loop, plot the histogram for this N value
    plt.hist(errors[N], bins=50, alpha=0.5, label=f'N={N}')
    plt.legend()

print("READY!")
# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time} seconds")
plt.show()


