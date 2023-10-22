# stochastic-calculus

Part 1 : Analytical Option Formulae

Consider the following European options:<br>
• Vanilla call/put<br>
• Digital cash-or-nothing call/put<br>
• Digital asset-or-nothing call/put<br>

Derive and implement the following models to value these options in Python:<br>
1 Black-Scholes model<br>
\begin{align*}
1. & \text{Black-Scholes Model:} \\
i) & S_T = S_0 \exp \left[ (r - \frac{\sigma^2}{2})T + \sigma W_T \right], \quad W_T \sim N(0, T) \\
& = S_0 \exp \left[ (r - \frac{\sigma^2}{2})T + \sigma X \right], \quad X \sim N(0, 1) \\
2) & \text{Vanilla call:} \\
V_c & = e^{-rT} E \left[ (S_T - K)^+ \right], \quad S_T > K \Rightarrow X = \frac{\log \left( \frac{S_0}{K} \right) + (r - \frac{\sigma^2}{2})T}{\sigma \sqrt{T}} = X^* \\
& = \frac{e^{-rT}}{\sqrt{2\pi}} \int_{X^*}^{\infty} (S_0 \exp \left[ (r - \frac{\sigma^2}{2})T + \sigma X \right] - K) dX \\
& = \frac{S_0}{\sqrt{2\pi}} \int_{X^*}^{\infty} e^{- \frac{(X - \frac{\sigma^2 T}{2})^2}{2}} dX - e^{-rT} \frac{K}{\sqrt{2\pi}} \int_{X^*}^{\infty} e^{- \frac{X^2}{2}} dX \\
& = S_0 \left[ \Phi(X^*) - \Phi \left( X^* - \sigma \sqrt{T} \right) \right] - K e^{-rT} \left[ \Phi(X^*) - \Phi(X^*) \right] \\
& = S_0 \Phi \left( \frac{\log \left( \frac{S_0}{K} \right) + (r + \frac{\sigma^2}{2})T}{\sigma \sqrt{T}} \right) - K e^{-rT} \Phi \left( \frac{\log \left( \frac{S_0}{K} \right) + (r - \frac{\sigma^2}{2})T}{\sigma \sqrt{T}} \right) \\
& = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2) \\
& \text{where, } d_2 = d_1 - \sigma \sqrt{T}
\end{align*}
2 Bachelier model<br>
3 Black76 model<br>
4 Displaced-diffusion model<br>

***

Part 2 : Model Calibration

On 1-Dec-2020, the S&P500 (SPX) index value was 3662.45, while the SPDR<br>
S&P500 Exchange Traded Fund (SPY) stock price was 366.02. The call and<br>
put option prices (bid & offer) over 3 maturities are provided in the<br>
spreadsheet:<br>
• SPX options.csv<br>
• SPY options.csv<br>

The discount rate on this day is in the file: zero rates 20201201.csv.<br>
Calibrate the following models to match the option prices:<br>
1 Displaced-diffusion model<br>
2 SABR model (fix β = 0.7)<br>

Plot the fitted implied volatility smile against the market data.<br>
Report the model parameters:<br>
1 σ, β<br>
2 α, ρ, ν<br>
And discuss how does change β in the displaced-diffusion model and ρ, ν in the<br>
SABR model affect the shape of the implied volatility smile.<br>

***

Part 3 : Static Replication<br>

Suppose on 1-Dec-2020, we need to evaluate an exotic European derivative<br>
expiring on 15-Jan-2021 which pays:<br>
1. Payoff function:<br>
<img width="170" alt="Screenshot 2023-08-31 at 12 09 52 AM" src="https://github.com/andreignatius/stochastic-calculus/assets/7924964/6fc45654-7476-46bf-bc14-d3ef8f59222a"><br>

2. “Model-free” integrated variance:<br>
<img width="153" alt="Screenshot 2023-08-31 at 12 09 56 AM" src="https://github.com/andreignatius/stochastic-calculus/assets/7924964/7eefdba0-7070-4e89-8577-d672ebdd0379">

Determine the price of these 2 derivative contracts if we use:<br>
1 Black-Scholes model (what σ should we use?)<br>
2 Bachelier model (what σ should we use?)<br>
3 Static-replication of European payoff (using the SABR model calibrated<br>
in the previous question)<br>

***

Part 4 : Dynamic Hedging<br>

Suppose S0 = $100, σ = 0.2, r = 5%, T = 1/12 year, i.e. 1 month, and<br>
K = $100. Use a Black-Scholes model to simulate the stock price. Suppose we<br>
sell this at-the-money call option, and we hedge N times during the life of the<br>
call option. Assume there are 21 trading days over the month.<br>

The dynamic hedging strategy for an option is<br>
Ct = ϕtSt − ψtBt,<br>
where<br>

<img width="394" alt="Screenshot 2023-08-31 at 12 00 13 AM" src="https://github.com/andreignatius/stochastic-calculus/assets/7924964/96f00026-fe22-4058-91b8-e52ee8637180">

Work out the hedging error of the dynamic delta hedging strategy by<br>
comparing the replicated position based on ϕ and ψ with the final call option<br>
payoff at maturity.<br>
Use 50,000 paths in your simulation, and plot the histogram of the hedging<br>
error for N = 21 and N = 84.<br>

Reference: http://pricing.free.fr/docs/when_you_cannot_hedge.pdf
