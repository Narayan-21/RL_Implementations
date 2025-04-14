# American Put option =- Binomial Tree model 
# Binomial tree model - over small steps the model will simulate if the price go up or down. At each point in time, the model checks if it's better to hold the option or exercise it.

import numpy as np
import yfinance as yf

def binomial_american_put(S0, K, T, r, sigma, N=100):
    dt = T / N
    # assumption - up factor
    up = np.exp(sigma*np.sqrt(dt))
    down = 1 / up
    p = (np.exp(r*dt) - down) / (up- down)
    
    # Initialize stock price tree
    stock_tree = np.zeros((N+1, N+1))
    stock_tree[0,0] = S0
    for i in range(1, N+1):
        stock_tree[i, 0] = stock_tree[i-1, 0] * up
        for j in range(1, i+1):
            stock_tree[i,j] = stock_tree[i-1, j-1] * down
            
    # Initialize option value tree
    option_tree = np.zeros((N+1, N+1))
    for j in range(N+1):
        option_tree[N, j] = max(K - stock_tree[N, j], 0) # Payoff at maturity
        
    # Backward induction
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            exercise = max(K-stock_tree[i,j], 0)
            hold = np.exp(-r * dt) * (p * option_tree[i+1, j] + (1-p) * option_tree[i+1, j+1])
            option_tree[i, j] = max(exercise, hold)
    return option_tree[0,0]


aapl = yf.Ticker("AAPL")
S0 = aapl.history(period="1d")['Close'].iloc[-1]  # Latest AAPL price
print("Latest apple price", S0)
K = S0 * 0.95  # Strike price (5% below current price)
T = 1.0  # 1 year
r = 0.05  # Risk-free rate (5%)
sigma = 0.2  # Historical volatility

put_price = binomial_american_put(S0, K, T, r, sigma)
print(f"American Put Option Price: ${put_price:.2f}")
