"""
Honest Backtesting: A Walk-Forward Signal Evaluator

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - to_log_returns
import numpy as np 

def to_log_returns(prices):
    prices = np.array(prices)
    log_returns = np.log(np.append(prices[1:], 1)) - np.log(prices)
    return log_returns[:-1]

# Step 2 - rolling_zscore
import numpy as np

def rolling_zscore(x, window):
    n = len(x)
    zscore = np.zeros(n)
    for i in range(n):
        if i < window - 1:
            zscore[i] = np.nan
            continue
        i_window = x[i - window + 1:i+1]
        i_window_std = np.std(i_window)
        if i_window_std == 0:
            zscore[i] = 0
            continue
        zscore[i] = (x[i] - np.mean(i_window)) / i_window_std
    
    return zscore

