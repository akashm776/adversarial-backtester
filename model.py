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

