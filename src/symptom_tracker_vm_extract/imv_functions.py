import numpy as np
from scipy.optimize import minimize


def ll(x, p):
    z = (np.log(p) * x) + (np.log(1-p) * (1 - x))
    return np.exp(np.sum(z) / len(z))
    
def get_w(a, guess=0.5, bounds=[(0.001, 0.9999)]):
    res = minimize(minimize_me, guess, args=a,
                   options = {'ftol': 0, 'gtol': 1e-09},
                   method = 'L-BFGS-B', bounds=bounds)
    return res['x'][0]

def minimize_me(p, a):
    return abs((p * np.log(p)) + ((1-p) * np.log(1-p)) - np.log(a))
    
def get_ew(w0, w1):
    return (w1-w0)/w0