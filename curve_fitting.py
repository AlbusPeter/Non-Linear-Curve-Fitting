from scipy.optimize import curve_fit
import numpy as np
from nonlinear_functions import vonmises
import matplotlib.pyplot as plt

def curve_Fitting(func, x, y, initials, bounds):
    ## make sure initials and bounds are coherent with func parameters ##
    best_vals, covar = curve_fit(func, x, y, p0=initials, bounds=bounds)
    return best_vals

if __name__ == "__main__":
    x = np.linspace(-np.pi, np.pi, 1000)

    ## generate random von Mises samples ##
    mu = 0
    kappa = 8.0
    amplitude = 10
    y = vonmises(x, kappa, mu, amplitude)
    noise = 1.0 * np.random.normal(size=y.size)
    y += noise

    ## Show original samples
    plt.figure()
    plt.scatter(x, y)

    ## Curve fitting setups
    initial_guess = [4,2,5]
    params_bounds = (0, [20., 5., 20.])

    ## Params will have the same order as you defined
    params = curve_Fitting(vonmises, x, y, initial_guess, params_bounds) ## change the fitting function and corresponding params here

    ## Show fitted curve ##
    plt.plot(x, vonmises(x, params[0], params[1], params[2]), 'r-')
    plt.show()
