from scipy.optimize import curve_fit
import numpy as np
from nonlinear_functions import vonmise_derivative
import matplotlib.pyplot as plt

def curve_Fitting(func, x, y, initials, bounds):
    ## make sure initials and bounds are coherent with func parameters ##
    best_vals, covar = curve_fit(func, x, y, p0=initials, bounds=bounds)
    return best_vals

if __name__ == "__main__":
    x = np.linspace(-np.pi, np.pi, 1000)

    ## generate random von Mises samples ##
    kappa = 8.0
    amplitude = 10
    y = vonmise_derivative(x, amplitude, kappa)
    noise = 1.0 * np.random.normal(size=y.size)
    y += noise

    ## Show original samples
    plt.figure()
    plt.scatter(x, y)

    ## Curve fitting setups
    initial_guess = [4,5]
    # params_bounds = (0, [20., 20.])
    params_bounds = ([0, 0], [20., 20.])
    bootstrap = True
    bsIter = 100
    permutation = True
    permIter = 100

    ## Params will have the same order as you defined
    params = curve_Fitting(vonmise_derivative, x, y, initial_guess, params_bounds) ## change the fitting function and corresponding params here

    ## Show fitted curve ##
    plt.plot(x, vonmise_derivative(x, params[0], params[1]), 'r-')
    plt.show()

    ## Bootstrap ##
    if bootstrap:
        OutA = [] # Output a array, store each trial's a
        bsSize = int(1.0 * len(x))
        for i in range(bsIter):
            RandIndex = np.random.choice(len(x), bsSize, replace=True) # get randi index of xdata
            xdataNEW = [x[i] for i in RandIndex] # change xdata index
            ydataNEW = [y[i] for i in RandIndex] # change ydata index
            try:
                temp_best_vals = curve_Fitting(vonmise_derivative, xdataNEW, ydataNEW, initial_guess, params_bounds)
                new_x = np.linspace(-np.pi, np.pi, 300)
                new_y = [vonmise_derivative(xi,temp_best_vals[0],temp_best_vals[1]) for xi in new_x]
                if new_x[np.argmax(new_y)] > 0: 
                    OutA.append(np.max(new_y))
                else: 
                    OutA.append(-np.max(new_y))
            except RuntimeError:
                pass
        print("bs_a:",round(np.mean(OutA),2),"	95% CI:",np.percentile(OutA,[2.5,97.5]))
        # np.save('amplitude_bootstrap.npy',OutA)
    
    if permutation:
        # perm_a, perm_b = repeate_sampling('perm', xdata, ydata, CurvefitFunc, size = permSize)
        OutA = [] # Output a array, store each trial's a
        perm_xdata = x
        for i in range(permIter):
            perm_xdata = np.random.permutation(perm_xdata) # permutate nonlocal xdata to update, don't change ydata
            try:
                temp_best_vals = curve_Fitting(vonmise_derivative, perm_xdata, y, initial_guess, params_bounds) # permutation make a sample * range(size) times
                new_x = np.linspace(-np.pi, np.pi, 300)
                new_y = [vonmise_derivative(xi,temp_best_vals[0],temp_best_vals[1]) for xi in new_x]
                if new_x[np.argmax(new_y)] > 0: 
                    OutA.append(np.max(new_y))
                else: 
                    OutA.append(-np.max(new_y))
            except RuntimeError:
                pass
        print("perm_a:",round(np.mean(OutA),2),"	90% CI:",np.percentile(OutA,[5,95]))
        # np.save('amplitude_permutation.npy',OutA)
