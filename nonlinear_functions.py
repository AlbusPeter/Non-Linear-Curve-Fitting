import numpy as np
from numpy import exp, cos, sin, pi
from scipy.special import i0, gamma

## Exponential ##
def exponential(x, a, b, c):
    return a * exp(-b * x) + c

## Gaussian ##
def Gaussian(x, mu, sigma2, amplitude):
    return amplitude * exp(-(x-mu)**2 / sigma2)

## Derivative of Gaussian ##
def DoG(x, mu, sigma2, amplitude):
    return - amplitude * x * exp(-(x-mu)**2 / sigma2)

## von Mises ## mu = 0
def vonmise(x, kappa, amplitude):
    return amplitude / (i0(kappa) * 2 * pi) * exp(kappa * cos(x))

def vonmise_derivative(xdata, a, kai):
    # xdata = xdata / 75 * np.pi
    return - a / (i0(kai) * 2 * np.pi) * exp(kai * cos(xdata)) * kai * sin(xdata) # Derivative of vonmise formula

## Derivative of von Mises ##
def DoV(x, kappa, mu, amplitude):
    return - amplitude / (i0(kappa) * 2 * pi) * exp(kappa * cos(x - mu)) * kappa * sin(x - mu)

## Gamma Distribution ##
def Gamma(xdata, a, alpha, beta):
    return a * np.power(beta, alpha) * np.power(xdata, alpha - 1) * exp(-beta * xdata) / gamma(alpha)