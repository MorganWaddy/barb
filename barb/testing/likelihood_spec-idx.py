import numpy as np
import pylab as plt
import matplotlib

matplotlib.use("Agg")
import math
import sys

def area(radius, gamma):
    # beam shape
    # b is Euclidean scaling that is valid for any population with a fixed
    # luminosity distribution, as long as the luminosity does not evolve with
    # redshift and the population has a uniform spatial distribution
    ar = np.pi * radius * radius / (gamma * np.log(2))
    return ar
# nothing should change for this function

def power_integral(sensitivity, beta):
    # references the cumm. rate for observed fluxes > 0 (from paper)
    return (sensitivity ** -(beta - 1)) / (beta - 1)
    # sensitivity is sigma, measured in janskys
    # from appendix A:
    # beta = gamma + 1

def freq_term(freq, freq_0):
    # alpha = Rref*gamma
    freq_0 = 1
    return (freq/freq_0)**(alpha*gamma)


def likelihood_list(data, alpha, beta):
    # runs through all data to return the likelihood that there will be
    # an FRB
    # a = Rref(freq/freq_0)**(alpha*gamma)
    if args.freq is not None:
        freq, nFRBs, radius, time, sensitivity, flux = data
    else:
        nFRBs, radius, time, sensitivity, flux = data
    A = area(radius, beta - 1)
    I = (-gamma)*((power_integral(sensitivity, beta))**(-gamma))
    taa = time * A * alpha
    ll = 0
    for idx, nburst in enumerate(nFRBs):
        # idx is just a number that identifies a place in the array
        if flux[idx] == [-1]:
            val = -taa[idx] * I[idx]
        else:
            val = (
                -taa[idx] * I[idx]
                + nburst * np.log(taa[idx])
                - beta * np.sum(np.log(((flux[idx])**((-gamma)-1))*(freq**(alpha*gamma))))
            )
        ll += val
    return ll


def cummlative_rate(flux, freq, Rref, gamma):
    # returns the cummulative rate of events
    return Rref*(freq**(alpha*gamma))*(flux**(-gamma-1))
