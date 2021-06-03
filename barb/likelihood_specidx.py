import numpy as np
import pylab as plt
import matplotlib

matplotlib.use("Agg")
import sys
import arg_stater
import math
import argparse
import logging
import emcee

def area(R, gamma):
    # beam shape
    # b is Euclidean scaling that is valid for any population with a fixed
    # luminosity distribution, as long as the luminosity does not evolve with
    # redshift and the population has a uniform spatial distribution
    ar = (np.pi * R ** 2) / (gamma * np.log(2))
    beta = gamma + 1
    return ar
# nothing should change for this function

def power_integral(FWHM_2, beta):
    # references the cumm. rate for observed fluxes > 0 (from paper)
    return (FWHM_2 ** -(beta - 1))
    # FWM_2 is sigma, measured in janskys
    # from appendix A:
    # beta = gamma + 1

def freq_term(freq, freq_0, powterm):
    # alpha = Rref*gamma
    freqqy = (freq/freq_0) ** -(powterm)
    return freqqy


def likelihood_list(vargroup, alpha, beta):
    # runs through all data to return the likelihood that there will be
    # an FRB
    # a = Rref(freqqy**(alpha*gamma))
    if args.freq is True:
        nFRBs, R, time, FWHM_2, flux, freq = vargroup
        powterm = alpha * (beta - 1)
        freqqy = freq_term(freq, 1, powterm)
    else:
        nFRBs, R, time, FWHM_2, flux = vargroup
    A = area(R, beta - 1)
    gamma = beta - 1
    Rref = alpha / gamma
    I = (power_integral(FWHM_2, beta))
    taa = time * A * alpha
    ll = 0
    bandw = 960 # MHz
    for idx, nburst in enumerate(nFRBs):
        # idx is just a number that identifies a place in the array
        if flux[idx] == [-1]:
            val = -taa[idx] * I[idx]
        else:
            if args.freq is True:
                val = (
                        -(((taa[idx] * Rref) / (1 - (alpha *gamma))) * I[idx]
                         * (bandw ** (1 - (alpha *gamma))))
                         + nburst * np.log(taa[idx])
                         + np.sum(np.log((Rref * ((np.pi*R**2)/( np.log(2))))*freqqy[idx]*(flux[idx])**(-gamma - 1)))
                )
            else:
                val = (
                    -taa[idx] * I[idx]
                     + nburst * np.log(taa[idx])
                     - beta * np.sum(np.log(flux[idx]))
                )
        ll += val
    return ll

def log_ll(varrest, vargroup):
    alpha, beta = varrest
    if args.freq is True:
        nFRBs, R, time, FWHM_2, flux, freq = vargroup
    else:
        nFRBs, R, time, FWHM_2, flux = vargroup
    alpha = 10 ** alpha
    if beta < 1:
        return -np.inf
    return likelihood_list(vargroup, alpha=alpha, beta=beta)


def cummlative_rate(flux, freq, Rref, gamma):
    # returns the cummulative rate of events
    return Rref*(freq**(alpha*gamma))*(flux**(-gamma-1))
