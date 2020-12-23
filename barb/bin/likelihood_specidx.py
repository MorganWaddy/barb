import numpy as np
import pylab as plt
import matplotlib

matplotlib.use("Agg")
import math
import sys
import argparse

parser = argparse.ArgumentParser(
    description="""Bayesian Rate-Estimation for FRBs (BaRB)
sample command: python barb.py -f -D <name of the surveys> -c <number of cpus> -s <name of npz file>

Surveys on the original data set: Agarwal 2019, Masui 2015, Men 2019, Bhandari 2018, Golpayegani 2019, Oslowski 2019, GREENBURST, Bhandari 2019, Qiu 2019, Shannon 2018, Madison 2019, Lorimer 2007, Deneva 2009, Keane 2010, Siemion 2011, Burgay 2012, Petroff 2014, Spitler 2014, Burke-Spolaor 2014, Ravi 2015, Petroff 2015, Law 2015, Champion 2016""",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "-D",
    "--dat",
    help="supply the input data after this flag",
    action="store",
    nargs="+",
    required=True,
)
parser.add_argument(
    "-c",
    "--cpus",
    help="supply the number of cpus you want to be used",
    action="store",
    nargs="+",
    required=True,
)
parser.add_argument(
    "-s",
    "--allsamples",
    help="supply the name of the output numpy array of the samples",
    nargs=1,
    action="store",
    required=True,
)
parser.add_argument(
    "-f",
    "--freq",
    action="store",
    nargs='?',
    help="to estimate spectral index limits use this flag",
    required=False,
)
args = parser.parse_args()

def area(radius, gamma):
    # beam shape
    # b is Euclidean scaling that is valid for any population with a fixed
    # luminosity distribution, as long as the luminosity does not evolve with
    # redshift and the population has a uniform spatial distribution
    ar = np.pi * radius * radius / (gamma * np.log(2))
    beta = gamma + 1
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
    freqqy = (freq/freq_0)**(alpha*gamma)
    return freqqy


def likelihood_list(data, alpha, beta):
    # runs through all data to return the likelihood that there will be
    # an FRB
    # a = Rref(freqqy**(alpha*gamma))
    if args.freq is True:
        freq, nFRBs, radius, time, sensitivity, flux = data
        freqqy = freq_term(freq, freq_0)
    else:
        nFRBs, radius, time, sensitivity, flux = data
    A = area(radius, beta - 1)
    gamma = beta - 1
    Rref = alpha / gamma
    I = (-gamma)*Rref*((power_integral(sensitivity, beta))**(-gamma))
    taa = time * A * alpha
    ll = 0
    for idx, nburst in enumerate(nFRBs):
        # idx is just a number that identifies a place in the array
        if flux[idx] == [-1]:
            val = -taa[idx] * I[idx]
        else:
            if args.freq is True:
                val = (
                    -taa[idx] * I[idx] * freqqy
                     + nburst * np.log(taa[idx])
                     - beta * np.sum(np.log(((flux[idx])**((-gamma)-1))*(freqqy)))
                )
            else:
                val = (
                    -taa[idx] * I[idx]
                     + nburst * np.log(taa[idx])
                     - beta * np.sum(np.log(flux[idx]))
                )
        ll += val
    return ll


def cummlative_rate(flux, freq, Rref, gamma):
    # returns the cummulative rate of events
    return Rref*(freq**(alpha*gamma))*(flux**(-gamma-1))
