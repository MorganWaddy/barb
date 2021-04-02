from multiprocessing import Pool
from multiprocessing import cpu_count
import argparse
import logging
import json
import sys
import emcee
from tqdm import tqdm
import math
import numpy as np
import pylab as plt
import matplotlib

matplotlib.use("Agg")

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
    action="store_true",
    help="to estimate spectral index limits use this flag",
    required=False,
)
args = parser.parse_args()

global data
if args.freq is True:
    data = nFRBs, R, time, FWHM_2, flux, freq
else:
    data = nFRBs, R, time, FWHM_2, flux

def log_ll(junk, data):
    alpha, beta = junk
    if args.freq is True:
        nFRBs, R, time, FWHM_2, flux, freq = data
    else:
        nFRBs, R, time, FWHM_2, flux = data
    alpha = 10 ** alpha
    if beta < 1:
        return -np.inf
    return likelihood_list(data, alpha=alpha, beta=beta)


def sampling(varr2):
    ndim, nwalkers = 2, 1200
    cups = "".join(args.cpus)
    ncpu = int(cups)
    pool = Pool(ncpu)
    if args.freq is True:
        nFRBs, R, time, FWHM_2, flux, freq = data
    else:
        nFRBs, R, time, FWHM_2, flux = data
   
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_ll, pool=pool)
    max_n=100000
    # tracking how the average autocorrelation time estimate changes
    index=0
    autocorr=np.empty(max_n)
    for sample in sampler.sample(varr2, iterations=max_n, progress=True):
        # only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # compute the autocorrelation time so far
        tau=sampler.get_autocorr_time(tol=0)
        autocorr[index]=np.mean(tau)
        index += 1

        # check convergence
        converged=np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau=tau
    return old_tau
