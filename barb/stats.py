from multiprocessing import Pool
from multiprocessing import cpu_count
from likelihood_specidx import log_ll
from MCMC_definer import cpu_num 
from MCMC_definer import MCMCdef
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
import arg_stater

def sampling(varr2, vargroup, varrest, cpu_num):
    ndim, nwalkers, ivar, p0 = MCMCdef()
    ncpu = cpu_num
    pool = Pool(ncpu)
    # pool paralelizes the execution of the functions over the cpus
    alpha, beta = varrest
    if args.freq is True:
        nFRBs, R, time, FWHM_2, flux, freq = vargroup
    else:
        nFRBs, R, time, FWHM_2, flux = vargroup
   
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_ll(varrest, vargroup), pool=pool)
    max_n=100000
    # tracking how the average autocorrelation time estimate changes
    index=0
    autocorr=np.empty(max_n)
    # variable for testing convergence
    old_tau = np.inf
    # sample for up to max_n steps
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
