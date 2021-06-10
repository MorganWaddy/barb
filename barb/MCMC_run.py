from multiprocessing import Pool
from multiprocessing import cpu_count
# from likelihood_specidx import log_ll
import sys
import emcee
from tqdm import tqdm
import numpy as np
import json
import logging

def sampling(p0, vargroup, cpu_num, nwalkers, ndim, filename = 'MCMC_results.h5'):
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    ncpu = cpu_num
    # pool = Pool(ncpu)
    # pool paralelizes the execution of the functions over the cpus
    nFRBs, R, time, FWHM_2, flux = vargroup
   
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_ll, args = (vargroup), backend = backend)
    max_n=100000
    # tracking how the average autocorrelation time estimate changes
    index=0
    autocorr=np.empty(max_n)
    # variable for testing convergence
    old_tau = np.inf
    # sample for up to max_n steps
    for sample in sampler.sample(p0, iterations=max_n, progress=True):
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
    return old_tau, sampler
