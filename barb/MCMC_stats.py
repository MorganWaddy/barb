from multiprocessing import Pool
from multiprocessing import cpu_count
# from likelihood_specidx import log_ll
import sys
import emcee
from tqdm import tqdm
import numpy as np
import json
import logging

def tau_definer(varr2, vargroup, varrest, cpu_num):
    pool = Pool(ncpu)
    # pool paralelizes the execution of the functions over the cpus
    alpha, beta = varrest
    if args.freq is True:
        nFRBs, R, time, FWHM_2, flux, freq = vargroup
    else:
        nFRBs, R, time, FWHM_2, flux = vargroup
   
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_ll(varrest, vargroup), pool=pool)

    tau = sampler.get_autocorr_time()
    # computes an estimate of the autocorrelation time for each parameter
    burnin = int(2 * np.max(tau))
    # steps that should be discarded
    thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    # gets the stored chain of MCMC samples
    # flatten the chain across ensemble, take only thin steps, discard burn-in
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    # gets the chain of log probabilities evaluated at the MCMC samples
    # discard burn-in steps, flatten chain across ensemble, take only thin steps
    return tau, burnin, thin, samples, log_prob_samples

def storing(samples):
    all_samples = samples
    # an array of the stored chain of MCMC samples
    all_samples[:, 0] = np.log10(
        (24 * 41253) * (10 ** all_samples[:, 0]) / (all_samples[:, 1] - 1)
        )
    # 0 corresponds to R
    all_samples[:, 1] -= 1
    # 1 corresponds to alpha
    return all_samples
