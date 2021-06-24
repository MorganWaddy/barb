from multiprocessing import Pool
from multiprocessing import cpu_count
from barb.likelihood import log_ll
import sys
import emcee
from tqdm import tqdm
import numpy as np
import json
import logging


def sampling(
    p0, vargroup, cpu_num, nwalkers, ndim, filename="MCMC_results.h5", 
    max_n=100000
    ):
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    nFRBs, FWHM_2, R, beams, tpb, flux = vargroup
    ncpu = cpu_num
    # pool paralelizes the execution of the functions over the cpus
    pool = Pool(ncpu)
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_ll, pool=pool, backend=backend
        )
    max_n = max_n

    # tracking how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # variable for testing convergence
    old_tau = np.inf

    # sample for up to max_n steps
    # this function samples until it converges at a estimate of 
    # rate for the events
    for sample in sampler.sample(p0, iterations=max_n, progress=True):
        # only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
        
        # compute the autocorrelation time so far
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        
        # check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
    pool.close()
    return old_tau


def read_samples(filename):
    reader = emcee.backends.HDFBackend(filename)
    tau = reader.get_autocorr_time()
    # computes an estimate of the autocorrelation time for each parameter
#    if np.isnan(tau).any():
#        burnin = 100
#        thin = 1
#    else:
    burnin = int(2 * np.max(tau))
    # steps that should be discarded
    thin = int(0.5 * np.min(tau))
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    # gets the stored chain of MCMC samples

    logging.info("burn-in: {0}".format(burnin))
    logging.info("thin: {0}".format(thin))
    logging.info("flat chain shape: {0}".format(samples.shape))
    logging.info("samples: {0}".format(samples))
    return samples


def convert_params(samples):
    all_samples = samples
    # an array of the stored chain of MCMC samples
    all_samples[:, 0] = np.log10(
        (24 * 41253) * (10 ** all_samples[:, 0]) / (all_samples[:, 1] - 1)
    )
    # 0 corresponds to R
    all_samples[:, 1] -= 1
    # 1 corresponds to alpha
    return all_samples
