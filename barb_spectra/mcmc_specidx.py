from multiprocessing import Pool
from multiprocessing import cpu_count
from barb_spectra.likelihood_specidx import log_ll_specidx
import sys
import emcee
from tqdm import tqdm
import numpy as np
import json
import logging

def sampling_specidx(
    p0, vargroup, cpu_num, nwalkers, ndim, filename="MCMC_results.h5", max_n=100000
):
    """
    MCMC sampler

    Args:
        p0 (float): a uniform random distribution mimicking the distribution of the data
        vargroup ([np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]): nFRBs, sensitivity, R, beams, tpb, flux, freq
            nFRBs: number of FRBs detected
            sensitivity: sensitivity at FWHM divided by 2 (measured in janskys)
            R: telescope beam radius
            beams: number of telescope beams
            tpb: time per beam
            flux: flux measurement of the FRB
            freq: central frequency (measured in MHz)
            bw (float): bandwidth (measured in MHz)
        cpu_num (float): number of cpus
        nwalkers (float):walkers are copies of a system evolving towards a minimum
        ndim (float): number of dimensions to the analysis
        filename (str): name of the output h5 file
        max_n (float): maximum number of iterations the MCMC sampler can run


    Returns:
        old_tau (np.ndarray[float]): variable to test convergence

    """
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    nFRBs, sensitivity, R, beams, tpb, flux, freq, bw = vargroup
    ncpu = cpu_num
    # pool paralelizes the execution of the functions over the cpus
    pool = Pool(ncpu)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_ll_specidx, args=(vargroup), pool=pool, backend=backend
    )
    max_n = max_n

    logging.info("\n" + "In the mcmc sampler function, the value of the likelihood at " + str(np.log10(15)) + ", 2.5 will be calculated with the nFRBs, sensitivity, R, beams, tpb, flux, and freq arrays" + "\n")

    # tracking how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # variable for testing convergence
    old_tau = np.inf

    # sample for up to max_n steps
    # this function samples until it converges at an estimate of
    # rate for the events
    for sample in sampler.sample(p0, iterations=max_n, progress=True):
        # adding this in to the log file to check progress
        logging.info("current place in index: {0}".format(index))
        
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

def read_samples_specidx(filename):
    """
    Analyzes output file to compute the samples from the MCMC sampler

    Args:
        filename (str): name of h5 output file

    Returns:
        samples (np.ndarray[float]): stored chain of MCMC samples

    """
    reader = emcee.backends.HDFBackend(filename)
    tau = reader.get_autocorr_time()
    # computes an estimate of the autocorrelation time for each parameter
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

def convert_params_specidx(samples):
    """
    Converts MCMC samples for the corner plot

    Args:
        samples (np.ndarray[float]): stored chain of MCMC samples

    Returns:
        all_samples (np.ndarray[float]): converted chain of MCMC samples

    """
    all_samples = samples
    # an array of the stored chain of MCMC samples
    all_samples[:, 0] = np.log10(
        (24 * 41253) * (10 ** all_samples[:, 0]) / (all_samples[:, 1] - 1)
    )
    # 0 corresponds to R
    all_samples[:, 1] -= 1
    # 1 corresponds to alpha
    return all_samples
