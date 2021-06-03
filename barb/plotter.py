import numpy as np
import pylab as plt
import matplotlib

matplotlib.use("Agg")
import math
import sys
import argparse
import logging
import emcee
import arg_stater
from MCMC_definer import cpu_num
from MCMC_definer import MCMCdef

def check_work1(bb, lk):
    plt.plot(bb - 1, -1 * np.array(lk))
    plt.yscale("log")
    plt.savefig("log_check.png")
    plt.close("log_check.png")
    plt.clf()
# in the mcmc you maximize the likelihood, this is to check if the functions are working

def check_work2(varr1):
    plt.hist(varr1[:, 0])
    plt.savefig("MCMC_hist1-check.png")
    plt.close("MCMC_hist1-check.png")
    plt.clf()
    plt.hist(varr1[:, 1])
    plt.savefig("MCMC_hist2-check.png")
    plt.close("MCMC_hist2-check.png")
    plt.clf()

def tau_definer(varr2, vargroup, varrest, cpu_num):
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

def plotting(allsamples):
    labels = [r"$\log \mathcal{R}$", r"$\alpha$"]

    np.savez("{0}".format(args.allsamples), allsamples)
    logging.info("the name of the np array file is {0}".format(args.allsamples) + ".npz")
    allsamples

    allsamples = np.load("{0}".format(args.allsamples) + ".npz")["arr_0"]

    quantile_val = 0.99

    import corner
    
    plt.figure(figsize=(15, 15))
    corner.corner(
        allsamples,
        labels=labels,
        quantiles=[(1 - 0.99) / 2, 0.5, 1 - (1 - 0.99) / 2],
        show_titles=True,
        bins=50,
        )
    
    # makes a corner plot displaying the projections of prob. distr. in space
    plt.savefig("rates_mc.png")
    plt.close("rates_mc.png")
    plt.clf()
