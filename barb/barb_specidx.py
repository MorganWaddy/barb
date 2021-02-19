import numpy as np
import pylab as plt
import matplotlib

matplotlib.use("Agg")
import math
from tqdm import tqdm
import emcee
import sys
import json
import logging
import argparse

sys.path.append('bin/')
from likelihood_specidx import area
from likelihood_specidx import power_integral
from likelihood_specidx import likelihood_list
from likelihood_specidx import cummlative_rate
from plotter import plotting
from multiprocessing import cpu_count
from multiprocessing import Pool
# from arg_parser import read_in
from stats import sampling

# sbatch -N1 -n19 --wrap="python barb/barb_specidx.py -f -D surveys/*.json -c 19 -s all_samples_MCMC"
logging.basicConfig(filename="FRB-rate-calc.log", level=logging.INFO)
parser = argparse.ArgumentParser(
    description="""Bayesian Rate-Estimation for FRBs (BaRB)
sample command: python barb_specidx.py -f -D <name of the surveys> -c <number of cpus> -s <name of npz file>

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
    help="to estimate spectral index limits use this flag",
    action="store_true",
    required=False,
    )
args = parser.parse_args()

surveys = []
# Number of FRBs discovered in the survey
nFRBs = []
# Sensitivity at FWHM divided by 2
FWHM_2 = []
# FWHM diameter in arcminutes divided by 2 to get radius divide by 60 to get degrees
R = []
# Number of beams
beams = []
# Time per beam
tpb = []
# Flux of FRB
flux = []
# frequency of observation
freq = []
# data feed in structure
j = len(args.dat)
k = j + 1
filename = args.dat

if j > 0:
    logging.info("Sanity Check:")
    logging.info("The number of user file(s) supplied is {0}".format(j) + "\n")
    logging.info("The supplied file(s) is/are {0}".format(filename) + "\n")
    for e in args.dat:
        with open(e, "r") as fobj:
            info = json.load(fobj)
            for p in info["properties"]:
                surveys.append(p["surveys"])
                nFRBs = np.append(nFRBs, p["nFRBs"])
                FWHM_2 = np.append(FWHM_2, p["FWHM_2"])
                R = np.append(R, p["R"])
                beams = np.append(beams, p["beams"])
                tpb = np.append(tpb, p["tpb"])
                flux = np.append(flux, p["flux"])
                if args.freq is True:
                    for e in args.dat:
                        with open(e, "r") as fobj:
                            info = json.load(fobj)
                            freq = np.append(freq, p["freq"])
                else:
                    freq = np.append(freq, 1)
                    fobj.close()
else:
    logging.info("No data was supplied, please supply data on the command line!")
nFRBs = np.array(nFRBs)
FWHM_2 = np.array(FWHM_2)
R = np.array(R)
beams = np.array(beams)
tpb = np.array(tpb)
if args.freq is True:
    freq = np.array(freq)
    
    
time = tpb * beams
flux = np.array(flux)


global data
if args.freq is True:
    data = nFRBs, R, time, FWHM_2, flux, freq
else:
    data = nFRBs, R, time, FWHM_2, flux
logging.info(str(data) + "\n")


def log_ll(junk):
    alpha, beta = junk
    if args.freq is True:
        nFRBs, R, time, FWHM_2, flux, freq = data
    else:
        nFRBs, R, time, FWHM_2, flux = data
    alpha = 10 ** alpha
    if beta < 1:
        return -np.inf
    return likelihood_list(data, alpha=alpha, beta=beta)

log_ll([0.97504624, 1.91163861])
# log_ll([alpha,beta])

logly = log_ll([0.97504624, 1.91163861])
logging.info("log_ll([0.97504624, 1.91163861]) = " + str(logly) + "\n\n")

bb = np.linspace(0.1, 3, 100)
lk = [log_ll([np.log10(586.88 / (24 * 41253)), b]) for b in bb]

bb[np.argmin(lk)]


plt.plot(bb - 1, -1 * np.array(lk))
plt.yscale("log")
plt.savefig("log_check.png")
plt.close("log_check.png")
plt.clf()
# in the mcmc you maximize the likelihood, this is to check if the functions are working

ndim, nwalkers = 2, 1200
# walkers are copies of a system evolving towards a minimum
ivar = np.array([np.log10(15), 2.5])
# ivar is an intermediate variable for sampling
logging.info("ivar is " + str(ivar) + "\n\n")


p0 = ivar + 0.05 * np.random.uniform(size=(nwalkers, ndim))
# returns a uniform random distribution mimicking the distribution of the data
plt.hist(p0[:, 0])
plt.savefig("MCMC_hist1-check.png")
plt.close("MCMC_hist1-check.png")
plt.clf()
plt.hist(p0[:, 1])
plt.savefig("MCMC_hist2-check.png")
plt.close("MCMC_hist2-check.png")
plt.clf()

cups = "".join(args.cpus)
ncpu = int(cups)
logging.info("{0} CPUs".format(ncpu))

pool = Pool(ncpu)
# pool paralelizes the execution of the functions over the cpus
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_ll, pool=pool)

max_n = 100000

# tracking how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# variable for testing convergence
old_tau = np.inf

# sample for up to max_n steps
old_tau = sampling(p0)

pool.close()
# this function samples until it converges at a estimate of rate for the events

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

logging.info("burn-in: {0}".format(burnin))
logging.info("thin: {0}".format(thin))
logging.info("flat chain shape: {0}".format(samples.shape))
logging.info("flat log prob shape: {0}".format(log_prob_samples.shape))


all_samples = samples
# an array of the stored chain of MCMC samples
all_samples[:, 0] = np.log10(
    (24 * 41253) * (10 ** all_samples[:, 0]) / (all_samples[:, 1] - 1)
)
# 0 corresponds to R
all_samples[:, 1] -= 1
# 1 corresponds to alpha

plotting(all_samples)
