import numpy as np
import pylab as plt
import matplotlib

matplotlib.use("Agg")
import sys
import arg_stater
import math
from tqdm import tqdm
import emcee
import json
import logging
import argparse
import glob
import arg_parser

from likelihood_specidx import area
from likelihood_specidx import power_integral
from likelihood_specidx import likelihood_list
from likelihood_specidx import log_ll
from likelihood_specidx import cummlative_rate
from plotter import plotting
from plotter import check_work1
from plotter import check_work2
from MCMC_definer import MCMCdef
from MCMC_definer import cpu_num
from multiprocessing import cpu_count
from multiprocessing import Pool
from stats import sampling
from plotter import tau_definer
from plotter import storing
from arg_parser import read_in

# sbatch -N1 -n19 --wrap="python barb/barb_specidx_test.py -f -D surveys/*.json -c 19 -s all_samples_MCMC"

logging.basicConfig(filename="FRB-rate-calc.log", level=logging.INFO)
logging.info("The logging file was created" + "\n")

# surveys = []
# Number of FRBs discovered in the survey
# nFRBs = []
# Sensitivity at FWHM divided by 2
# FWHM_2 = []
# FWHM diameter in arcminutes divided by 2 to get radius divide by 60 to get degrees
# R = []
# Number of beams
# beams = []
# Time per beam
# tpb = []
# Flux of FRB
# flux = []
# frequency of observation
# freq = []
# data feed in structure

j = len(args.dat)
k = j + 1
jsons = args.dat

read_in(jsons)
jsons = glob.glob('{}'.format(jsons))
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


log_ll([0.97504624, 1.91163861], data)
# log_ll([alpha,beta])
logly = log_ll([0.97504624, 1.91163861], data)
logging.info("log_ll([0.97504624, 1.91163861]) = " + str(logly) + "\n\n")

bb = np.linspace(0.1, 3, 100)
lk = [log_ll([np.log10(586.88 / (24 * 41253)), b], data) for b in bb]

bb[np.argmin(lk)]

check_work1(bb, lk)

logging.info("ivar is " + str(ivar) + "\n\n")

check_work2(p0)

logging.info("{0} CPUs".format(ncpu))

# sample for up to max_n steps
junk =  alpha, beta
old_tau = sampling(p0, data, junk, args.cpus)

pool.close()
# this function samples until it converges at a estimate of rate for the events

logging.info("burn-in: {0}".format(burnin))
logging.info("thin: {0}".format(thin))
logging.info("flat chain shape: {0}".format(samples.shape))
logging.info("flat log prob shape: {0}".format(log_prob_samples.shape))

tau, burnin, thin, samples, log_prob_samples = tau_definer(p0, data, junk, args.cpus)
all_samples = storing(samples)
plotting(all_samples)

