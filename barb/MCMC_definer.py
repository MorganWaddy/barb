import numpy as np
import math
import sys
import argparse
import logging
import emcee
import arg_stater

def MCMCdef():
    ndim, nwalkers = 2, 1200
    # walkers are copies of a system evolving towards a minimum
    ivar = np.array([np.log10(15), 2.5])
    # ivar is an intermediate variable for sampling

    p0 = ivar + 0.05 * np.random.uniform(size=(nwalkers, ndim))
    # returns a uniform random distribution mimicking the distribution of the data
    return ndim, nwalkers, ivar, p0 

def cpu_num(cup_num):
    cups = "".join(args.cpus)
    ncpu = int(cups)
    return ncpu
