import numpy as np
import pylab as plt
import matplotlib

matplotlib.use("Agg")
import math
from tqdm import tqdm
import emcee
import argparse
import sys
import scipy as sp

# cummulative rate of FRBs 
def incd(f_inc, hunk):
    a, b = hunk
    r_incd = a * np.power(f_inc, -b)
    return r_incd

global hunk

# instantaneous rate 
def obs(f_inc, radius, p):
    rs = np.power(radius/R, 2)
    # p = ?
    # p is the function representing beam attenuation
    f_obs = f_inc * p
    rat = np.power(2, rs) * f_obs
    r_obs = a * b * np.power(rat, -b-1) * np.power(2, rs) * 2 * np.pi * radius
    return r_obs

# log likelihood
def logger(flux, radius, p):
    r_obs = obs(flux, radius, p)
    f1 = lambda u: r_obs
    integral1 = sp.integrate.quad(f1, 0, np.inf)
    f2 = lambda u, v: r_obs
    integral2 = sp.integrate.dblquad(f2, 0, np.inf, lambda v: 0, lambda v: np.inf)
    l1 = -tpb*(integral1) + np.sum(math.log(r_obs[, base]))
    l2 = -tpb*(integral2) + np.sum(math.log(r_obs[, base]))
    loggy = np.sum([l1, l2])
    return loggy
