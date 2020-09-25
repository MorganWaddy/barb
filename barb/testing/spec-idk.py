import numpy as np
import pylab as plt
import matplotlib

matplotlib.use("Agg")
import math
from tqdm import tqdm
import emcee
import argparse
import sys

# cummulative rate of FRBs 
def incd(f_inc, hunk):
    a, b = hunk
    r_incd = a * np.power(f_inc, -b)
    return r_incd

global hunk

# instantaneous rate 
def obs(f_inc, radius, p):
    rs = np.power(radius/R, 2)
    f_obs = f_inc * p
    # p is the function representing beam attenuation
    rat = np.power(2, rs) * f_obs
    r_obs = a * b * np.power(rat, -b-1) * np.power(2, rs) * 2 * np.pi * radius
    return r_obs

# log likelihood
def logger(
