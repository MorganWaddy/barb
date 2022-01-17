import pytest
from multiprocessing import Pool
from multiprocessing import cpu_count
import emcee
from tqdm import tqdm
import numpy as np
import json
import logging

from barb.likelihood import area
from barb.likelihood import power_integral
from test_likelihood_list import likelihood_list
from test_log_ll import log_ll
from barb.mcmc import sampling

vargroup = (
    [4],
    12,
    5,
    13,
    np.array([6, 6, 6, 6]),
    [9, 9, 9, 9],
)
vargroup = np.array(vargroup, dtype=object)
cpu_num=1
ndim, nwalkers = 2, 12
ivar = np.array([np.log10(15), 2.5])
p0 = ivar + 0.05 * np.random.uniform(size=(nwalkers, ndim))
filename="test_MCMC_results.h5"
max_n=10

def test_sampling():
    old_tau = sampling(
        p0, vargroup, cpu_num, nwalkers, ndim, filename=h5name, max_n=max_n
    )
    assert old_tau ==

