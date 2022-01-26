import pytest
from multiprocessing import Pool
from multiprocessing import cpu_count
import emcee
from tqdm import tqdm
import numpy as np
import json
import logging
import os

from barb.likelihood import area, power_integral, likelihood_list, log_ll
from barb.mcmc import sampling, convert_params, read_samples

vargroup = (
    [4],
    12,
    5,
    13,
    np.array([6]),
    [9],
)
vargroup = np.array(vargroup, dtype=object)
cpu_num = 1
ndim, nwalkers = 2, 12
ivar = np.array([np.log10(15), 2.5])
p0 = ivar + 0.05 * np.random.uniform(size=(nwalkers, ndim))
h5name = "test_MCMC_results.h5"
max_n = 10


def test_sampling():
    old_tau = sampling(
        p0, vargroup, cpu_num, nwalkers, ndim, filename=h5name, max_n=max_n
    )
    assert old_tau == np.inf


def test_read_samples():
    h5name = "test_MCMC_results.h5"
    assert os.path.isfile(h5name)

def test_convert_params():
    samples = read_samples(h5name)
    converted_params = convert_params(samples)
    if not converted_params:
        test = good
    else:
        test = bad
    assert test == good
