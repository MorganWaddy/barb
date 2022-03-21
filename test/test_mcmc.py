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
from barb.plotting import make_corner

vargroup = (
    np.array([4]),
    np.array([12]),
    np.array([5]),
    np.array([13]),
    np.array([6]),
    np.array([9]),
)

vargroup = np.array(vargroup, dtype=object)
cpu_num = 1
ndim, nwalkers = 2, 12
ivar = np.array([np.log10(15), 2.5])
p0 = ivar + 0.05 * np.random.uniform(size=(nwalkers, ndim))
h5name = "test_MCMC_results.h5"
max_n = 10
cornername = "Test_MCMC.png"


def test_sampling():
    old_tau = sampling(
        p0, vargroup, cpu_num, nwalkers, ndim, filename=h5name, max_n=max_n
    )
    assert old_tau == np.inf


def test_read_samples():
    assert os.path.isfile(h5name)


def test_convert_params():
    samples = [
        [-3.08573175, 1.85962243],
        [-2.95279279, 1.86707565],
        [-3.07843741, 1.8282952],
        [-3.1737073, 1.77846164],
        [-3.10455859, 1.7959741],
        [-3.03914302, 1.81618995],
    ]
    converted_params = convert_params(np.array(samples))
    assert converted_params.any()
    assert converted_params == pytest.approx(
        np.array(
            [
                [2.97562729, 0.85962243],
                [3.104817, 0.86707565],
                [2.99904422, 0.8282952],
                [2.93072226, 0.77846164],
                [2.99020925, 0.7959741],
                [3.04473252, 0.81618995],
            ]
        ),
        0.000001,
    )


def test_make_corner():
    samples = [
        [-3.08573175, 1.85962243],
        [-2.95279279, 1.86707565],
        [-3.07843741, 1.8282952],
        [-3.1737073, 1.77846164],
        [-3.10455859, 1.7959741],
        [-3.03914302, 1.81618995],
    ]
    converted_params = convert_params(np.array(samples))
    make_corner(converted_params, figname=cornername, save=True)
    assert os.path.isfile(cornername)
