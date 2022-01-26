import pytest
from multiprocessing import Pool
from multiprocessing import cpu_count
import emcee
from tqdm import tqdm
import numpy as np
import json
import logging
import os

from barb.likelihood import area
from barb.likelihood import power_integral
from barb.likelihood import likelihood_list
from barb.likelihood import log_ll
from barb.mcmc import sampling
from barb.mcmc import read_samples

def test_read_samples():
    h5name = "test_MCMC_results.h5"
    assert os.path.isfile(h5name)
