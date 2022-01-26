import pytest
import numpy as np
from test_area import area
from test_power_integral import power_integral
import sys

from barb.likelihood import likelihood_list

vargroup = (
    [4],
    12,
    5,
    13,
    np.array([6]),
    [9],
)
vargroup = np.array(vargroup, dtype=object)


def test_likelihood_list():
    idx = 0
    nburst = 1
    ll = likelihood_list(vargroup, 3, 7)
    assert ll == 18.19390241314793
