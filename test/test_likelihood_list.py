import pytest
import numpy as np
from test_area import area
from test_power_integral import power_integral
import sys

from barb.likelihood import likelihood_list

vargroup = (
    [np.int64(4)],
    np.int64(12),
    np.int64(5),
    np.int64(13),
    np.array([6]),
    [np.int(9)],
)
vargroup = np.array(vargroup, "dtype=object")


def test_likelihood_list():
    idx = 0
    nburst = 1
    ll = likelihood_list(vargroup, 3, 7)
    assert ll == -8.213534499339293
