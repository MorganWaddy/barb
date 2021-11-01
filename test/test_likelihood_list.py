import pytest
import numpy as np
from test_area import area
from test_power_integral import power_integral
import sys

# sys.path.insert(0, "../barb/")
from barb.likelihood import likelihood_list

vargroup = [4], 12, 5, 13, [6], [9]


def test_likelihood_list():
    idx = 0
    nburst = 1
    assert likelihood_list(vargroup, 3, 7) == -8.213534499339293
