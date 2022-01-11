import pytest
import numpy as np
import sys

from barb.likelihood import area
from barb.likelihood import power_integral
from barb.likelihood import likelihood_list
from barb.likelihood import log_ll

vargroup = [4], 12, 5, 13, np.array([6]), [9]
vargroup = np.array(vargroup)


def test_log_likelihood_list():
    idx = 0
    nburst = 1
    assert log_ll([3, 7], [4], 12, 5, 13, np.array([6]), [9]) == -1.3369332333190052
