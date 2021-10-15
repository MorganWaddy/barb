import pytest
import numpy as np
import sys

sys.path.insert(0, "../barb/")
from likelihood import area
from likelihood import power_integral
from likelihood import likelihood_list
from likelihood import log_ll

vargroup = [4], 12, 5, 13, [6], [9]


def test_log_likelihood_list():
    idx = 0
    nburst = 1
    assert log_ll([10 ** 3, 7], [4], 12, 5, 13, [6], [9]) == -1.3369332333190052
