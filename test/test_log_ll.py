import pytest
import numpy as np
from test_area import area
from test_power_integral import power_integral
from test_likelihood_list import likelihood_list
import sys
sys.path.insert(0, '../barb/')
from likelihood import log_ll

vargroup = [4], 12, 5, 13, [6], [9]

def test_log_likelihood_list():
    idx = 0
    nburst = 1
    assert log_ll([10 ** 3, 7], [4], 12, 5, 13, [6], [9]) == -1.3369332333190052
