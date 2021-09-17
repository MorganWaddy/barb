import pytest
import numpy as np
import sys
sys.path.insert(0, '../barb/')
from likelihood import power_integral


def test_power_integral():
    assert power_integral(12, 1.5) == 0.5773502691896257
