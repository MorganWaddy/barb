import pytest
import numpy as np
import sys

sys.path.insert(0, "../barb/")
from likelihood import area


def test_area():
    assert area(5, 7) == 16.187000506525692
