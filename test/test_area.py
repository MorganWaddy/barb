import pytest
import numpy as np

def area(R, gamma):
    # beam shape
    # gamma is Euclidean scaling that is valid for any population with a fixed
    # luminosity distribution, as long as the luminosity does not evolve with
    # redshift and the population has a uniform spatial distribution
    ar = (np.pi * R ** 2) / (gamma * np.log(2))
    beta = gamma + 1
    return ar


def test_area():
    assert area(5, 7) == 16.187000506525692
