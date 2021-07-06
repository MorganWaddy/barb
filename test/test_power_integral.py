import pytest
import numpy as np

def power_integral(FWHM_2, beta):
    # references the cummulative rate for observed fluxes > 0 (from paper)
    # this returns the value for the integral of
    # [sensitivity^(beta)]d(sensitivity)
    return (FWHM_2 ** -(beta - 1)) / (beta - 1)
    # FWHM_2 is sigma (sensitivty), measured in janskys

def test_power_integral():
    assert power_integral(12, 1.5) == 0.5773502691896257
