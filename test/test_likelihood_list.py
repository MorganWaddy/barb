import pytest
import numpy as np
from test_area import area
from test_power_integral import power_integral

def likelihood_list(vargroup, alpha, beta):
    # runs through all data to return the likelihood that there will be
    # an FRB
    # from appendix A: alpha = Rref*gamma
    nFRBs, FWHM_2, R, beams, tpb, flux = vargroup
    A = np.int_(area(R, beta - 1))
    I = [power_integral(FWHM_2, beta)]
    time = tpb*beams
    taa = np.int_(time * A * alpha)
    ll = 0
    for idx, nburst in enumerate(nFRBs):
        # idx is just a number that identifies a place in the array
        if flux[idx] == [-1]:
            val = -taa[idx] * I[idx]
        else:
            val = (
                -taa[idx] * I[idx]
                + nburst * np.log(taa[idx])
                - beta * np.sum(np.log(flux[idx]))
            )
        ll += val
    return ll

vargroup = [4], 12, 5, 13, [6], [9]

def test_likelihood_list():
    idx = 0
    nburst = 1
    assert likelihood_list(vargroup, 3, 7) == -8.213534499339293
