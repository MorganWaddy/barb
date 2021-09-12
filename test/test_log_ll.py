import pytest
import numpy as np
from test_area import area
from test_power_integral import power_integral
from test_likelihood_list import likelihood_list

def log_ll(varrest, nFRBs, FWHM_2, R, beams, tpb, flux):
    """
    Calculates the log of the result from likelihood_list

    Args:
        nFRBs: number of FRBs detected
        FWHM_2: full width at half-maximum divided by two
        R: telescope radius
        beams: number of telescope beams
        tpb: time per beam
        flux: flux measurement of the FRB
        varrest (float, float): alpha, beta
            alpha (float): product of reference rate and gamma
            beta (float): Euclidean scaling (gamma +1)

    Returns:
        log_ll (np.ndarray[float]): log of the list of likelihoods that there will be an FRB
    """
    alpha, beta = varrest
    alpha = 10 ** alpha
    vargroup = nFRBs, FWHM_2, R, beams, tpb, flux
    if beta < 1:
        return -np.inf
        # returns a positive infinity multiplied by -1
        # (np.inf is a floating point constant value in the numpy library)
    return likelihood_list(vargroup, alpha=alpha, beta=beta)

def test_log_likelihood_list():
    idx = 0
    nburst = 1
    assert log_ll([10 ** 3, 7], [4], 12, 5, 13, [6], [9]) == -1.3369332333190052
