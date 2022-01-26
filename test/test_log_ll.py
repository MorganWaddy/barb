import pytest
import numpy as np
import sys

from barb.likelihood import area
from barb.likelihood import power_integral
from barb.likelihood import likelihood_list
from barb.likelihood import log_ll

vargroup = (
    [4],
    12,
    5,
    13,
    np.array([6, 6, 6, 6]),
    [9, 9, 9, 9],
)
vargroup = np.array(vargroup, dtype=object)


def log_ll(varrest, nFRBs, sensitivity, R, beams, tpb, flux):
    """
    Calculates the log of the result from likelihood_list

    Args:
        nFRBs: number of FRBs detected
        sensitivity: sensitivity at FWHM divided by 2 (measured in janskys)
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
    vargroup = nFRBs, sensitivity, R, beams, tpb, flux
    if beta < 1:
        return -np.inf
        # returns a positive infinity multiplied by -1
        # (np.inf is a floating point constant value in the numpy library)
    return likelihood_list(vargroup, alpha=alpha, beta=beta)


def test_log_likelihood_list():
    idx = 0
    nburst = 1
    assert (
        log_ll([3, 7], [4], 12, 5, 13, np.array([6, 6, 6, 6]), [9, 9, 9, 9])
        == 41.348502624892795
    )
