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

def likelihood_list(vargroup, alpha, beta):
    """
    Analyzes all available data to return the likelihood that there will be an FRB

    Args:
        vargroup ([np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]): nFRBs, sensitivity, R, beams, tpb, flux
            nFRBs: number of FRBs detected
            sensitivity: sensitivity at FWHM divided by 2 (measured in janskys)
            R: telescope radius
            beams: number of telescope beams
            tpb: time per beam
            flux: flux measurement of the FRB
        alpha (float): product of reference rate and gamma
        beta (float): Euclidean scaling (gamma +1)

    Returns:
        likelihood_list (np.ndarray[float]): list of likelihoods that there will be an FRB
    """
    nFRBs, sensitivity, R, beams, tpb, flux = vargroup
    A = area(R, beta - 1)
    I = power_integral(sensitivity, beta)
    time = tpb * beams
    taa = time * A * alpha
    taa = np.array(taa)
    flux = np.array(flux)
    ll = 0
    for idx, nburst in enumerate(nFRBs):
        # idx is just a number that identifies a place in the array
        if flux[idx] == [-1]:
            val = -taa[idx] * I
        else:
            val = (
                -taa[idx] * I
                + nburst * np.log(taa[idx])
                - beta * np.sum(np.log(flux[idx]))
            )
        ll += val
    return ll

def test_log_likelihood_list():
    idx = 0
    nburst = 1
    assert log_ll([3, 7], [4], 12, 5, 13, np.array([6]), [9]) == -1.3369332333190052
