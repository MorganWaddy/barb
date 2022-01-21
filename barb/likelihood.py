import numpy as np


def area(R, gamma):
    """
    Calculate the beam shape

    Args:
        R (float): telescope radius
        gamma (float): Euclidean scaling that is valid for any population with a fixed luminosity distribution, as long as the luminosity does not evolve with redshift and the population has a uniform spatial distribution

    Returns:
        ar (float): shape of the beam
    """
    ar = (np.pi * R ** 2) / (gamma * np.log(2))
    return ar


def power_integral(sensitivity, beta):
    """
    Calculates the integral of [sensitivity^(beta)]d(sensitivity)

    Args:
        sensitivity (float): sensitivity at FWHM divided by 2 (measured in janskys)
        beta (float): Euclidean scaling (gamma +1)

    Returns:
        (sensitivity ** -(beta - 1)) / (beta - 1) (float)
    """
    return (sensitivity ** -(beta - 1)) / (beta - 1)


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
    if type(I) == float or np.shape(I) == ():
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
    else:
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
