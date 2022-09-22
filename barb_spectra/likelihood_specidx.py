import numpy as np

# delete this
def area(R, gamma):
    """
    Calculate the beam shape

    Args:
        R (float): telescope beam radius
        gamma (float): Euclidean scaling that is valid for any population with a fixed luminosity distribution, as long as the luminosity does not evolve with redshift and the population has a uniform spatial distribution

    Returns:
        ar (float): shape of the beam
    """
    ar = (np.pi * R ** 2) / (gamma * np.log(2))
    beta = gamma + 1
    return ar

# delete this
def power_integral(sensitivity, beta):
    """
    Calculates the integral of [sensitivity^(beta)]d(sensitivity)

    Args:
        sensitivity (float): sensitivity at FWHM divided by 2 (measured in janskys)
        beta (float): Euclidean scaling (gamma + 1)

    Returns:
        (sensitivity ** -(beta - 1)) / (beta - 1) (float)
    """
    return (sensitivity ** -(beta - 1)) / (beta - 1)

def freq_term(freq, pow_term):
    """
    Defines the frequency term that is scaled by the spectral index

    Args:
        freq (float): central frequency (measured in MHz)
        beta (float): Euclidean scaling (gamma + 1)
    """
    return freq ** pow_term


def likelihood_list_specidx(vargroup, alpha, beta, spectr):
    """
    Analyzes all available data to return the likelihood that there will be an FRB

    Args:
        vargroup ([np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]): nFRBs, sensitivity, R, beams, tpb, flux, freq
            nFRBs: number of FRBs detected
            sensitivity: sensitivity at FWHM divided by 2 (measured in janskys)
            R: telescope radius
            beams: number of telescope beams
            tpb: time per beam
            flux: flux measurement of the FRB
            freq: central frequency (measured in MHz)
        alpha (float): product of reference rate and gamma
        beta (float): Euclidean scaling (gamma +1)
        spectr (float): spectral index

    Returns:
        likelihood_list (np.ndarray[float]): list of likelihoods that there will be an FRB
    """
    nFRBs, sensitivity, R, beams, tpb, flux, freq = vargroup
    powterm = 1 - (spectr * (beta - 1))
    freqterm = freq_term(freq, powterm)
    A = area(R, beta - 1)
    I = power_integral(sensitivity, beta)
    time = tpb * beams
    arr = time * A * (alpha / ((beta - 1) ** 2)) * (1 / (1 - (spectr * (beta - 1))))
    arr = np.array(arr)
    flux = np.array(flux)
    ll = 0
    if type(I) == float or np.shape(I) == ():
        for idx, nburst in enumerate(nFRBs):
            # idx is just a number that identifies a place in the array
            if flux[idx] == [-1]:
                val = -arr[idx] * I * freqterm
            else:
                val = (
                    -arr[idx] * I * freqterm
                    + nburst * (np.sum(np.log(alpha / (beta - 1)) - (spectr * gamma * np.log(freq)) - (beta * np.log(flux))))
                )
            ll += val
    else:
        for idx, nburst in enumerate(nFRBs):
            # idx is just a number that identifies a place in the array
            if flux[idx] == [-1]:
                val = -arr[idx] * I[idx] * freqterm
            else:
                val = (
                    -taa[idx] * I[idx] * freqterm
                    + nburst * (np.sum(np.log(alpha / (beta - 1)) - (spectr * gamma * np.log(freq)) - (beta * np.log(flux))))
                )
            ll += val
    return ll

def log_ll_specidx(varrest, nFRBs, sensitivity, R, beams, tpb, flux, freq):
    """
    Calculates the log of the result from likelihood_list

    Args:
        nFRBs: number of FRBs detected
        sensitivity: sensitivity at FWHM divided by 2 (measured in janskys)
        R: telescope radius
        beams: number of telescope beams
        tpb: time per beam
        flux: flux measurement of the FRB
        varrest (float, float, float): alpha, beta, spectr
            alpha (float): product of reference rate and gamma
            beta (float): Euclidean scaling (gamma +1)
            spectr (float): spectral index

    Returns:
        log_ll (np.ndarray[float]): log of the list of likelihoods that there will be an FRB
    """
    alpha, beta, spectr = varrest
    alpha = 10**alpha
    vargroup = nFRBs, sensitivity, R, beams, tpb, flux, freq
    if beta < 1:
        return -np.inf
        # returns a positive infinity multiplied by -1
        # (np.inf is a floating point constant value in the numpy library)
    return likelihood_list(vargroup, alpha=alpha, beta=beta, spectr=spectr)
