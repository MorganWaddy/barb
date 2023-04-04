import numpy as np

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

def beam_atten_flux(flux, sensitivity, beta):
    """
    Calculates the flux with beam attenuation

    Args:
        sensitivity (float): sensitivity at FWHM divided by 2 (measured in janskys)
        beta (float): Euclidean scaling (gamma + 1)

    Returns:
        fluxat (float): flux divided sensitivity at HWHM attenuated by the scaling factor,
        accounting for the telescope's beam attenuation
    """
    fluxat = (flux / sensitivity) ** (1 - beta)
    return fluxat

def freq_term(freq, bw, spectr, gamma):
    """
    Defines the frequency term that is scaled by the spectral index

    Args:
        freq (float): central frequency (measured in MHz)
        beta (float): Euclidean scaling (gamma + 1)
        bw (float): bandwidth (measured in MHz)
    Returns:
        freqle (float): the frequency term scaled by the spectral index
    """
    freqle = (bw / freq) ** (1 - (spectr * (gamma)))
    beta = gamma + 1
    return freqle

def ref_rate(time, alpha, beta, spectr):
    """
    Calculates the first set of terms coming from the integral over flux and frequency,
    including the reference rate.
    
    Args:
        time (float): the time per beam multiplied by the number of beams
        alpha (float): product of reference rate and gamma
        beta (float): Euclidean scaling (gamma + 1)
        spectr (float): spectral index term

    Returns:
    """
    ref = ((-time) * (alpha / (beta - 1))) / (1 - (spectr * (beta - 1)))
    return ref

def likelihood_list_specidx(vargroup, alpha, beta, spectr):
    """
    Analyzes all available data to return the likelihood that there will be an FRB

    Args:
        vargroup ([np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]): nFRBs, sensitivity, R, beams, tpb, flux, freq, bw
            nFRBs: number of FRBs detected
            sensitivity: sensitivity at FWHM divided by 2 (measured in janskys)
            R: telescope radius
            beams: number of telescope beams
            tpb: time per beam
            flux: flux measurement of the FRB
            freq: central frequency (measured in MHz)
            bw (float): bandwidth (measured in MHz)
        alpha (float): product of reference rate and gamma
        beta (float): Euclidean scaling (gamma +1)
        spectr (float): spectral index

    Returns:
        likelihood_list (np.ndarray[float]): list of likelihoods that there will be an FRB
    """
    nFRBs, sensitivity, R, beams, tpb, flux, freq, bw = vargroup
    freqterm = freq_term(freq, bw, spectr, beta)
    flux = np.array(flux)
    A = area(R, beta - 1)
    atten_flux = beam_atten_flux(flux, sensitivity, beta)
    time = tpb * beams
    r_ref = ref_rate(time, alpha, beta, spectr)
    r_ref = np.array(r_ref)
    ll = 0

    """
    if type(atten_flux) == float or np.shape(atten_flux) == ():
        for idx, nburst in enumerate(nFRBs):
            if flux[idx] == [-1]:
                val = r_ref[idx] * A[idx] * (sensitivity ** (1 - beta)) * freqterm[idx]

            else:
                val = (
                    r_ref[idx] * A[idx] * atten_flux[idx] * freqterm[idx]
                    + nburst * (np.log((np.pi * R ** 2) / np.log(2))
                                - ((spectr * (beta -1)) * (np.log(bw[idx]) - np.log(freq[idx]))))
                    - ((beta - 1) * (np.sum(np.log(flux[idx]) - np.log(sensitivity[idx]))))
                )
            ll += val
    else:
    """

    for idx, nburst in enumerate(nFRBs):
        if flux[idx] == [-1]:
            val = r_ref[idx] * A[idx] * (sensitivity ** (1 - beta)) * freqterm[idx]
            
        else:
            val = (
                r_ref[idx] * A[idx] * atten_flux[idx] * freqterm[idx]
                + nburst * (np.log((np.pi * R ** 2) / np.log(2))
                            - ((spectr * (beta -1)) * (np.log(bw[idx]) - np.log(freq[idx]))))
                - ((beta - 1) * (np.sum(np.log(flux[idx]) - np.log(sensitivity[idx]))))
            )
            ll += val
        return ll

def log_ll_specidx(varrest, nFRBs, sensitivity, R, beams, tpb, flux, freq, bw):
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
    vargroup = nFRBs, sensitivity, R, beams, tpb, flux, freq, bw
    if beta < 1:
        return -np.inf
        # returns a positive infinity multiplied by -1
        # (np.inf is a floating point constant value in the numpy library)
    return likelihood_list_specidx(vargroup, alpha=alpha, beta=beta, spectr=spectr)
