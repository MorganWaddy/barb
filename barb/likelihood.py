import numpy as np


def area(R, gamma):
    # beam shape
    # gamma is Euclidean scaling that is valid for any population with a fixed
    # luminosity distribution, as long as the luminosity does not evolve with
    # redshift and the population has a uniform spatial distribution
    ar = (np.pi * R ** 2) / (gamma * np.log(2))
    beta = gamma + 1
    return ar


def power_integral(FWHM_2, beta):
    # references the cummulative rate for observed fluxes > 0 (from paper)
    # this returns the value for the integral of
    # [sensitivity^(beta)]d(sensitivity)
    return (FWHM_2 ** -(beta - 1)) / (beta - 1)
    # FWHM_2 is sigma (sensitivty), measured in janskys


def likelihood_list(vargroup, alpha, beta):
    # runs through all data to return the likelihood that there will be
    # an FRB
    # from appendix A: alpha = Rref*gamma
    nFRBs, R, time, FWHM_2, flux = vargroup
    A = area(R, beta - 1)
    I = power_integral(FWHM_2, beta)
    taa = time * A * alpha
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


def log_ll(varrest):
    alpha, beta = varrest
    alpha = 10 ** alpha
    if beta < 1:
        return -np.inf
        # returns a positive infinity multiplied by -1
        # (np.inf is a floating point constant value in the numpy library)
    if alpha < 0 or alpha > 5:
        return -np.inf
    return likelihood_list(vargroup, alpha=alpha, beta=beta)
