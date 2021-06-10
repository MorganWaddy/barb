import numpy as np

def area(R, gamma):
    # beam shape
    # b is Euclidean scaling that is valid for any population with a fixed
    # luminosity distribution, as long as the luminosity does not evolve with
    # redshift and the population has a uniform spatial distribution
    ar = (np.pi * R ** 2) / (gamma * np.log(2))
    beta = gamma + 1
    return ar
# nothing should change for this function

def power_integral(FWHM_2, beta):
    # references the cumm. rate for observed fluxes > 0 (from paper)
    return (FWHM_2 ** -(beta - 1))
    # FWM_2 is sigma, measured in janskys


def likelihood_list(vargroup, alpha, beta):
    # runs through all data to return the likelihood that there will be
    # an FRB
    nFRBs, R, time, FWHM_2, flux = vargroup
    A = area(R, beta - 1)
    gamma = beta - 1
    Rref = alpha / gamma
    I = (power_integral(FWHM_2, beta))
    taa = time * A * alpha
    ll = 0
    bandw = 960 # MHz
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

def log_ll(varrest, nFRBs, R, time, FWHM_2, flux):
    alpha, beta = varrest
    vargroup = nFRBs, R, time, FWHM_2, flux
    alpha = 10 ** alpha
    if beta < 1:
        return -np.inf
    if alpha < 0 or alpha > 5:
        return -np.inf
    return likelihood_list(vargroup, alpha=alpha, beta=beta)
