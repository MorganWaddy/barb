# this term is from the paper (Lawrence. et. al) - this function finds the effective area of the reciever
def area(radius, b):
    # b is Euclidean scaling that is valid for any population with a fixed
    # luminosity distribution, as long as the luminosity does not evolve with
    # redshift and the population has a uniform spatial distribution
    ar = np.pi * radius * radius / (b * np.log(2))
    return ar

def power_integral(sensitivity, beta):
    # references the cumm. rate for observed fluxes > 0 (from paper)
    return (sensitivity ** -(beta - 1)) / (beta - 1)
    # sensitivity is sigma, measured in janskys
    # from appendix A:
    # beta = b + 1


def likelihood_list(data, alpha, beta):
    # runs through all data to return the likelihood that there will be
    # an FRB
    # from appendix A:
    # alpha = ab
    nFRBs, radius, time, sensitivity, flux = data
    A = area(radius, beta - 1)
    I = power_integral(sensitivity, beta)
    taa = time * A * alpha
    ll = 0
    for idx, nburst in enumerate(nFRBs):
        # flux to the power of some gamma that is spec. idx
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


def cummlative_rate(flux, a, b):
    # returns the cummulative rate of events
    return a * flux ** (-b)
