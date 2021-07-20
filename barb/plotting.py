import pylab as plt
import numpy as np
import corner
import matplotlib

matplotlib.use("Agg")


def check_likelihood(bb, lk, save=False):
    """
    Checks to see if the functions are working

    Args:
        bb (float): returns numbers spaced evenly w.r.t interval. Similar to a range but instead of step it uses sample number
        lk (np.ndarray[float]): runs log likelihood function for all instances of b in bb
    """
    plt.plot(bb - 1, -1 * np.array(lk))
    plt.yscale("log")
    if save == True:
        plt.savefig("log_check.png")
        plt.close("log_check.png")
    plt.show()


def make_hist(data, save=False, output_name="hist_MCMC"):
    """
    Checks to see if the functions are working

    Args:
        data ([np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]): nFRBs, FWHM_2, R, beams, tpb, flux
            nFRBs: number of FRBs detected
            FWHM_2: full width at half-maximum divided by two
            R: telescope radius
            beams: number of telescope beams
            tpb: time per beam
            flux: flux measurement of the FRB
        output_name (str): name of output image
    """
    plt.hist(data)
    if save == True:
        plt.savefig(output_name)
        plt.close(output_name)


def make_corner(allsamples, figname="rates_mc.png", save=False):
    """
    Makes corner plot

    Args:
        data ([np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]): nFRBs, FWHM_2, R, beams, tpb, flux
            nFRBs: number of FRBs detected
            FWHM_2: full width at half-maximum divided by two
            R: telescope radius
            beams: number of telescope beams
            tpb: time per beam
            flux: flux measurement of the FRB
        figname (str): name of the output plot
    """
    labels = [r"$\log \mathcal{R}$", r"$\alpha$"]

    quantile_val = 0.99

    plt.figure(figsize=(15, 15))
    corner.corner(
        allsamples,
        labels=labels,
        quantiles=[(1 - 0.99) / 2, 0.5, 1 - (1 - 0.99) / 2],
        show_titles=True,
        bins=50,
    )
    if save == True:
        # makes a corner plot displaying the projections of
        # probablity distrbution in space
        plt.savefig(figname)
        plt.close(figname)
