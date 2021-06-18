import pylab as plt
import numpy as np
import corner
import matplotlib
matplotlib.use("Agg")

def check_likelihood(bb, lk, save = False):
    # in the mcmc you maximize the likelihood, this is to check if the functions are working
    plt.plot(bb - 1, -1 * np.array(lk))
    plt.yscale("log")
    if save == True:
        plt.savefig("log_check.png")
        plt.close("log_check.png")
    plt.show()

def make_hist(data, save = False, output_name = 'hist_MCMC'):
    plt.hist(data)
    if save == True:
        plt.savefig(output_name)
        plt.close(output_name)

def make_corner(allsamples, figname = 'rates_mc.png', save = False):
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
        # makes a corner plot displaying the projections of prob. distr. in space
        plt.savefig(figname)
        plt.close(figname)
