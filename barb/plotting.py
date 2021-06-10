import matplotlib.pyplot as plt
import numpy as np

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
