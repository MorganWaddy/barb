import numpy as np
import pylab as plt
import matplotlib

matplotlib.use("Agg")
import math
import sys
import argparse
import logging
import emcee

parser = argparse.ArgumentParser(
    description="""Bayesian Rate-Estimation for FRBs (BaRB)
sample command: python barb.py -D <name of the surveys> -c <number of cpus> -s <name of npz file>

Surveys on the original data set: Agarwal 2019, Masui 2015, Men 2019, Bhandari 2018, Golpayegani 2019, Oslowski 2019, GREENBURST, Bhandari 2019, Qiu 2019, Shannon 2018, Madison 2019, Lorimer 2007, Deneva 2009, Keane 2010, Siemion 2011, Burgay 2012, Petroff 2014, Spitler 2014, Burke-Spolaor 2014, Ravi 2015, Petroff 2015, Law 2015, Champion 2016""",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "-D",
    "--dat",
    help="supply the input data after this flag",
    action="store",
    nargs="+",
    required=True,
)
parser.add_argument(
    "-c",
    "--cpus",
    help="supply the number of cpus you want to be used",
    action="store",
    nargs="+",
    required=True,
)
parser.add_argument(
    "-s",
    "--allsamples",
    help="supply the name of the output numpy array of the samples",
    action="store",
    required=True,
)
args = parser.parse_args()

def plotting(allsamples):
    labels = [r"$\log \mathcal{R}$", r"$\alpha$"]

    np.savez("{0}".format(args.allsamples), allsamples)
    logging.info("the name of the np array file is {0}".format(args.allsamples) + ".npz")
    allsamples

    allsamples = np.load("{0}".format(args.allsamples) + ".npz")["arr_0"]

    quantile_val = 0.99

    import corner
    
    plt.figure(figsize=(15, 15))
    corner.corner(
        allsamples,
        labels=labels,
        quantiles=[(1 - 0.99) / 2, 0.5, 1 - (1 - 0.99) / 2],
        show_titles=True,
        bins=50,
        )
    
    # makes a corner plot displaying the projections of prob. distr. in space
    plt.savefig("rates_mc.png")
    plt.close("rates_mc.png")
    plt.clf()
