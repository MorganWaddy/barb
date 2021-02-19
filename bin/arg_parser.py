import numpy as np
import pylab as plt
import matplotlib

matplotlib.use("Agg")
import math
from tqdm import tqdm
import emcee
import sys
import json
import logging
import argparse

parser = argparse.ArgumentParser(
    description="""Bayesian Rate-Estimation for FRBs (BaRB)
sample command: python barb.py -f -D <name of the surveys> -c <number of cpus> -s <name of npz file>

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
    nargs=1,
    action="store",
    required=True,
)
parser.add_argument(
    "-f",
    "--freq",
    action="store_true",
    help="to estimate spectral index limits use this flag",
    required=False,
)
args = parser.parse_args()

def read_in(surveys, nFRBs, FWHM_2, R, beams, tpb, flux, freq):       
# data feed in structure
        j = len(args.dat)
        k = j + 1
        filename = args.dat
        
        if j > 0:
            logging.info("Sanity Check:")
            logging.info("The number of user file(s) supplied is {0}".format(j) + "\n")
            logging.info("The supplied file(s) is/are {0}".format(filename) + "\n")
            for e in args.dat:
                with open(e, "r") as fobj:
                    info = json.load(fobj)
                    for p in info["properties"]:
                        surveys.append(p["surveys"])
                        nFRBs = np.append(nFRBs, p["nFRBs"])
                        FWHM_2 = np.append(FWHM_2, p["FWHM_2"])
                        R = np.append(R, p["R"])
                        beams = np.append(beams, p["beams"])
                        tpb = np.append(tpb, p["tpb"])
                        flux = np.append(flux, p["flux"])
                        if args.freq is True:
                            for e in args.dat:
                                with open(e, "r") as fobj:
                                    info = json.load(fobj)
                                    freq = np.append(freq, p["freq"])
                        else:
                            freq = np.append(freq, 1)
                            fobj.close()
            else:
                            logging.info("No data was supplied, please supply data on the command line!")
        nFRBs = np.array(nFRBs)
        FWHM_2 = np.array(FWHM_2)
        R = np.array(R)
        beams = np.array(beams)
        tpb = np.array(tpb)
        if args.freq is True:
            freq = np.array(freq)


        time = tpb * beams
        flux = np.array(flux)
