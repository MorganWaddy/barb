#!/usr/bin/env python
import numpy as np
import glob
import argparse
import logging
import sys
import os

from barb.read_inputs import read_in
from barb_spectra.read_inputs_specidx import read_in_specidx
from barb.plotting import make_corner
from barb.mcmc import sampling, convert_params, read_samples
from barb_spectra.mcmc_specidx import sampling_specidx, convert_params_specidx, read_samples_specidx

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Bayesian Rate-Estimation for FRBs (BaRB)
sample command: python run_analysis.py -D <name of the surveys> -c <number of cpus> -r <name of h5 file> -n <name of corner plot> -m <max iterations of mcmc> -s

Surveys on the original data set: Agarwal 2019, Masui 2015, Men 2019, Bhandari 2018, Golpayegani 2019, Oslowski 2019, GREENBURST, Bhandari 2019, Qiu 2019, Shannon 2018, Madison 2019, Lorimer 2007, Deneva 2009, Keane 2010, Siemion 2011, Burgay 2012, Petroff 2014, Spitler 2014, Burke-Spolaor 2014, Ravi 2015, Petroff 2015, Law 2015, Champion 2016""",
        formatter_class=argparse.RawTextHelpFormatter,
        prog="run_analysis.py",
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
        required=True,
    )
    parser.add_argument(
        "-r",
        "--results",
        help="supply the name of the h5 result file that will stored from the mcmc",
        action="store",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--cornername",
        help="supply the name of the corner plot producd from the mcmc results",
        action="store",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--max_n",
        action="store",
        help="the maximum number of itertions the mcmc sampler will run",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--spectralidx",
        action="store_true",
        help="to estimate spectral index limits use this flag",
        required=False,
    )
    args = parser.parse_args()

    # a logging file that tracks information going into the program
    logging.basicConfig(filename="FRB-rate-calc.log", level=logging.INFO)
    logging.info("The logging file was created" + "\n")

    logging.info("Input Arguments:-")
    for arg, value in sorted(vars(args).items()):
        logging.info("%s: %r", arg, value)

    files = args.dat
    max_n = int(args.max_n)
    cornername = "{}".format(args.cornername)
    h5name = "{}".format(args.results)
    cpu_num = int(args.cpus)
    specidx_analysis = args.spectralidx
    if args.spectralidx == True:
        specidx_analysis == True
    else:
        specidx_analysis == False
    

    # global varrgroup
    if specidx_analysis == True:
        varrgroup = read_in_specidx(files)
    if specidx_analysis == False:
        varrgroup = read_in(files)
    logging.info("Data Provided: {0}".format(varrgroup) + "\n")

    ndim, nwalkers = 2, 1200
    # walkers are copies of a system evolving towards a minimum
    ivar = np.array([np.log10(15), 2.5])
    # ivar is an intermediate variable for sampling
    p0 = ivar + 0.05 * np.random.uniform(size=(nwalkers, ndim))
    # returns a uniform random distribution mimicking the distribution
    # of the data

    logging.info("{0} CPUs".format(cpu_num))

    if specidx_analysis == True:
        old_tau = sampling_specidx(
            p0, varrgroup, cpu_num, nwalkers, ndim, filename=h5name, max_n=max_n
        )
        logging.info("Tau from the Sampler: {0}".format(old_tau) + "\n")
        samples = read_samples_specidx(h5name)
        converted_params = convert_params_specidx(samples)
        
    if specidx_analysis == False:
        old_tau = sampling(
            p0, varrgroup, cpu_num, nwalkers, ndim, filename=h5name, max_n=max_n
        )
        logging.info("Tau from the Sampler: {0}".format(old_tau) + "\n")
        samples = read_samples(h5name)
        converted_params = convert_params(samples)

    make_corner(converted_params, figname=cornername, save=True)
