#!/usr/bin/env python
import numpy as np
from barb.read_inputs import read_in
import glob
import argparse
from barb.plotting import make_corner
from barb.mcmc import sampling, convert_params, read_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Bayesian Rate-Estimation for FRBs (BaRB)
sample command: python run_analysis.py -D <name of the surveys> -c <number of cpus> -r <name of h5 file> -n <name of corner plot> -m <max iterations of mcmc>

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
        #    nargs=1,
        required=True,
    )
    parser.add_argument(
        "-r",
        "--results",
        help="supply the name of the h5 result file that will stored from the mcmc",
        #    nargs=1,
        action="store",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--cornername",
        help="supply the name of the corner plot producd from the mcmc results",
        #    nargs=1,
        action="store",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--max_n",
        #    nargs=1,
        action="store",
        help="the maximum number of itertions the mcmc sampler will run",
        required=True,
    )
    args = parser.parse_args()

    files = args.dat
    max_n = int(args.max_n)
    cornername = "{}".format(args.cornername)
    h5name = "{}".format(args.results)
    cpu_num = int(args.cpus)

    varrgroup = read_in(files)

    ndim, nwalkers = 2, 1200
    # walkers are copies of a system evolving towards a minimum
    ivar = np.array([np.log10(15), 2.5])
    # ivar is an intermediate variable for sampling
    p0 = ivar + 0.05 * np.random.uniform(size=(nwalkers, ndim))
    # returns a uniform random distribution mimicking the distribution of the data

    old_tau, sampler = sampling(
        p0, varrgroup, cpu_num, nwalkers, ndim, filename=h5name, max_n=max_n
    )

    samples = read_samples(h5name)
    converted_params = convert_params(samples)
    make_corner(converted_params, figname=cornername, save=True)
