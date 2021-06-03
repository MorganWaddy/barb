import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Bayesian Rate-Estimation for FRBs (BaRB)
sample command: python barb.py -f -D <name of the surveys> -c <number of cpus> -s <name of npz file>

Surveys on the original data set: Agarwal 2019, Masui 2015, Men 2019, Bhandari 2018, Golpayegani 2019, Oslowski 2019, GREENBURST, Bhandari 2019, Qiu 2019, Shannon 2018, Madison 2019, Lorimer 2007, Deneva 2009, Keane 2010, Siemion 2011, Burgay 2012, Petroff 2014, Spitler 2014, Burke-Spolaor 2014, Ravi 2015, Petroff 2015, Law 2015, Champion 2016""",
        formatter_class=argparse.RawTextHelpFormatter,
        prog="barb_specidx_test.py",
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
