# barb
barb is a bayesian rate estimation python package for FRBs. <br />
Bayesian Rate-estimation for frBs (BaRB)<br />

## Installation
```bash
git clone https://github.com/MorganWaddy/barb
cd barb
python setup.py install
```

## Usage
First, create a json file with your relevant data using the json file-maker.
```bash
python auto-json.py -N <name> -Y <year> -n <nFRBs> -S <FWHM_2> -R <radius> -b <beams> -t <tpb> -f <flux>
```

Second, group your json file with the json data files. <br />

Third, run the program.
```bash
bin/run_analysis.py -D <name of the surveys> -c <number of cpus> -r <name of h5 file> -n <name of final plot> -m <maximum number of iterations>
```

## Requirements
* astropy
* numpy
* matplotlib
* tqdm
* emcee

## Citations
Please cite the following papers if you use barb. <br />
[Lawrence et al. (2017)](https://iopscience.iop.org/article/10.3847/1538-3881/aa844e/pdf)
