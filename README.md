# barb
barb is a bayesian rate estimation python package for FRBs. <br />
Bayesian Rate-estimation for frBs (BaRB)<br />

## Installation
Install the rate estimation package.
```bash
git clone https://github.com/MorganWaddy/barb
cd barb
python setup.py install
```

## Usage
First, create a json file with your relevant data using [auto-json](https://github.com/MorganWaddy/auto-json/blob/master/README.md "README.md").
```bash
python bin/auto-json.py -N <name of primary author of the paper> -Y <year> -n <nFRBs> -S <sensitivity> -R <radius of the beam> -b <beams> -t <time per beam> -f <flux>
```

Second, group your json file(s) with the original dataset, located in the surveys directory. <br />

Third, run the program.
```bash
bin/run_analysis.py -D <name of the surveys> -c <number of cpus> -r <name of h5 file> -n <name of final plot> -m <maximum number of iterations>
```
_**For a more in-depth tutorial and explanation of the package's functions, please see the tutorial jupyter notebook.**_

## Requirements
* astropy
* numpy
* matplotlib
* tqdm
* emcee

## Citations
Please cite the following papers if you use barb. <br />
[Lawrence et al. (2017)](https://iopscience.iop.org/article/10.3847/1538-3881/aa844e/pdf)
