# barb
barb is a bayesian rate estimation python package for FRBs.
Bayesian Rate-estimation for frBs (BaRB)

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

Second, group your json file with the json data files.

Third, run the program.
```bash
barb.py -D <name of the surveys> -c <number of cpus> -s <name of npz file>
```

## Requirements
*astropy
*numpy
*matplotlib
*tqdm
*scipy

## Citations
Please site the following papers if you use barb.
[Lawrence et al. (2017)](https://iopscience.iop.org/article/10.3847/1538-3881/aa844e/pdf)