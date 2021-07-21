import glob
import re
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("barb/__init__.py", "r") as f:
    vf = f.read()

version = re.search(r"^_*version_* = ['\"]([^'\"]*)['\"]", vf, re.M).group(1)

setup(
    name="barb",
    version=version,
    packages=["barb"],
    url="https://github.com/MorganWaddy/barb",
    author="Morgan Waddy",
    scripts=glob.glob("bin/*"),
    author_email="mdw4ux@virginia.edu",
    license="",
    long_description=long_description,
    description="Bayesian Rate Estimation for FRBs",
    install_requires=["astropy", "numpy", "matplotlib", "tqdm", "scipy", "emcee"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
