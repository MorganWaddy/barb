import glob

from setuptools import setup

setup(
    name='barb',
    version='0.1',
    packages=['barb'],
    url='https://github.com/MorganWaddy/barb',
    author='Morgan Waddy',
    scripts=glob.glob("bin/*"),
    author_email='mdw4ux@virginia.edu',
    license='',
    description='Bayesian Rate Estimation for FRBs',
    install_requires=[
        'astropy', 
        'numpy', 
        'matplotlib', 
        'tqdm', 
        'scipy', 
        'emcee'],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy"],
)
