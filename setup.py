from setuptools import setup
setup(
    name='barb',
    version='0.1',
    packages=['barb'],
    url='https://github.com/MorganWaddy/barb',
    author='Morgan Waddy',
    author_email='mdw4ux@virginia.edu',
    license='',
    description='Bayesian Rate Estimation for FRBs',
    install_requires=['astropy', 'numpy', 'matplotlib', 'tqdm', 'scipy'],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "License :: ? :: ?",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy"],
)