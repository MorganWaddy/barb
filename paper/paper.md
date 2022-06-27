---
title: 'barb: Bayesian Rate Estimation for FRBs'
tags:
  - Python
  - astronomy
  - fast radio bursts
  - Bayesian statistics
  - MCMC
authors:
  - name: Morgan Waddy^[co-first author]
    orcid: 0000-0002-2699-2445
    affiliation: "1, 2"
  - name: Kshitij Aggarwal^[co-first author]
    orcid: 0000-0002-2059-0525
    affiliation: "1, 2"
  - name: Devansh Agarwal^[co-first author]
    orcid: 0000-0003-0385-491X
    affiliation: "1, 2"
  - name: Maura McLaughlin^[corresponding author]
    orcid: 0000-0001-7697-7422
    affiliation: "1, 2"
affiliations:
 - name: West Virginia University, Department of Physics and Astronomy, P. O. Box 6315, Morgantown 26506, WV, USA
   index: 1
 - name: Center for Gravitational Waves and Cosmology, West Virginia University, Chestnut Ridge Research Building, Morgantown 26506, WV, USA
   index: 2
date: 9 July 2021
bibliography: paper.bib

---

# Summary
Fast Radio Bursts (FRBs) are transient radio sources that were first discovered in pulsar surveys. These sources bear a resemblance to pulsar single pulses, but there are important differences. Only some FRBs repeat their signals, and the large, frequency-dependent (dispersive) delays of FRBs indicate that they originate outside of the Milky Way, and are therefore significantly more luminous than pulsars. Several studies have attempted to capture an all-sky rate of FRBs. @Lawrence:2017 found the all-sky FRB rate to be $\mathcal{R} = 587\substack{+924 \\ -272}$ events per sky per day above a flux density of 1 Jy. However, using more data and improving upon the math used in the @Lawrence:2017 paper by allowing for spectral index as a parameter in calculations, @Chawla:2017 found the rate to be $\mathcal{R} = 3.6 \times 10^3$ per sky per day above a peak flux density of 0.63 Jy. @Agarwal:2020 included more data in addition to the data used in the @Lawrence:2017 and @Chawla:2017 papers, and estimated the FRB all-sky rate to be $\mathcal{R} = 1140\substack{+200 \\ -180}$  events per sky per day above a flux density of 1 Jy. @Chawla:2017 placed a 95% confidence upper limit on the FRB rate, while the @Lawrence:2017 and @Agarwal:2020 estimates both have error quoted at 95% confidence. There is not an agreed-upon rate, partly due to there being no unified method to calculate this rate. In order to move forward in the knowledge of these rare events, we must create a unified methodology for rate calculation. 
Using the framework from @Lawrence:2017, `barb` employs Markov Chain Monte Carlo to estimate the rate of FRBs. For each input survey, the framework uses the results of FRB surveys: flux of the detected FRB(s), sensitivity at full-width half-maximum (FWHM) of the beam, radius of the beam, number of beams, and observation time to calculate an all-sky FRB rate at the frequency of the survey.
The study of FRBs is impacted by having low detection rates and observational bias. Other methods of estimating the rate of FRBs scale FRB detections by the product of the beam shape and the time on the sky; they assume a cumulative rate above a certain flux, usually do not account for flux measurements, and some model the telescope sensitivity as flat, which makes calculation easier but is physically unrealistic. The ``non-homogeneous Poisson process`` used herein is more robust than other methods of FRB rate estimation because it allows for the instantaneous rate to vary over multiple observations, and it addresses other non-ideal aspects of FRB rate estimation. This software provides a statistically sound method of rate estimation that is user-friendly for communal usage.


# Statement of need

`barb` (Bayesian Rate Estimation for FRBs) is a purely python-based package that uses non-homogeneous Poisson process, Markov Chain Monte Carlo (MCMC) methods to estimate an all-sky rate of FRB events. The package uses survey results from multiple FRB surveys [@Agarwal:2019; @2018MNRAS.475.1427B; @Bhandari:2019; @2012MNRAS.423.1351B; @Burke-Spolaor:2014; @Champion:2016; @Deneva:2009; @GREENBURST; @2019MNRAS.489.4001G; @2010MNRAS.401.1057K; @Law:2015; @Lorimer:2007; @Madison:2019; @Masui:2015; @2019MNRAS.489.3643M; @Shannon:2018; @Siemion:2011; @Oslowski:2019; @Spitler:2014; @Petroff:2014; @Petroff:2015; @Ravi:2015; @Qiu:2019] to generate a cumulative all-sky FRB rate. It can also be used to visualize the estimated rates. This package was designed in order to provide a user-friendly method of rate calculation for FRBs with user-input data, and so in addition to the initial data set, users can use the auto-json function in the package to create more data files and update the rate.

According to @Lawrence:2017, rates for astronomical events are usually calculated with only some of the available info when some part of the parameter space is non-ideal. This simplifies analysis, but should not be done when they are rare or subject to observational bias, for example FRBs. The non-homogeneous Poisson model assumes the instantaneous rate varies over the space of the observation (eg: a radio telescope has a varying sensitivity over its field of view). The program can accommodate both single-dish telescopes and interferometers, assumes all telescopes have a Gaussian beam shape, that FRB events have a uniform spatial distribution, and beam attenuation is radially symmetric. Future versions of the package will explore relaxing these assumptions. 

The basic structure of the rate calculation is $\Lambda = a(s)^{-b}$, where $a$ is the reference rate, $s$ is the flux, and $b$ is the source count index that scales flux. First `barb` takes in the data that is passed in through the command line and separates it into 7 NumPy arrays: $surveys$, $nFRBs$, $sensitivity$, $R$, $beams$, $tpb$, and $flux$. $Surveys$ contains the names of the surveys, $nFRBs$ contains the number of FRBs discovered in each survey, $sensitivity$ is the sensitivity at half-width at half-maximum, $R$ is the beam radius, $beams$ is the number of beams, and $tpb$ is time per beam. Barb then calculates beam shape, the power integral of $sensitivity$ and beta ($b+1$). $b$ is a Euclidean scaling factor that is valid for any population with a fixed luminosity distribution (as long as luminosity does not evolve with redshift). These values are then used to calculate the logarithmic likelihood of FRB detection with the formula. The program does extensive logging of all the steps carried out in the package. 

Then, `barb` uses MCMC to obtain the posterior distributions for rate and source count index. In the final steps, a corner plot is produced with the results of the MCMC analysis. With the non-homogeneous Poisson process, @Lawrence:2017 estimates an all-sky rate of FRB events a rate of $\mathcal{R} = 587\substack{+924 \\ -272}$ sky$^{-1}$day$^{-1}$ above a flux of 1 Jy. @Agarwal:2020 estimates the FRB all-sky rate to be $\mathcal{R} = 1140\substack{+200 \\ -180}$ sky$^{-1}$ day$^{-1}$. Our final estimate of the all-sky FRB rate was calculated to be $\mathcal{R} = 1148\substack{+201  \\ -171}$ sky$^{-1}$ day$^{-1}$, as displayed in \autoref{fig:corner}. All errors are quoted at 95% confidence.

![Corner plot displaying the one-dimensional projections of $\log(\mathcal{R})$ and $\alpha$, as well as a two-dimensional projection of the samples to show the covariances. The vertical dotted lines represent the quantiles that specify where a proportion of the distribution of data resides. The contours are 1$\sigma$, 2$\sigma$, 3$\sigma$, etc. \label{fig:corner}](rates_mc.png)

barb can be used by experienced researchers as well as students who do not have prior experience with calculating FRB rates or running MCMC simulations. The package is written fully in python and uses matplotlib [@Hunter:2007] for plotting, emcee [@emcee] for performing the MCMC operations, and additionally uses NumPy [@Numpy:2020], Astropy [@Astropy:2013], and SciPy [@2020SciPy-NMeth] for several of the functions.

# Acknowledgements

MW, KA, DA and MAM are members of the NANOGrav Physics Frontiers Center, supported by NSF award number award number 1430284. K. A. acknowledges support from NSF award numbers 1714897and 2108673.

# References
