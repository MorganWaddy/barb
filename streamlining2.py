# this is the second take at streamlining the code
import numpy as np
import pylab as plt
import matplotlib
matplotlib.use('Agg')
import math
from tqdm import tqdm
import emcee
import sys
import json


# this is feeding in all the previous data
surveys = ['Lorimer 2007', 'Deneva 2009', 'Keane 2010', 'Siemion 2011', 'Burgay 2012', 
             'Petroff 2014', 'Spitler 2014', 'Burke-Spolaor 2014',
             'Ravi 2015', 'Petroff 2015', 'Law 2015', 'Champion 2016']

n = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 10])
# Sensitivity at FWHM divided by 2
S =np.array([0.590, 0.365, 0.447, 118, 0.529, 0.868, 0.210, 0.615, 0.555, 0.555, 0.240, 0.560]) / 2
# FWHM diameter in arcminutes divided by 2 to get radius divide by 60 to get degrees
R = np.array([14, 4, 14, 150, 14, 14, 4, 14, 14, 14, 60, 14])/(2*60)
# Number of beams.
beams =np.array([13, 7, 13, 30, 13, 13, 7, 13, 13, 13, 1, 13])
# Time per beam
tpb =np.array([490.5, 459.1, 1557.5, 135.8, 532.6, 926, 11503.3, 917.3, 68.4, 85.5, 166, 2786.5])

# observed flux
flux = [[30], # Lorimer 2007
        [None],
        [None],
        [None],
        [None],
        [None],
          [0.4], 	# Spitler 2014
          [0.3],	# Burke-Spolaor 2014
          [2.1],	# Ravi et al. (estimated from Figure 3, 1432 MHz)
          [0.47], # Petroff 2015
        [None],
          [1.3, 0.4, 0.5, 0.5, # Chamption 2016 (these values are actually from Thornton 2013)
          0.87, 0.42, 0.69, 1.67, 0.18, # Champion 2016 (estimated from Figure 1)
          2.2]]  # The last Champion entry that Scott found somewhere (FRB Cat?)

# Zhang et al. 2019: A new fast radio burst in the datasets containing the Lorimer burst
n[0] = 3
flux[0].append(0.25)
# Zhang et al 2020: Parkes transient events: I. Database of single pulses, initial results and missing FRBs
flux[0].append(0.42)
tpb[0] += 250

# Petroff et al. 2018: A fast radio burst with a low dispersion measure
n[-1] = 11
flux[-1].append(27)

# Shannon et al. 2018 20 ASKAP FRBs
surveys.append("Shannon 2018")
n=np.append(n,20)
S=np.append(S,24.6)
R=np.append(R, 54/2/60)
beams = np.append(beams, 36)
tpb = np.append(tpb, 1700)
flux.append([58/2.4, 97/5.0, 34/4.4, 52/3.5, 74/2.5, 81/2.0, 219/5.4, 200/1.7, 63/2.3, 133/1.5, 40/1.9, 420/3.2,
        110/2.7, 51/2.9, 66/2.3, 95/4.1, 100/4.5, 96/1.81])

# Bhandari et al. 2019 ASKAP FRB
surveys.append("Bhandari 2019")
n=np.append(n,1)
S=np.append(S,24.6)
R=np.append(R, 54/2/60)
beams = np.append(beams, 36)
tpb = np.append(tpb, 1286)
flux.append([46/1.9])

# Qiu et al. 2019 ASKAP FRB
surveys.append("Qiu 2019")
n=np.append(n,1)
S=np.append(S,24.6)
R=np.append(R, 54/2/60)
beams = np.append(beams, 36)
tpb = np.append(tpb, 963.89)
flux.append([177])

# Agarwal et al. 2019 ASKAP FRB
surveys.append("Agarwal 2019")
n=np.append(n,1)
S=np.append(S,24.6/np.sqrt(7))
R=np.append(R, 54/2/60)
beams = np.append(beams, 36)
tpb = np.append(tpb, 300)
flux.append([22])

# Bhandari et al. 2018 Superb
surveys.append("Bhandari 2018")
n=np.append(n,4)
S=np.append(S,0.560/2)
R=np.append(R, 14/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 2722)
flux.append([0.7, 0.3, 0.43, 0.5])

# Oslowski et al. 2019 PPTA
surveys.append("Oslowski 2019")
n=np.append(n,4)
S=np.append(S,0.560/2)
R=np.append(R, 14/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 659.5)
flux.append([1.2,23.5, 0.15, 0.6])

# Masui et al 2015 GBT
surveys.append("Masui 2015")
n=np.append(n,1)
S=np.append(S,0.27)
R=np.append(R, 15/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 400)
flux.append([0.6])

# Non detections

# Men et al. 2019
surveys.append("Men (Arecibo) 2019")
n=np.append(n,0)
S=np.append(S,0.021)
R=np.append(R, 3.5/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, (340.7+448.8)/60)
flux.append([None])

surveys.append("Men (GBT) 2019")
n=np.append(n,0)
S=np.append(S,0.087)
R=np.append(R, 9/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, (70.6 + 82.5 + 131.3 + 76.5)/60)
flux.append([None])

# Madison et al. 2019
surveys.append("Madison (Arecibo) 2019")
n=np.append(n,0)
S=np.append(S,0.021)
R=np.append(R, 3.5/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 20)
flux.append([None])

surveys.append("Madison (GBT) 2019")
n=np.append(n,0)
S=np.append(S,0.087)
R=np.append(R, 9/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 60)
flux.append([None])

# GBTrans
surveys.append("Golpayegani 2019")
n=np.append(n,0)
S=np.append(S,6*1.26)
R=np.append(R, 48/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 503*24)
flux.append([None])

# GREENBURST

surveys.append("This work (L-band)")
n=np.append(n,0)
S=np.append(S,0.14*1.2)
R=np.append(R, 9/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 2194)
flux.append([None])

surveys.append("This work (X-band)")
n=np.append(n,0)
S=np.append(S,0.89*1.2)
R=np.append(R, 9/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 615)
flux.append([None])

surveys.append("This work (C-band)")
n=np.append(n,0)
S=np.append(S,0.25*1.2)
R=np.append(R, 9/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 556)
flux.append([None])

surveys.append("This work (Ku-band)")
n=np.append(n,0)
S=np.append(S,0.8*1.2)
R=np.append(R, 9/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 210)
flux.append([None])

surveys.append("This work (Mustang)")
n=np.append(n,0)
S=np.append(S,0.26*1.2)
R=np.append(R, 9/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 181)
flux.append([None])

# this is for the user to feed in their own data
j = len(sys.argv[1:])
k = j+1
filename = sys.argv[1:k]
print("Sanity Check:")
print("The number of user file(s) supplied is", j)
print("The name of the user file(s) supplied is/are", filename)

# print("\nThe User Data Supplied:")
# this can take multiple datasets in the same file, or multiple files
if j > 0:
    for e in sys.argv[1:]:
        with open(e, "r") as fobj:
            info = json.load(fobj)
            for p in info['properties']:
                surveys.append(p['surveys'])
                n = np.append(n,p['n'])
                S = np.append(S,p['S'])
                R = np.append(R,p['R'])
                beams = np.append(beams,p['beams'])
                tpb = np.append(tpb,p['tpb'])
                flux.append(p['flux'])
        fobj.close()
# the usual way the R value is written it cannot be parsed by JSON
# make sure you do the ?/2/60 calculation b4 inputting the data


time = tpb*beams

# this term is from the paper - this function finds the effective area of the 
# reciever
def area(radius, b):
    # b is Euclidean scaling that is valid for any population with a fixed 
    # luminosity distribution, as long as the luminosity does not evolve with 
    # redshift and the population has a uniform spatial distribution
    ar=np.pi*radius*radius/(b*np.log(2))
    return ar

# power rule for integrals
# returns the value for the integral of [sensitivity^(beta)]d(sensitivity)
def power_integral(sensitivity, beta):
    # references the cumm. rate for observed fluxes > 0 (from paper)
    return (sensitivity**-(beta-1)) / (beta-1)
    # sensitivity is sigma, measured in janskys
    # from appendix A:
    # beta = b + 1

def likelihood_list(data, alpha, beta):
    # runs through all data to return the likelihood that there will be 
    # an FRB
    # from appendix A:
    # alpha = ab
    n, radius, time, sensitivity, flux = data
    A = area(radius, beta-1)
    I = power_integral(sensitivity, beta)
    taa = time*A*alpha
    ll = 0
    for idx, nburst in enumerate(n):
        # idx is just a number that identifies a place in the array
        if flux[idx] == [None]:
            val=-taa[idx]*I[idx]
        else:
            val=-taa[idx]*I[idx] + nburst*np.log(taa[idx]) -beta*np.sum(np.log(flux[idx]))
        ll+=val
    return ll

global data
data = n, R, time, S, flux

# defining a file to put the data arrays
with open('input_array.txt', 'w') as dat:
    dat.write(str(data)+"\n")
    dat.close()

def log_ll(junk):
    alpha, beta = junk
    alpha = 10**alpha
    if (beta < 1):
        return -np.inf
    return likelihood_list(data, alpha=alpha, beta=beta)

# returns results of logarithmic likelihood function
log_ll([0.97504624, 1.91163861])
# log_ll([alpha,beta])
# this is supplying the alpha and beta for the calculations
# defining a file that houses the results of the functions
with open('calculations.out', 'a') as calc:
    logly = log_ll([0.97504624, 1.91163861])
    calc.write("log_ll([0.97504624, 1.91163861]) = "+str(logly)+"\n\n")
    calc.close()

bb = np.linspace(0.1,3,100)
# bb supplies 100 evenly spaced numbers over the interval
lk = [log_ll([np.log10(586.88/(24 * 41253)), b]) for b in bb]
# lk performs the log_ll calculation on the bb array

# returns the indices of the minimum values of bb
bb[np.argmin(lk)]

# this plots on a log scale the array of values from lk multiplied by -1 vs the bb array - 1
# in the mcmc you maximize the likelihood, this is to check if the functions are working
plt.plot(bb-1,-1*np.array(lk))
plt.yscale('log')
# saving this to view later
plt.savefig('log_check.png')
plt.close('log_check.png')
# these plots will help us see if the functions are working 


ndim, nwalkers = 2, 1200
# walkers are copies of a system evolving towards a minimum
ivar = np.array([np.log10(15),2.5])
print("ivar is ", ivar)
# ivar is an intermediate variable
with open('calculations.out', 'a') as calc:
    calc.writelines("ivar is "+ str(ivar)+"\n\n")
    calc.close()


p0 = ivar + 0.05* np.random.uniform(size=(nwalkers, ndim))
# this returns a uniform random distribution mimicking the distribution of 
# the data
plt.hist(p0[:,0])
plt.savefig('MCMC_hist1.png')
plt.close('MCMC_hist1.png')
plt.hist(p0[:,1])
plt.savefig('MCMC_hist2.png')
plt.close('MCMC_hist2.png')
# these plots visualize that distribution

from multiprocessing import cpu_count
ncpu = cpu_count() - 5#//2
print("{0} CPUs".format(ncpu))
from multiprocessing import Pool
# gettign ready to run the heavy-duty jobs so this counts the cpus

pool = Pool(ncpu)
# pool paralelizes the execution of the functions over the cpus
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_ll, pool = pool)
# emcee.EnsembleSampler is the main interface offered by emcee
# basically we get the logarithmic probability, then define the nwalkers and 
# ndim, then we use this to sample from our probability distribution

def cummlative_rate(flux,a,b):
    # returns the cummulative rate of events
    return a*flux**(-b)

max_n = 100000
# the algorithim needs  a certain number of steps before it can forget where
# it started, so this max is to make sure the algorithm has enough space, but
# makes sure it doesn't go on for forever if stuff goes wrong

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to max_n steps
for sample in sampler.sample(p0, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau

pool.close()
# this function samples until it converges at a estimate of rate for the events
# the estimate may or may not be trustworthy


tau = sampler.get_autocorr_time()
# this computes an estimate of the autocorrelation time for each parameter
burnin = int(2*np.max(tau))
# these are steps that should be discarded
thin = int(0.5*np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
# gets the stored chain of MCMC samples
# flatten the chain across ensemble, take only thin steps, discard burn-in
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
# gets the chain of log probabilities evaluated at the MCMC samples
# discard burn-in steps, flatten chain across ensemble, take only thin steps

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))


all_samples = samples
# an array of the stored chain of MCMC samples
all_samples[:,0]=np.log10((24 * 41253)*(10**all_samples[:,0])/(all_samples[:,1]-1))
# not sure
all_samples[:,1]-=1
# not sure

labels = [r"$\log \mathcal{R}$",r"$\alpha$"]

np.savez('all_samples_may_12_snr_12',all_samples)
all_samples
# this saves the array

all_samples = np.load('all_samples_may_12_snr_12.npz')['arr_0']
# this loads the saved array

quantile_val = 0.99

import corner

# all_samples[:,0] = (all_samples[:,0]).astype(np.int)
# all_samples[:,0] = (10**all_samples[:,0]).astype(np.int)
# removing this statement completely fixed the problem where the program plotted R instead of log(R)

plt.figure(figsize=(15,15))
corner.corner(all_samples, labels=labels, quantiles=[(1-0.99)/2,0.5,1-(1-0.99)/2],show_titles=True, bins=50)
# makes a corner plot displaying the projections of out prob. distr. in space
plt.savefig('rates_mc.png')
plt.close('rates_mc.png')



fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.flatchain
# accesses chain flattened along the zeroth (walker) axis
labels = ["a", "b"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:nwalkers*burnin, i], "k", alpha=0.3)
    ax.set_ylabel(labels[i])

axes[-1].set_xlabel("step number");


fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.flatchain
# accesses chain flattened along the zeroth (walker) axis
labels = ["a", "b", ]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, i], "k", alpha=0.3)
    ax.set_xlim(0, len(all_samples))
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");



quantile_val = 0.01
flux = np.logspace(-2,2,100)
plt.figure(figsize=(10,6))
total_label=[]
plt.plot(flux,cummlative_rate(flux,10**np.mean(all_samples[:,0]),np.mean(all_samples[:,1])),'k')
total_label.append("Best fit")
plt.plot(flux,cummlative_rate(flux,10**np.quantile(all_samples,quantile_val,axis=0)[0]
                                        ,np.quantile(all_samples,quantile_val,axis=0)[1]),'k--')
total_label.append(r"3$\sigma$ Error")

plt.xlabel('Flux Limit (Jy)')
plt.ylabel(r'Rate (day$^{-1} sky^{-1}$)')
plt.xscale('log')
plt.yscale('log')

num_plots=len(surveys)
my_colors = plt.rcParams['axes.prop_cycle']() 

plt.errorbar(26, 37, yerr=8,fmt='.')
total_label.append("Shannon 2018")

plt.errorbar(0.560/2, 1700, yerr=np.array([1700-800,3200-1700]).reshape(2,1),fmt='.')
total_label.append("Bhandari 2018")

plt.errorbar(0.560/2, 10000, yerr=np.array([5000,6000]).reshape(2,1),fmt='.')
total_label.append("Thronton 2013")

plt.errorbar(1.0, 587, yerr=np.array([587-272,924-587]).reshape(2,1),fmt='.')
total_label.append("Lawrence 2017")

plt.errorbar(0.560/2, 3300, yerr=np.array([3300-1100,7000-3300]).reshape(2,1),fmt='.')
total_label.append("Crawford 2016")

plt.errorbar(0.590/2, 7000, yerr=np.array([1000,12000-7000]).reshape(2,1),fmt='.')
total_label.append("Champion 2016")

plt.errorbar(0.590/2, 4400,yerr=np.array([4400-1300,9600-4400]).reshape(2,1),fmt='.')
total_label.append("Rane 2016")

plt.errorbar(0.590/2, 225,fmt='.')
total_label.append("Lorimer 2007")

plt.errorbar(1, 5e3 ,fmt='.')
total_label.append("Masui 2015")

# insert a way to put the user data in the plot

plt.grid()
plt.legend(total_label, ncol=5, loc='lower center', 
           bbox_to_anchor=[0.5, 1.1], 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)

plt.plot(flux,cummlative_rate(flux,10**np.quantile(all_samples,1-quantile_val,axis=0)[0]
                                        ,np.quantile(all_samples,1-quantile_val,axis=0)[1]),'k--')

plt.savefig("rates.png", bbox_inches='tight')
plt.close('rates.png')


# cumm_rate(flux, a, b)
# flux = 1
# this section is where Devansh manually computes the quantiles to check if chainconsumer is working properly

cummlative_rate(1,10**np.quantile(all_samples,0.5,axis=0)[0],np.quantile(all_samples,0.5,axis=0)[1])
# a = 10**np.quantile(all_samples,0.5,axis=0)[0]
# b = np.quantile(all_samples,0.5,axis=0)[1]

cummlative_rate(1,10**np.quantile(all_samples,quantile_val,axis=0)[0],np.quantile(all_samples,quantile_val,axis=0)[1])
# a = 10**np.quantile(all_samples,quantile_val,axis=0)[0]
# b = np.quantile(all_samples,quantile_val,axis=0)[1]

cummlative_rate(1,10**np.quantile(all_samples,1-quantile_val,axis=0)[0],np.quantile(all_samples,1-quantile_val,axis=0)[1])
# a = 10**np.quantile(all_samples,1-quantile_val,axis=0)[0]
# b = np.quantile(all_samples,1-quantile_val,axis=0)[1]

cummlative_rate(1,10**np.quantile(all_samples,0.5,axis=0)[0],np.quantile(all_samples,0.5,axis=0)[1])-cummlative_rate(1,10**np.quantile(all_samples,quantile_val,axis=0)[0],np.quantile(all_samples,quantile_val,axis=0)[1])
# a = 10**np.quantile(all_samples,0.5,axis=0)[0]
# b = np.quantile(all_samples,0.5,axis=0)[1])-cummlative_rate(1,10**np.quantile(all_samples,quantile_val,axis=0)[0],np.quantile(all_samples,quantile_val,axis=0)[1]

cummlative_rate(1,10**np.quantile(all_samples,1-quantile_val,axis=0)[0],np.quantile(all_samples,1-quantile_val,axis=0)[1])-cummlative_rate(1,10**np.quantile(all_samples,0.5,axis=0)[0],np.quantile(all_samples,0.5,axis=0)[1])
# a = 10**np.quantile(all_samples,1-quantile_val,axis=0)[0]
# b = np.quantile(all_samples,1-quantile_val,axis=0)[1])-cummlative_rate(1,10**np.quantile(all_samples,0.5,axis=0)[0],np.quantile(all_samples,0.5,axis=0)[1]

cummlative_rate(1,10**np.quantile(all_samples,0.5,axis=0)[0],np.quantile(all_samples,0.5,axis=0)[1])
# a = 10**np.quantile(all_samples,0.5,axis=0)[0]
# b = np.quantile(all_samples,0.5,axis=0)[1]

np.quantile(samples[nwalkers*burnin:, 1], 0.5)
# computes the 0.5-th quantile of the data

np.quantile(samples[nwalkers*burnin:, 1], 0.5) - np.quantile(samples[nwalkers*burnin:, 1], quantile_val)
# subtracts ^ from the (quantile_val)-th quantile of the data

-np.quantile(samples[nwalkers*burnin:, 1], 0.5) + np.quantile(samples[nwalkers*burnin:, 1], 1-quantile_val)
# subtracts ^^ from the (1-quantile_val)-th quantile value of the data
# why is this negative?

np.quantile(samples[nwalkers*burnin:, 1], quantile_val) - np.quantile(samples[nwalkers*burnin:, 1], 0.5)
# subtracts (quantile_val)-th quantile of the data from ^^^

get_ipython().system('pip install chainconsumer --upgrade')
# installs the most current package for last plot

from chainconsumer import ChainConsumer
from matplotlib import rc # this is the matplotlib suggestion
rc('text', usetex=False)


c = ChainConsumer()
labels = [r"$\log \mathcal{R}$",r"$\alpha$"]#list(map(r"$\theta_{{{0}}}$".format, range(1, ndim+1)))
c.add_chain(all_samples, parameters=labels)
#c.plotter.plot(filename="example.png", figsize="column")
c.configure(flip=False, sigma2d=False, sigmas=[1, 2])  # The default case, so you don't need to specify sigma2d
fig = c.plotter.plot()
plt.savefig('last.png')
plt.close('last.png')

import os
os.sys.path.append("/usr/bin/")
