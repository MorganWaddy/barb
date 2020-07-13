# this is the first take stab at streamlining the code
import numpy as np
import pylab as plt
import matplotlib
matplotlib.use('Agg')
import math
from tqdm import tqdm
import emcee
import sys
import json

# this is for the text problem I encountered at the end of the code
from matplotlib import rc
rc('text', usetex=False)

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


def area(radius, b):
    ar=np.pi*radius*radius/(b*np.log(2))
    return ar

# power rule for integrals
# this return the value for the integral of [sensitivity^(beta)]d(sensitivity)
def power_integral(sensitivity, beta):
    return (sensitivity**-(beta-1)) / (beta-1)
    # sensitivity is sigma, measured in janskys
    # from appendix A:
    # beta = b + 1

def likelihood_list(data, alpha, beta):
    # from appendix A:
    # alpha = ab
    n, radius, time, sensitivity, flux = data
    A = area(radius, beta-1)
    I = power_integral(sensitivity, beta)
    taa = time*A*alpha
    ll = 0
    for idx, nburst in enumerate(n):
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

# returns results of log likelihood function using the modified alpha
log_ll([0.97504624, 1.91163861])
# log_ll([alpha,beta])
# this is supplying the alpha and beta for the calculations
# defining a file that houses the results of the functions
with open('calculations.out', 'a') as calc:
    logly = log_ll([0.97504624, 1.91163861])
    calc.write("log_ll([0.97504624, 1.91163861]) = "+str(logly)+"\n\n")
    calc.close()

###############################################################################
# commenting out all plots b/c of the problem: _tkinter.TclError: couldn't connect to display ":6.0"
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


ndim, nwalkers = 2, 1200
ivar = np.array([np.log10(15),2.5])
print("ivar is ", ivar)
# ivar is an intermediate variable
with open('calculations.out', 'a') as calc:
    calc.writelines("ivar is "+ str(ivar)+"\n\n")
    calc.close()


p0 = ivar + 0.05* np.random.uniform(size=(nwalkers, ndim))
plt.hist(p0[:,0])
plt.savefig('MCMC_hist1.png')
plt.hist(p0[:,1])
plt.savefig('MCMC_hist2.png')

from multiprocessing import cpu_count
ncpu = cpu_count() - 5#//2
print("{0} CPUs".format(ncpu))
from multiprocessing import Pool

pool = Pool(ncpu)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_ll, pool = pool)

def cummlative_rate(flux,a,b):
    return a*flux**(-b)

max_n = 100000

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


tau = sampler.get_autocorr_time()
burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))


all_samples = samples
all_samples[:,0]=np.log10((24 * 41253)*(10**all_samples[:,0])/(all_samples[:,1]-1))
all_samples[:,1]-=1

labels = [r"$\log \mathcal{R}$",r"$\alpha$"]

np.savez('all_samples_may_12_snr_12',all_samples)
all_samples
