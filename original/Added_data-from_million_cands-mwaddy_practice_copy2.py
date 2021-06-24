#!/usr/bin/env python
# coding: utf-8

# In[21]:


import sys
print(sys.version, sys.platform, sys.executable)
# this shows the system that I'm running and my environment


# In[22]:


import numpy as np
import pylab as plt
import math
# tqdm makes your loops show a smart progress meter
from tqdm import tqdm
#import corner
import emcee
#import os
#os.environ["OMP_NUM_THREADS"] = "1"


# In[23]:


from matplotlib import rc
rc('text', usetex=False)


# In[24]:


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


# In[25]:


# Zhang et al. 2019: A new fast radio burst in the datasets containing the Lorimer burst
n[0] = 3
flux[0].append(0.25)
# Zhang et al 2020: Parkes transient events: I. Database of single pulses, initial results and missing FRBs
flux[0].append(0.42)
tpb[0] += 250


# In[26]:


# Petroff et al. 2018: A fast radio burst with a low dispersion measure
n[-1] = 11
flux[-1].append(27)


# In[27]:


# Shannon et al. 2018 20 ASKAP FRBs
surveys.append("Shannon 2018")
n=np.append(n,20)
S=np.append(S,24.6)
R=np.append(R, 54/2/60)
beams = np.append(beams, 36)
tpb = np.append(tpb, 1700)
flux.append([58/2.4, 97/5.0, 34/4.4, 52/3.5, 74/2.5, 81/2.0, 219/5.4, 200/1.7, 63/2.3, 133/1.5, 40/1.9, 420/3.2,
        110/2.7, 51/2.9, 66/2.3, 95/4.1, 100/4.5, 96/1.81])


# In[28]:


# Bhandari et al. 2019 ASKAP FRB
surveys.append("Bhandari 2019")
n=np.append(n,1)
S=np.append(S,24.6)
R=np.append(R, 54/2/60)
beams = np.append(beams, 36)
tpb = np.append(tpb, 1286)
flux.append([46/1.9])


# In[29]:


# Qiu et al. 2019 ASKAP FRB
surveys.append("Qiu 2019")
n=np.append(n,1)
S=np.append(S,24.6)
R=np.append(R, 54/2/60)
beams = np.append(beams, 36)
tpb = np.append(tpb, 963.89)
flux.append([177])


# In[30]:


# Agarwal et al. 2019 ASKAP FRB
surveys.append("Agarwal 2019")
n=np.append(n,1)
S=np.append(S,24.6/np.sqrt(7))
R=np.append(R, 54/2/60)
beams = np.append(beams, 36)
tpb = np.append(tpb, 300)
flux.append([22])


# In[31]:


# Bhandari et al. 2018 Superb
surveys.append("Bhandari 2018")
n=np.append(n,4)
S=np.append(S,0.560/2)
R=np.append(R, 14/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 2722)
flux.append([0.7, 0.3, 0.43, 0.5])


# In[32]:


# Oslowski et al. 2019 PPTA
surveys.append("Oslowski 2019")
n=np.append(n,4)
S=np.append(S,0.560/2)
R=np.append(R, 14/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 659.5)
flux.append([1.2,23.5, 0.15, 0.6])


# In[33]:


# Masui et al 2015 GBT
surveys.append("Masui 2015")
n=np.append(n,1)
S=np.append(S,0.27)
R=np.append(R, 15/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 400)
flux.append([0.6])


# In[34]:


# Non detections


# In[35]:


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


# In[36]:


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


# In[37]:


# GBTrans
surveys.append("Golpayegani 2019")
n=np.append(n,0)
S=np.append(S,6*1.26)
R=np.append(R, 48/2/60)
beams = np.append(beams, 1)
tpb = np.append(tpb, 503*24)
flux.append([None])


# In[38]:


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


# In[ ]:





# In[39]:


time = tpb*beams


# In[ ]:





# In[40]:


def area(radius, b):
    ar=np.pi*radius*radius/(b*np.log(2))
    return ar


# In[41]:


# power rule for integrals
# this return the value for the integral of [sensitivity^(beta)]d(sensitivity)
def power_integral(sensitivity, beta):
    return (sensitivity**-(beta-1)) / (beta-1)
    # sensitivity is sigma, measured in janskys
    # from appendix A:
    # beta = b + 1


# In[42]:


# list of likelihoods 
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


# In[43]:


data = n, R, time, S, flux


# In[44]:


global data


# In[45]:


print(data)


# In[46]:


def log_ll(junk):
    alpha, beta = junk
    alpha = 10**alpha
    if (beta < 1):
        return -np.inf
    # returns a positive infinity multiplied by -1 (np.inf is a floating point constant value in the numpy library)
    return likelihood_list(data, alpha=alpha, beta=beta)


# In[47]:


log_ll([0.97504624, 1.91163861])
# log_ll([alpha,beta])
# returns results of log likelihood function using the modified alpha


# In[48]:


# numpy.linspace(start, stop, num = 50, endpoint = True, retstep = False, dtype = None)
# Returns number spaces evenly w.r.t interval. Similar to a range but instead of step it uses sample number.
bb = np.linspace(0.1,3,100)
# bb is to check what the log likelihood looks like
lk = [log_ll([np.log10(586.88/(24 * 41253)), b]) for b in bb]
# lk is running this log likelihood function with new junk in a loop for all instances of b in bb
# why np.log10(586.88/(24 * 41253)) and b instead of alpha and beta?


# In[49]:


bb[np.argmin(lk)]
# finds the minimum indice of lk, and then uses that as number to locate a specific element of bb


# In[50]:


plt.plot(bb-1,-1*np.array(lk))
plt.yscale('log')
# this plots on a log scale the array of values from lk multiplied by -1 vs the bb array - 1
# in the mcmc you maximize the likelihood, this is to check if the functions are working


# In[51]:


ndim, nwalkers = 2, 1200
# what is nwalkers? look up emcee
#ivar = np.array([np.log10(10*586.88/(24 * 41253)), 0.5])
ivar = np.array([np.log10(15),2.5])
print("ivar is ", ivar)
# what purpose does ivar serve? intermediate variable


# In[52]:


p0 = ivar + 0.05* np.random.uniform(size=(nwalkers, ndim))
#p0[:,0] = -8 + 8*p0[:,0]
plt.hist(p0[:,0])
plt.show()
#p0[:,1] = 1 + 3*p0[:,1]
plt.hist(p0[:,1])
plt.show()
# I think this creates a histogram of period of FRB detections vs. the ? (something something nwalkers)
# starting points


# In[53]:


from multiprocessing import cpu_count
ncpu = cpu_count() - 5#//2
print("{0} CPUs".format(ncpu))
from multiprocessing import Pool


# In[54]:


pool = Pool(ncpu)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_ll, pool = pool)
#sampler.run_mcmc(p0, 10000, progress=True)
#pool.close()


# In[55]:


def cummlative_rate(flux,a,b):
    return a*flux**(-b)


# In[56]:


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


# In[57]:


pool.close()


# In[58]:


tau = sampler.get_autocorr_time()
burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
#log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
#print("flat log prior shape: {0}".format(log_prior_samples.shape))


# In[59]:


all_samples = samples #np.concatenate((
#samples, log_prob_samples[:, None], ), axis=1)
all_samples[:,0]=np.log10((24 * 41253)*(10**all_samples[:,0])/(all_samples[:,1]-1))
all_samples[:,1]-=1

labels = [r"$\log \mathcal{R}$",r"$\alpha$"]#list(map(r"$\theta_{{{0}}}$".format, range(1, ndim+1)))
#labels += ["log prob"]


# In[60]:


np.savez('all_samples_may_12_snr_12',all_samples)
all_samples


# In[61]:


all_samples = np.load('all_samples_may_12_snr_12.npz')['arr_0']


# In[62]:


quantile_val = 0.99


# In[63]:


import corner


# In[64]:


all_samples[:,0] = (10**all_samples[:,0]).astype(np.int)


# In[65]:


plt.figure(figsize=(15,15))
corner.corner(all_samples, labels=labels, quantiles=[(1-0.99)/2,0.5,1-(1-0.99)/2],show_titles=True, bins=50)
#plt.savefig("rates_mc.pdf")
plt.show()


# In[ ]:





# In[66]:


fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.flatchain
labels = ["a", "b"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:nwalkers*burnin, i], "k", alpha=0.3)
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");


# In[67]:


fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.flatchain
labels = ["a", "b", ]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, i], "k", alpha=0.3)
    ax.set_xlim(0, len(all_samples))
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");


# In[68]:


quantile_val = 0.01
flux = np.logspace(-2,2,100)
plt.figure(figsize=(10,6))
total_label=[]
plt.plot(flux,cummlative_rate(flux,10**np.mean(all_samples[:,0]),np.mean(all_samples[:,1])),'k')
total_label.append("Best fit")
plt.plot(flux,cummlative_rate(flux,10**np.quantile(all_samples,quantile_val,axis=0)[0]
                                        ,np.quantile(all_samples,quantile_val,axis=0)[1]),'k--')
total_label.append(r"3$\sigma$ Error")

#total_label.append(r"3$\sigma$ Error")

#plt.plot(flux,cummlative_rate(flux,10**2.54,1.5),'b--')
#total_label.append("Eucledian")

#plt.plot(flux,cummlative_rate(flux,587,0.91),'b--')
#total_label.append("Lawrence et al")

plt.xlabel('Flux Limit (Jy)')
plt.ylabel(r'Rate (day$^{-1} sky^{-1}$)')
plt.xscale('log')
plt.yscale('log')

num_plots=len(surveys)
#colormap = plt.cm.gist_ncar
#plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
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

#


#for flux_limit, label_val in zip(S,surveys):
#    plt.axvline(flux_limit, **next(my_colors))#, label=label_val)
#    total_label.append(label_val)
#plt.errorbar(26/1.26, 37, yerr=8)
plt.grid()
#plt.legend(location=below)
plt.legend(total_label, ncol=5, loc='lower center', 
           bbox_to_anchor=[0.5, 1.1], 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)

plt.plot(flux,cummlative_rate(flux,10**np.quantile(all_samples,1-quantile_val,axis=0)[0]
                                        ,np.quantile(all_samples,1-quantile_val,axis=0)[1]),'k--')

plt.savefig("rates.pdf", bbox_inches='tight')
plt.show()


# In[69]:


cummlative_rate(1,10**np.quantile(all_samples,0.5,axis=0)[0],np.quantile(all_samples,0.5,axis=0)[1])
print(np.quantile(all_samples,0.5,axis=0)[0])
print(np.quantile(all_samples,0.5,axis=0)[1])


# In[70]:


cummlative_rate(1,10**np.quantile(all_samples,quantile_val,axis=0)[0],np.quantile(all_samples,quantile_val,axis=0)[1])


# In[71]:


cummlative_rate(1,10**np.quantile(all_samples,1-quantile_val,axis=0)[0],np.quantile(all_samples,1-quantile_val,axis=0)[1])


# In[72]:


cummlative_rate(1,10**np.quantile(all_samples,0.5,axis=0)[0],np.quantile(all_samples,0.5,axis=0)[1])-cummlative_rate(1,10**np.quantile(all_samples,quantile_val,axis=0)[0],np.quantile(all_samples,quantile_val,axis=0)[1])


# In[73]:


cummlative_rate(1,10**np.quantile(all_samples,1-quantile_val,axis=0)[0],np.quantile(all_samples,1-quantile_val,axis=0)[1])-cummlative_rate(1,10**np.quantile(all_samples,0.5,axis=0)[0],np.quantile(all_samples,0.5,axis=0)[1])


# In[74]:


cummlative_rate(1,10**np.quantile(all_samples,0.5,axis=0)[0],np.quantile(all_samples,0.5,axis=0)[1])


# In[75]:


np.quantile(samples[nwalkers*burnin:, 1], 0.5)


# In[76]:


np.quantile(samples[nwalkers*burnin:, 1], 0.5) - np.quantile(samples[nwalkers*burnin:, 1], quantile_val)


# In[77]:


-np.quantile(samples[nwalkers*burnin:, 1], 0.5) + np.quantile(samples[nwalkers*burnin:, 1], 1-quantile_val)


# In[78]:


np.quantile(samples[nwalkers*burnin:, 1], quantile_val) - np.quantile(samples[nwalkers*burnin:, 1], 0.5)


# In[79]:


get_ipython().system('pip install chainconsumer --upgrade')


# In[80]:


from chainconsumer import ChainConsumer
from matplotlib import rc # this is the matplotlib suggestion
rc('text', usetex=False)


# In[81]:


c = ChainConsumer()
labels = [r"$\log \mathcal{R}$",r"$\alpha$"]#list(map(r"$\theta_{{{0}}}$".format, range(1, ndim+1)))
c.add_chain(all_samples, parameters=labels)
#c.plotter.plot(filename="example.png", figsize="column")
c.configure(flip=False, sigma2d=False, sigmas=[1, 2])  # The default case, so you don't need to specify sigma2d
fig = c.plotter.plot()


# In[82]:


import os
os.sys.path.append("/usr/bin/")


# In[ ]:




