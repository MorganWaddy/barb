# this is what I'm coverting into individual json files

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
