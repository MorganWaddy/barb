import numpy as np
import json
import logging


def read_in_specidx(jsons):
    """
    Reads data from the command line
    
    Args:
        jsons ([str]): input data from json files
    
    Returns:
        nFRBs ([float]): number of FRBs detected
        sensitivity ([float]): sensitivity at FWHM divided by 2
        R ([float]): telescope beam radius
        beams ([float]): number of telescope beams
        tpb ([float]): time per beam
        flux ([float]): flux measurement of the FRB
        freq ([float]): center frequency of the observation
    """
    j = len(jsons)
    k = j + 1
    filename = jsons

    if j > 0:
        surveys = []
        nFRBs = []
        sensitivity = []
        R = []
        beams = []
        tpb = []
        freq = []
        flux = []
        logging.info("The number of user file(s) supplied is {0}".format(j) + "\n")
        logging.info("The supplied file(s) is/are {0}".format(filename) + "\n")
        for e in jsons:
            with open(e, "r") as fobj:
                info = json.load(fobj)
                for p in info["properties"]:
                    surveys.append(p["surveys"])
                    nFRBs = np.append(nFRBs, p["nFRBs"])
                    sensitivity = np.append(sensitivity, p["FWHM_2"])
                    R = np.append(R, p["R"])
                    beams = np.append(beams, p["beams"])
                    tpb = np.append(tpb, p["tpb"])
                    flux = np.append(flux, p["flux"])
                    #if specidx_analysis == True:
                       # for e in args.dat:
                            #with open(e, "r") as fobj:
                               # info = json.load(fobj)
                    freq = np.append(freq, p["freq"])
                    #else:
                    #freq = np.append(freq, 1)
                        #fobj.close()
    else:
            logging.info(
                "No data was supplied, please supply data on the command line!")
    return nFRBs, sensitivity, R, beams, tpb, flux, freq
