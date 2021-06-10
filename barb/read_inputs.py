import numpy as np
import json
import logging

def read_in(jsons):
    # data feed in structure
    j = len(jsons)
    k = j + 1
    filename = jsons

    if j > 0:
        surveys = []
        nFRBs = []
        FWHM_2 = []
        R = []
        beams = []
        tpb = []
        flux = []
        logging.info("Sanity Check:")
        logging.info(
            "The number of user file(s) supplied is {0}".format(j) + "\n")
        logging.info("The supplied file(s) is/are {0}".format(filename) + "\n")
        for e in jsons:
            with open(e, "r") as fobj:
                info = json.load(fobj)
                for p in info["properties"]:
                    surveys.append(p["surveys"])
                    nFRBs = np.append(nFRBs, p["nFRBs"])
                    FWHM_2 = np.append(FWHM_2, p["FWHM_2"])
                    R = np.append(R, p["R"])
                    beams = np.append(beams, p["beams"])
                    tpb = np.append(tpb, p["tpb"])
                    flux = np.append(flux, p["flux"])
        else:
            logging.info(
                "No data was supplied, please supply data on the command line!")
    return nFRBs, FWHM_2, R, tpb, flux
