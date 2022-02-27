import numpy as np
import json
import logging
import os
from barb.read_inputs import read_in

files = [
    "/home/mwaddy/FRB-rate/barb/test/Dummier_2020_Rate-Data.json",
    "/home/mwaddy/FRB-rate/barb/test/Dummy_2020_Rate-Data.json",
]
vargroup = [
    [2, 4],
    [0.24, 0.45],
    [0.1, 0.11666666666666667],
    [2, 3],
    [524, 568],
    [[4.2, 57.8, 10.2, 7.5], [7.2, 53.8, 9.2, 3.5]],
]
vargroup = np.array(vargroup, dtype=object)


def test_read_in():
    jsons = np.array(read_in(files))
    Dummy1 = "/home/mwaddy/FRB-rate/barb/test/Dummier_2020_Rate-Data.json"
    Dummy2 = "/home/mwaddy/FRB-rate/barb/test/Dummy_2020_Rate-Data.json"
    assert os.path.isfile(Dummy1)
    assert os.path.isfile(Dummy2)
    assert jsons.all() == vargroup.all()
