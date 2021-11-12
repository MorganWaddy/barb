import numpy as np
import json
import logging
import sys

# sys.path.insert(0, "../barb/")
from barb.read_inputs import read_in

jsons = ["Dummier_2020_Rate-Data.json", "Dummy_2020_Rate-Data.json"]
# vargroup = (
#    [2, 4],
#    [0.24, 0.45],
#    [0.1, 0.11666666666666667],
#    [2, 3],
#    [524, 568],
#    [[4.2, 57.8, 10.2, 7.5], [7.2, 53.8, 9.2, 3.5]],
# )
vargroup = nFRBs, sensitivity, R, beams, tpb, flux


def test_read_in():
    assert read_in(jsons) == vargroup
