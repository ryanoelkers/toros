import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from lsst.rsp import get_tap_service

service = get_tap_service("tap")
assert service is not None
filter_names = ['u', 'g', 'r', 'i', 'z', 'y']
filter_colors = get_multiband_plot_colors()
filter_symbols = get_multiband_plot_symbols()
ra_var = 94.9226329830
dec_var = -25.2318482104
search_rad = 0.5/3600


query = "SELECT fsodo.diaObjectId, fsodo.coord_ra, fsodo.coord_dec, "\
        "fsodo.visit, fsodo.detector, fsodo.band, "\
        "fsodo.psfDiffFlux, fsodo.psfDiffFluxErr, "\
        "fsodo.psfFlux as psfFlux, fsodo.psfFluxErr, "\
        "vis.expMidptMJD "\
        "FROM dp1.ForcedSourceOnDiaObject as fsodo "\
        "JOIN dp1.Visit as vis ON vis.visit = fsodo.visit "\
        "WHERE CONTAINS (POINT('ICRS', coord_ra, coord_dec), "\
        "CIRCLE('ICRS', " + str(ra_var) + ", "\
        + str(dec_var) + ", " + str(search_rad) + ")) = 1 "
print(query)
