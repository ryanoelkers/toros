import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from lsst.rsp import get_tap_service

service = get_tap_service("tap")
assert service is not None

query = """SELECT coord_ra, coord_dec, g_psfMag, g_psfMagErr, 
        i_psfMag, i_psfMagErr, r_psfMag, r_psfMagErr, detect_isisoldated
        FROM dp1.Object
        WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec),
        CIRCLE('ICRS', 53, -28, 0.01)) = 1"""

job = service.submit_job(query)
job.run()
job.wait(phases=['COMPLETED', 'ERROR'])
print('Job phase is', job.phase)
if job.phase == 'ERROR':
    job.raise_if_error()
assert job.phase == 'COMPLETED'
results = job.fetch_result().to_table()

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
