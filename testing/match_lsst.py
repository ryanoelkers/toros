import pandas as pd
import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from config import Configuration
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

# read in the star list
star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)

# read in the list of LSST variable objects
lsst_vars = pd.read_csv(Configuration.DATA_DIRECTORY + 'lsst/lsst_data_47tuc_variables.csv',
                        header=0,
                        index_col=0,
                        low_memory=False)

# read in the list of LSST transient objects
lsst_trans = pd.read_csv(Configuration.DATA_DIRECTORY + 'lsst/lsst_data_47tuc_transients.csv',
                        header=0,
                        index_col=0,
                        low_memory=False)

# first we want to make sure we only do matching across one object at a time
lsst_vars_obs = lsst_vars.groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean'}).reset_index()
lsst_vars_obs['toros_id'] = 0
lsst_trans_obs = lsst_trans.groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean'}).reset_index()

# convert everything to astropy coordinates
star_list_ra = star_list.ra.to_numpy() * u.degree
star_list_de = star_list.dec.to_numpy() * u.degree
star_list_coords = SkyCoord(ra=star_list_ra, dec=star_list_de, frame='icrs')
lsst_vars_ra = lsst_vars_obs.coord_ra.to_numpy() * u.degree
lsst_vars_de = lsst_vars_obs.coord_dec.to_numpy() * u.degree
lsst_vars_coords = SkyCoord(ra=lsst_vars_ra, dec=lsst_vars_de, frame='icrs')
lsst_trans_ra = lsst_trans.coord_ra.to_numpy() * u.degree
lsst_trans_de = lsst_trans.coord_dec.to_numpy() * u.degree
lsst_trans_coords = SkyCoord(ra=lsst_trans_ra, dec=lsst_trans_de, frame='icrs')

Utils.log('Star process', "info")
for idx, var in enumerate(lsst_vars_coords):
    if idx % 1000:
        print('hello')
    var_sep = var.separation(star_list_coords).arcsec.min() / Configuration.PIXEL_SIZE
    if var_sep < 5:
        star_list_idx = np.argmin(var.separation(star_list_coords))
        lsst_vars_obs.loc[idx, 'toros_id'] = star_list.source_id[star_list_idx]