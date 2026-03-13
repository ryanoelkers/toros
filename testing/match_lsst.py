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
star_list = pd.read_csv(Configuration.ONE_DRIVE + 'master\\' + Configuration.FIELD + '\\' + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)
star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

star_list = star_list[(star_list.gc_star == 0) & (star_list.master_mag < 23)].copy().reset_index(drop=True)

# read in the list of LSST variable objects
# lsst_vars = pd.read_csv(Configuration.ONE_DRIVE + 'lsst\\lsst_data_47tuc_variables.csv',
#                        header=0,
#                        index_col=0,
#                        low_memory=False)

# read in the list of LSST transient objects
#lsst_trans = pd.read_csv(Configuration.ONE_DRIVE + 'lsst\\lsst_data_47tuc_transients.csv',
#                        header=0,
#                        index_col=0,
#                        low_memory=False)

# read in the list of LSST transient objects
lsst_objects = pd.read_csv(Configuration.ONE_DRIVE + 'lsst\\lsst_data_47tuc_objects.csv',
                        header=0,
                        index_col=0,
                        low_memory=False)
lsst_objects = lsst_objects[(~np.isnan(lsst_objects.g_psfMag)) &
                            (~np.isnan(lsst_objects.i_psfMag)) &
                            (~np.isnan(lsst_objects.r_psfMag))].copy().reset_index(drop=True)
lsst_objects = lsst_objects.sort_values(by='g_psfMag').reset_index(drop=True)

# convert everything to astropy coordinates
star_list_ra = star_list.ra.to_numpy() * u.degree
star_list_de = star_list.dec.to_numpy() * u.degree
star_list_coords = SkyCoord(ra=star_list_ra, dec=star_list_de, frame='icrs')
lsst_objects_ra = lsst_objects.coord_ra.to_numpy() * u.degree
lsst_objects_de = lsst_objects.coord_dec.to_numpy() * u.degree
lsst_objects_coords = SkyCoord(ra=lsst_objects_ra, dec=lsst_objects_de, frame='icrs')
lsst_objects['master_mag'] = -1

for idx, star in enumerate(lsst_objects_coords):

    sep = star.separation(star_list_coords).arcsec / Configuration.PIXEL_SIZE

    sep_pass = len(sep[sep < 10])
    sep_idx = np.argwhere(sep < 10).flatten()
    if sep_pass >= 1:
        if sep_pass > 1:
            mag_chk = (star_list.loc[np.argwhere(sep < 10).flatten(), "master_mag"].to_numpy() -
                     lsst_objects.loc[idx, "g_psfMag"])
            g_off = np.argwhere((mag_chk > 4.7) & (mag_chk < 5.5)).flatten()
            if len(g_off) == 1:
                lsst_objects.loc[idx, 'master_mag'] = star_list.loc[sep_idx[g_off], "master_mag"].values
            elif len(g_off) > 1:
                mn_off = np.argmin(sep[sep_idx[g_off]])
                lsst_objects.loc[idx, 'master_mag'] = star_list.loc[sep_idx[g_off[mn_off]], "master_mag"]
        elif sep_pass == 1:
            mag_chk = (star_list.loc[np.argwhere(sep < 10).flatten(), "master_mag"].to_numpy() -
                     lsst_objects.loc[idx, "g_psfMag"])
            if (mag_chk > 4.7) & (mag_chk < 5.5):
                lsst_objects.loc[idx, 'master_mag'] = star_list.loc[sep_idx, "master_mag"].values
    if idx % 1000 == 0:
        print(str(len(lsst_objects_coords) - idx - 1) + " stars remain.")
lsst_objects = lsst_objects[lsst_objects.master_mag > -1].copy().reset_index(drop=True)
plt.scatter(lsst_objects.master_mag, lsst_objects.g_psfMag - lsst_objects.master_mag)
plt.show()
print('hold')