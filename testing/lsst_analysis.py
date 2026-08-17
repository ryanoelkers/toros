import pandas as pd
import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
import matplotlib.pyplot as plt
from config import Configuration
import numpy as np
from astropy.stats import sigma_clipped_stats as scs
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import warnings
warnings.simplefilter('error', RuntimeWarning)

# read in the full star list that includes variability information
mac_dir = "/Users/yuw816/Data/toros/commissioning/"

star_list = pd.read_csv(mac_dir + "lc/" + Configuration.FIELD + "_varstats.txt",
                       delimiter=' ',
                       header=0,
                       low_memory=False)


# read in the lsst data
lsst_vars = pd.read_csv(mac_dir + "lsst/" + "lsst_data_47tuc_variables.csv",
                        delimiter=',',
                        header=0,
                        low_memory=False,
                        index_col=0)

lsst_vars = lsst_vars.groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean', 'psfFlux':'max'}).reset_index()
lsst_vars['psfMag'] = lsst_vars.apply(lambda x: 31.4 - 2.5 * np.log10(x.psfFlux), axis=1)

# get the header file and convert to x/y pixel positions
master, master_header = fits.getdata(mac_dir + "/master/FIELD_0e.001/FIELD_0e.001_master.fits", header=True)
w = WCS(master_header)
ra = lsst_vars.coord_ra.to_numpy()
dec = lsst_vars.coord_dec.to_numpy()

# convert to x, y
# edge of the frame (600 < x < 10560) (0 < y < 9700)
x, y = w.all_world2pix(ra, dec, 0)
lsst_vars = lsst_vars[(x > 500) & (x < 10540) & (y > 20) & (y < 9700)].copy().reset_index(drop=True)

star_list = pd.read_csv(mac_dir + "lc/" + Configuration.FIELD + "_varstats.txt",
                       delimiter=' ',
                       header=0,
                       low_memory=False)

iso_star_list = star_list[(star_list.prox == 0) & (star_list.var_id == '--')].copy().reset_index(drop=True)
iso_star_list['lsst_g'] = -9
iso_star_list['lsst_r'] = -9
iso_star_list['lsst_i'] = -9

# read in the lsst data
lsst = pd.read_csv(mac_dir + "lsst/" + "lsst_data_47tuc_objects.csv",
                   delimiter=',',
                   header=0,
                   low_memory=False,
                   index_col=0)

for idx, row in iso_star_list.iterrows():
    star_dist_asec = np.sqrt((row.ra - lsst.coord_ra.to_numpy()) ** 2 + (row.dec - lsst.coord_dec.to_numpy()) ** 2) * 3600.
    star_dist_pix = star_dist_asec / Configuration.PIXEL_SIZE

    pss = np.argwhere(star_dist_pix < 2).flatten()
    if len(pss) == 1:

        bp_m_rp = row.phot_bp_mean_mag - row.phot_rp_mean_mag
        g_to_r = (row.phot_g_mean_mag + 0.09837 - 0.8592 * bp_m_rp - 0.1907 * bp_m_rp ** 2 +
                  0.1701 * bp_m_rp ** 3 - 0.02263 * bp_m_rp ** 4)
        g_to_i = (row.phot_g_mean_mag + 0.293 - 0.6404 * bp_m_rp + 0.09609 * bp_m_rp ** 2 +
                  0.002104 * bp_m_rp ** 3)
        g_to_g = (row.phot_g_mean_mag - 0.2199 + 0.6365 * bp_m_rp + 0.1548 * bp_m_rp ** 2 -
                  0.0064 * bp_m_rp ** 3)

        gmi = lsst.loc[pss, 'g_psfMag'].to_numpy() - lsst.loc[pss, 'i_psfMag'].to_numpy()
        rmi = lsst.loc[pss, 'r_psfMag'].to_numpy() - lsst.loc[pss, 'i_psfMag'].to_numpy()

        ggs = lsst.loc[pss, 'g_psfMag'].to_numpy() - 0.1064 - 0.4964 * gmi - 0.09339 * gmi ** 2 + 0.004444 * gmi ** 3
        grs = lsst.loc[pss, 'r_psfMag'].to_numpy() - 0.01664 - 0.2662 * rmi - 0.649 * rmi ** 2 + 0.08227 * rmi ** 3
        gis = lsst.loc[pss, 'i_psfMag'].to_numpy() - 0.01066 + 1.298 * rmi - 0.7595 * rmi ** 2 + 0.1492 * rmi ** 3

        offset_g = np.abs(row.phot_g_mean_mag - ggs)

        if offset_g < 0.05:
            iso_star_list.loc[idx, 'lsst_g'] = lsst.loc[pss, 'g_psfMag'].values
            iso_star_list.loc[idx, 'lsst_r'] = lsst.loc[pss, 'r_psfMag'].values
            iso_star_list.loc[idx, 'lsst_i'] = lsst.loc[pss, 'i_psfMag'].values

iso_star_list = iso_star_list[(iso_star_list.lsst_g > 0) & (iso_star_list.lsst_r > 0) & (iso_star_list.lsst_i > 0)]

plt.hist(lsst_vars.psfMag)

plt.show()