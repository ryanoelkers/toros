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
import warnings
warnings.simplefilter('error', RuntimeWarning)

# read in the full star list that includes variability information
star_list = pd.read_csv("/Users/yuw816/Data/toros/commissioning/lc/" + Configuration.FIELD + "_varstats.txt",
                       delimiter=' ',
                       header=0,
                       low_memory=False)

ra = star_list['ra'].to_numpy()
dec = star_list['dec'].to_numpy()
pmra = star_list['pmra'].to_numpy()
pmdec = star_list['pmdec'].to_numpy()
parallax = star_list['parallax'].to_numpy()
ref_epoch = np.zeros(len(star_list)) + 2015.5

# Rows with usable proper motion
has_pm = np.isfinite(pmra) & np.isfinite(pmdec)

# Rows with a usable (positive, finite) parallax -> usable distance
has_plx = np.isfinite(parallax) & (parallax > 0)

# Default policy: missing PM -> treat as 0 (no motion correction for that star)
pmra_filled = np.where(has_pm, pmra, 0.0)
pmdec_filled = np.where(has_pm, pmdec, 0.0)

# Default policy: missing/bad parallax -> omit distance (angular-only correction, no perspective effect)
# astropy treats "no distance" as effectively at infinity, so just don't pass distance for those rows
distance_pc = np.where(has_plx, 1000.0 / parallax, 1000)  # NaN distance -> handled below

c = SkyCoord(
    ra=ra * u.deg,
    dec=dec * u.deg,
    pm_ra_cosdec=pmra_filled * u.mas/u.yr,
    pm_dec=pmdec_filled * u.mas/u.yr,
    # distance=distance_pc * u.pc,
    obstime=Time(ref_epoch, format='jyear'),
    frame='icrs'
    # distance intentionally omitted here — see note below
)

j2000 = Time(2000.0, format='jyear')
c_j2000 = c.apply_space_motion(new_obstime=j2000)

star_list['ra_j2000'] = c_j2000.ra.deg
star_list['dec_j2000'] = c_j2000.dec.deg

# read in the lsst data
lsst = pd.read_csv("/Volumes/OUMUAMUA/toros/commissioning/lsst/lsst_data_47tuc_objects.csv",
                   delimiter=',',
                   header=0,
                   low_memory=False,
                   index_col=0)

# now match the two data sets