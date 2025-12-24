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
from astropy.stats import sigma_clipped_stats as scs
from astropy.stats import sigma_clip as sc
import numpy as np
import statistics

# read in the star list
star_list = pd.read_csv(Configuration.ONE_DRIVE + 'master/' + Configuration.FIELD + '/' + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)
star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

star_list = star_list[(star_list.gc_star == 0) & (star_list.object_type != 'Star')].copy().reset_index(drop=True)

# read in the list of LSST variable objects
lsst_vars = pd.read_csv(Configuration.ONE_DRIVE + 'lsst\\lsst_data_47tuc_variables.csv',
                        header=0, index_col=0, low_memory=False)
var_list = lsst_vars.groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean'}).reset_index()

gt = 5.77
rt = 5.57
it = 5.72

star_list_ra = star_list.ra.to_numpy() * u.degree
star_list_de = star_list.dec.to_numpy() * u.degree
star_list_coords = SkyCoord(ra=star_list_ra, dec=star_list_de, frame='icrs')

lsst_objects_ra = var_list.coord_ra.to_numpy() * u.degree
lsst_objects_de = var_list.coord_dec.to_numpy() * u.degree
lsst_objects_coords = SkyCoord(ra=lsst_objects_ra, dec=lsst_objects_de, frame='icrs')

for idx, vary in star_list.iterrows():

    sep = star_list_coords[idx].separation(lsst_objects_coords).arcsec / Configuration.PIXEL_SIZE
    nmatch = len(sep[sep < 16])
    if nmatch > 1:
        vv = np.argwhere(sep < 16).flatten()
        shrt_list = lsst_vars[lsst_vars.diaObjectId.isin(var_list.iloc[vv].diaObjectId.values)].copy().reset_index(drop=True)
        mgs = -2.5 * np.log10(shrt_list.psfFlux.to_numpy()) + 31.4
        bnds = shrt_list.band.to_numpy()

        g_off = mgs[bnds == 'g'] - vary.master_mag

        ok = np.argwhere((g_off < -5) & (g_off > -6))
        if len(ok) > 0:
            lc = pd.read_csv(Configuration.ONE_DRIVE + "lc\\" + Configuration.FIELD + "\\" + Configuration.FIELD + '_' + vary['source_id'] + '.lc', sep=' ', header=0)
            ok_data = sc(lc.mag, sigma=3)
            ok_data.mask[np.argwhere(lc.mag == 0)] = True
            ok_data.mask[np.argwhere(lc.mag > 25)] = True

            lc = lc[~ok_data.mask].copy().reset_index(drop=True)

            print('hold')