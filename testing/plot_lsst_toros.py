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

star_list = star_list[(star_list.var_id != '--') & (star_list.var_period > 0) & (star_list.phot_g_mean_mag > 17)].copy().reset_index(drop=True)

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

for idx, row in star_list.iterrows():
    sep = star_list_coords[idx].separation(lsst_objects_coords).arcsec

    nmatch = len(sep[sep < .5])
    vv = np.argwhere(sep < .5).flatten()

    if nmatch > 0:
        perd = row['var_period']

        lc = pd.read_csv(Configuration.ONE_DRIVE + "lc\\" + Configuration.FIELD + "\\FIELD_0e.001_" + row['source_id'] + ".lc", sep=' ', header=0)
        ok_data = sc(lc.mag, sigma=3)
        ok_data.mask[np.argwhere(lc.mag == 0)] = True
        ok_data.mask[np.argwhere(lc.mag > 25)] = True

        lc = lc[~ok_data.mask].copy().reset_index(drop=True)
        lc['ph'] = (lc.jd - lc.jd.min()) / perd % 1
        lc['dys'] = lc.jd.to_numpy().astype(int)
        lc_bin = lc.groupby('dys').agg({'jd': 'mean',
                                        'mag': 'mean',
                                        'err': 'mean'}).reset_index()
        lc_bin_err = lc.groupby('dys').agg({'jd': 'mean',
                                            'mag': 'std'}).reset_index()

        lc_bin_err.loc[np.isnan(lc_bin_err.mag), 'mag'] = lc_bin[np.isnan(lc_bin_err.mag)]['err']

        shrt_list = lsst_vars[lsst_vars.diaObjectId.isin(var_list.iloc[vv].diaObjectId.values)].copy().reset_index(drop=True)
        shrt_list['mgs'] = -2.5 * np.log10(shrt_list.psfFlux.to_numpy()) + 31.4
        shrt_list['ers'] = (2.5 / np.log(10)) * (shrt_list.psfFluxErr.to_numpy() / shrt_list.psfFlux.to_numpy())
        shrt_list['jds'] = shrt_list.expMidptMJD.to_numpy() - 60630
        shrt_list = shrt_list[shrt_list.mgs < 25].copy().reset_index(drop=True)
        shrt_list = shrt_list[~np.isnan(shrt_list.mgs)].copy().reset_index(drop=True)

        lsst_ids = shrt_list.diaObjectId.unique()

        for idd in lsst_ids:
            gg = shrt_list[(shrt_list.diaObjectId == idd) & (shrt_list.band == 'g')].copy().reset_index(drop=True)
            gg['dys'] = gg.jds.to_numpy().astype(int)
            g = gg.groupby('dys').agg({'jds':'mean', 'mgs': 'mean', 'ers': 'mean'}).reset_index()
            g_bin = gg.groupby('dys').agg({'mgs': 'std'}).reset_index()

            rr = shrt_list[(shrt_list.diaObjectId == idd) & (shrt_list.band == 'r')].copy().reset_index(drop=True)
            rr['dys'] = rr.jds.to_numpy().astype(int)
            r = rr.groupby('dys').agg({'jds':'mean', 'mgs': 'mean', 'ers': 'mean'}).reset_index()
            r_bin = rr.groupby('dys').agg({'mgs': 'std'}).reset_index()

            ii = shrt_list[(shrt_list.diaObjectId == idd) & (shrt_list.band == 'i')].copy().reset_index(drop=True)
            ii['dys'] = ii.jds.to_numpy().astype(int)
            i = ii.groupby('dys').agg({'jds':'mean', 'mgs': 'mean', 'ers': 'mean'}).reset_index()
            i_bin = ii.groupby('dys').agg({'mgs': 'std'}).reset_index()

            plt.figure(figsize=(9, 6))

            plt.errorbar(lc_bin.jd - 2460630, lc_bin.mag, yerr=lc_bin_err.mag, c='k', fmt='none')
            plt.scatter(lc_bin.jd - 2460630, lc_bin.mag, c='k')

            plt.errorbar(r.jds, r.mgs + rt, yerr=r_bin.mgs, c='r', fmt='none')
            plt.scatter(r.jds, r.mgs + rt, c='r')

            plt.errorbar(g.jds, g.mgs + gt - 2, yerr=g_bin.mgs, c='darkgreen', fmt='none')
            plt.scatter(g.jds, g.mgs + gt -2 , c='darkgreen')

            plt.errorbar(i.jds, i.mgs + it +1, yerr=i_bin.mgs, c='maroon', fmt='none')
            plt.scatter(i.jds, i.mgs + it +1, c='maroon')

            plt.ylim([25., 23.])
            plt.ylabel('T', fontsize=20)
            plt.yticks(fontsize=15)

            plt.xlabel('JD - 2460630', fontsize=20)
            plt.xticks(fontsize=15)
            plt.savefig("mira.png", dpi=200, bbox_inches='tight')
            plt.show()
            plt.close()
