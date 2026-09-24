import pandas as pd
import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
from config import Configuration
from astropy.stats import sigma_clipped_stats as scs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import r_regression

def clipped_std(x, sigma=3):
    mean, median, std = scs(x, sigma=sigma)
    return std

ntr_stars = 500
lc_sze = 253
tot_stars = 60000

# the directories where the light curves reside
lc_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/star_list/raw/"

# read in the star list
star_list = pd.read_csv("/Users/yuw816/Data/toros/commissioning/master/FIELD_0e.001/"
                        + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)

star_list = star_list[star_list.source_id != star_list.lsst_id].copy().reset_index(drop=True)

# determine the 47Tuc distance
xcen_47tuc = 6853
ycen_47tuc = 5375
rad_47tuc = 1000

star_list['47T_dist'] = np.sqrt((star_list.xcen - xcen_47tuc) ** 2 +
                                (star_list.ycen - ycen_47tuc) ** 2)

f = open("/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/error_chk.txt", "w")
f.write("name mag mean_rms min_rms full_rms 47TUC_star\n")
tuc47 = 0
for idx, row in star_list[25921:tot_stars].iterrows():

    try:
        # update the star list to get the trend stars
        star_list['dmag'] = np.abs(row.master_mag - star_list['master_mag'])
        star_list['dist'] = np.sqrt((star_list.y - row.ycen) ** 2 + (star_list.x - row.xcen) ** 2)

        # get the trend stars
        if row['47T_dist'] > rad_47tuc:
            trend_list = star_list[(star_list.dist > 2 * Configuration.APER_SIZE) &
                                   (star_list['47T_dist'] > rad_47tuc)].copy().sort_values(by=['dmag']).reset_index(drop=True)
            tuc47 = 0
        else:
            trend_list = star_list[(star_list.dist > 2 * Configuration.APER_SIZE)].copy().sort_values(by=['dmag']).reset_index(drop=True)
            tuc47 = 1
        trend_list = trend_list[:ntr_stars].copy().reset_index(drop=True)

        # get the unique days
        if row.chip < 10:
            lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                             "0" + str(row.chip) +
                             "/FIELD_0e.001_" + str(row.source_id) + ".lc",
                             sep=" ")
        else:
            lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                             str(row.chip) +
                             "/FIELD_0e.001_" + str(row.source_id) + ".lc",
                             sep=" ")

        # the holders for each day
        lc_hold = np.zeros([lc_sze, ntr_stars])

        for idy, trow in trend_list.iterrows():

            # read in the light curves
            try:
                if trow.chip < 10:
                    tr = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                                     "0" + str(trow.chip) +
                                     "/FIELD_0e.001_" + str(trow.source_id) + ".lc",
                                     sep=" ")
                else:
                    tr = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                                     str(trow.chip) +
                                     "/FIELD_0e.001_" + str(trow.source_id) + ".lc",
                                     sep=" ")

                # update the matrix values
                lc_hold[:, idy] = tr.mag.to_numpy() - tr[tr.mag > 0].mag.median()
                if len(tr[tr.mag < 0].mag) > 0:
                    lc_hold[tr.mag.to_numpy() < 0, idy] = -9.9999
            except:
                lc_hold[:, idy] = np.zeros(lc_sze) - 9.9999

        # correlations
        cln_lc = np.zeros(lc_sze)
        raw = lc.mag.to_numpy()

        corrs = r_regression(lc_hold, raw)
        pass_corrs_idx = np.argwhere(corrs > 0.8).flatten()
        pass_corrs = len(corrs[corrs > 0.8])

        if pass_corrs > 0:
            lc_hold = lc_hold[:, pass_corrs_idx]

        _, off, _ = scs(lc_hold, axis=1, sigma=2.5, mask_value=-9.9999)

        cln_lc = raw - off
        cln_lc[raw < 0] = -9.9999

        lc['dys'] = lc.jd.astype(int)
        lc['mag'] = cln_lc

        agg_lc = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg(std_mag=('mag', clipped_std),
                                                                    total_obs=('mag', 'count'))
        mean_std = agg_lc[agg_lc.total_obs >= 6].std_mag.median()
        min_std = agg_lc[agg_lc.total_obs >= 6].std_mag.min()
        _, mean_mag, full_std = scs(lc[lc.mag > 0].mag, sigma=2.5)

        ph = (lc.jd - lc.jd.min()) / row.var_period % 1

        line = (str(row.source_id) + " " +
                str(np.around(mean_mag, decimals=4)) + " " +
                str(np.around(mean_std, decimals=4)) + " " +
                str(np.around(min_std, decimals=4)) + " " +
                str(np.around(full_std, decimals=4)) + " " +
                str(tuc47) + "\n")
        f.write(line)

        if idx % 100 == 0:
            Utils.log(str(tot_stars - idx - 1) + " stars remain.", "info")
    except:
        continue
f.close()
