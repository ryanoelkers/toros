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
from astropy.timeseries import LombScargle
import warnings
warnings.simplefilter('error', RuntimeWarning)


# remove stars near 47 Tuc and the small cluster
star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                        sep=' ',
                        header=0,
                        low_memory=False)

errors = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + '/varstats/' + Configuration.FIELD + '_errors.txt',
                     sep=' ',
                     header=0,
                     low_memory=False)

varstats_full = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + '/varstats/' + Configuration.FIELD + '_varstats.txt',
                       sep=' ',
                       header=0,
                       low_memory=False)
varstats = varstats_full[(star_list.object_type == 'Xray') & (star_list.parallax > 0) & (varstats_full.p1 > 0)].copy().reset_index(drop=True)
var_list = star_list[(star_list.object_type == 'Xray') & (star_list.parallax > 0) & (varstats_full.p1 > 0)].copy().reset_index(drop=True)

plt.scatter(var_list.phot_bp_mean_mag - var_list.phot_rp_mean_mag,
            var_list.phot_g_mean_mag + 5 - 5 * np.log10(1000./var_list.parallax),
            c=varstats['p1'])
plt.ylabel('G')
plt.xlabel(r"$B_P - R_P$")
plt.colorbar()
plt.show()

for idx, row in varstats.iterrows():

    np1_sim = len(varstats_full[(varstats_full.p1 == row['p1']) & (varstats_full.pwr1 > row['pwr1'])])
    np2_sim = len(varstats_full[(varstats_full.p2 == row['p2']) & (varstats_full.pwr1 > row['pwr2'])])
    np3_sim = len(varstats_full[(varstats_full.p3 == row['p3']) & (varstats_full.pwr1 > row['pwr3'])])
    np4_sim = len(varstats_full[(varstats_full.p4 == row['p4']) & (varstats_full.pwr1 > row['pwr4'])])
    np5_sim = len(varstats_full[(varstats_full.p5 == row['p5']) & (varstats_full.pwr1 > row['pwr5'])])

    if np1_sim <= 10:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DETREND_DIRECTORY + '/' + row['name'], sep=' ')
        lc['ph'] = (lc.jd - lc.jd.min()) / row['p1'] % 1

        plt.figure(figsize=(11, 8))
        plt.subplot(2, 1, 2)
        plt.scatter(lc[lc.mag > 0].jd - 2460000, lc[lc.mag > 0].mag, marker='.', c='k')
        plt.gca().invert_yaxis()
        plt.ylabel('Instrumental TOROS Magnitude')
        plt.xlabel('MJD [JD - 2460000]')

        plt.subplot(2, 1, 1)
        plt.scatter(lc[lc.mag > 0].ph, lc[lc.mag > 0].mag, marker='.', c='k')
        plt.gca().invert_yaxis()
        plt.ylabel('Instrumental TOROS Magnitude')
        plt.xlabel('Phase')
        plt.title(var_list.loc[idx, 'var_id'] + ' ' + var_list.loc[idx, 'var_type'] + ' P=' + str(row['p1']) + 'd')
        plt.show()
        plt.close()