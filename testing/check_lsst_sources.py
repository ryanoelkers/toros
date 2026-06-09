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
import os

# star list from TOROS
star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)

# star list from LSST
lsst_list = pd.read_csv(Configuration.ANALYSIS_DIRECTORY + "lsst_sources.csv",
                        delimiter=',',
                        header=0,
                        low_memory=False)
f = open(Configuration.ANALYSIS_DIRECTORY + "lsst_sources/lsst_sources_region.reg", 'w')
f.write("# Region file format: DS9 version 4.1\n")
f.write("global color=red\n")

for idx, row in lsst_list.iterrows():

    # get the distances
    dist = np.sqrt((star_list.ra - row.ra)**2 + (star_list.dec - row.dec)**2)

    f.write('fk5;circle(' + str(row.ra) + ',' + str(row.dec) + ', 2")  # color=blue text={'+ str(row.id) +'} font="times 14 bold"\n')
    # get the minimum star distance
    min_idx = np.argmin(dist)
    min_dist = (np.min(dist) * 3600.) / Configuration.PIXEL_SIZE

    if min_dist < 1:
        lc = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY +
                     Configuration.FIELD + "/detrend/" +
                     Configuration.FIELD + "_" + str(star_list.source_id[min_idx]) + ".lc", sep=' ')
        os.system("cp " + Configuration.LIGHTCURVE_DIRECTORY +
                  Configuration.FIELD + "/detrend/" +
                  Configuration.FIELD + "_" + str(star_list.source_id[min_idx]) + ".lc " +
                  Configuration.ANALYSIS_DIRECTORY + '/lsst_sources/' + str(row.id) + '.lc')

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.errorbar(lc.jd[lc.mag > 0] - 2460584., lc.mag[lc.mag > 0] - 5.5, yerr=lc.err[lc.mag > 0], fmt='none', c='k')
        plt.scatter(lc.jd[lc.mag > 0] - 2460584., lc.mag[lc.mag > 0] - 5.5, c='k', label='De-Trended', marker='.')
        plt.xlabel('JD - 2460584 [d]')
        plt.ylabel('T$_G$')
        plt.title(row.id)
        plt.legend()
        plt.gca().invert_yaxis()

        plt.subplot(2, 1, 2)
        plt.errorbar(lc.jd[lc.mag > 0] - 2460584., lc.raw[lc.mag > 0] - 5.5, yerr=lc.err[lc.mag > 0], fmt='none', c='r')
        plt.scatter(lc.jd[lc.mag > 0] - 2460584., lc.raw[lc.mag > 0] - 5.5, c='r', label='Raw', marker='.')

        plt.xlabel('JD - 2460584 [d]')
        plt.ylabel('T$_G$')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(Configuration.ANALYSIS_DIRECTORY + '/lsst_sources/' + str(row.id) + '.png', bbox_inches='tight')
        plt.close()

f.close()