import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from config import Configuration
import numpy as np
tv_zpt = 5.4
xcen_47tuc = 6853
ycen_47tuc = 5375
rad_47tuc = 270

data_dir = "/Volumes/nomad_vandy/toros/lc_stats/"
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
varstats['v'] = varstats['mag'] - 5.4

lsststats = varstats[((varstats.var_pass == 1) | (varstats.per_pass == 1) | (varstats.day_pass == 1)) &
                     (varstats.cat_source == 'lsst')].copy().reset_index(drop=True)

# directories
known = pd.read_csv(data_dir + "lsst_data_47tuc_variables.csv",
                    delimiter=',',
                    header=0,
                    low_memory=False,
                    index_col=0)

# set up the magnitude and error
known['mag'] = -2.5 * np.log10(known.psfFlux) + 31.4
known['err'] = (np.log(10.) / 2.5) * (known['psfFluxErr'] / known['psfFlux'])
known = known[(known.mag > 0) & (known.err > 0) & (known.err < 1)].copy().reset_index(drop=True)
known['expMidptMJD'] = known['expMidptMJD'] + 2400000.5

lc_dir = "/Volumes/nomad_vandy/toros/lsst/fin/"

for idx, row in lsststats.iterrows():
    if row.per_pass == 1:

        lsst_lc = known[known.diaObjectId == int(row.source_id)].copy().reset_index(drop=True)

        if row.chip < 10:
            lc = pd.read_csv(lc_dir +
                             "0" + str(row.chip) + "/" +
                             Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                             sep=" ")
        else:
            lc = pd.read_csv(lc_dir +
                             str(row.chip) + "/" +
                             Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                             sep=" ")
        lc = lc[(lc.mag > 0) & (lc.err > 0)].copy().reset_index(drop=True)
        lc['ph'] = (lc.jd.to_numpy() - lc.jd.min()) / row.prd % 1

        plt.figure(figsize=(12, 9))

        plt.errorbar(lc.ph, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k')
        plt.scatter(lc.ph, lc.mag- tv_zpt, c='k')
        plt.gca().invert_yaxis()
        plt.title(str(np.around(row.prd, decimals=4)))

        lsst_lc['ph'] = (lsst_lc.expMidptMJD - lc.jd.min()) / row.prd % 1

        plt.errorbar(lsst_lc.ph, lsst_lc.mag - lsst_lc.mag.median() + np.median(lc.mag- tv_zpt), fmt='none', c='r')
        plt.scatter(lsst_lc.ph, lsst_lc.mag - lsst_lc.mag.median() + np.median(lc.mag- tv_zpt), c='r')

        plt.show()
        plt.close()
