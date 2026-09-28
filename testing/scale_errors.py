import matplotlib
import logging
import gc
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
import matplotlib.pyplot as plt
from config import Configuration
from libraries.utils import Utils
import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats as scs
from libraries.photometry import Photometry

# remove stars near 47 Tuc and the small cluster
star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + "_star_list.txt",
                        sep=' ', low_memory=False, index_col=0)
star_list['cat_source'] = 'toros'
star_list.loc[star_list.source_id == star_list.lsst_id, 'cat_source'] = 'lsst'

errors = pd.read_csv("/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/FIELD_0e.001_errors.txt",
                     delimiter=' ', low_memory=False)
re_chk = 'N'
if re_chk == 'Y':
    f = open(Configuration.LIGHTCURVE_STATS_DIRECTORY + Configuration.FIELD + "_scale_errors.txt", "w")
    f.write('name mag rms erms\n')

    for idx, row in star_list.iterrows():

        if row.cat_source == 'toros':
            if row.chip < 10:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                                 "star_list/detrend/" +
                                 "0" + str(row.chip) + "/" +
                                 Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                                 sep=" ")
            else:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                                 "star_list/detrend/" +
                                 str(row.chip) + "/" +
                                 Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                                 sep=" ")

            # now calculate the errors
            lc['dys'] = lc.jd.astype(int)

            # aggregate the light curve on a daily level and get the number of observations per day and clipped std
            agg_lc = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg(med_mag=('mag', 'median'),
                                                                        std_mag=('mag', Photometry.clipped_std),
                                                                        std_err=('err', Photometry.clipped_median),
                                                                        total_obs=('mag', 'count'))
            # get the median rms, minimum rms on the daily level and the full light curve rms
            med_rms = agg_lc[agg_lc.total_obs >= 6].std_mag.median()
            med_erms = agg_lc[agg_lc.total_obs >= 6].std_err.median()
            med_mag = agg_lc[agg_lc.total_obs >= 6].med_mag.median()

            line = (Configuration.FIELD + "_" + str(row.source_id) + ".lc" + " " +
                    str(np.around(med_mag, decimals=4)) + " " +
                    str(np.around(med_rms, decimals=4)) + " " +
                    str(np.around(med_erms, decimals=4)) + "\n")
            f.write(line)

            del agg_lc, lc
        if idx % 1000 == 0:
            Utils.log("Getting scale values for the next 1000 stars. " + str(len(star_list) - idx - 1) + " stars remain.", "info")
    f.close()

# read in the error file
errs = pd.read_csv(Configuration.LIGHTCURVE_STATS_DIRECTORY + Configuration.FIELD + "_scale_errors.txt",
                   sep=" ", low_memory=False)

# determine the scaling factor based on magnitude
mx_mag = np.nanmax(errs[errs.rms > 0].mag.to_numpy())
mn_mag = np.nanmin(errs[errs.rms > 0].mag.to_numpy())

stp_sze = 0.1
mgs = []
ers = []

# find the best errors per magnitude
for ii in np.arange(int(np.floor(mn_mag)), int(np.ceil(mx_mag)), stp_sze):
    clp_df = errs[(errs.mag > ii) & (errs.mag <= ii + stp_sze)].copy().reset_index(drop=True)

    if len(clp_df) > 50:
        err_clp = clp_df.rms.quantile(0.1)
        mgs.append(ii + stp_sze/2)
        ers.append(err_clp)

# re-scale photometric errors to make the rms-level magnitude
e_rms = np.interp(errors.mag, mgs, ers)
# errs.loc[errs.erms == 0, 'erms'] = 0.0001
# scl_rms = e_rms / errs.erms

# re-scale the errors
for idx, row in star_list.iterrows():

    if row.cat_source == 'toros':
        try:
            if row.chip < 10:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                                 "star_list/detrend/" +
                                 "0" + str(row.chip) + "/" +
                                 Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                                 sep=" ")
            else:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                                 "star_list/detrend/" +
                                 str(row.chip) + "/" +
                                 Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                                 sep=" ")
        except:
            if row.chip < 10:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                                 "lsst/detrend/" +
                                 "0" + str(row.chip) + "/" +
                                 Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                                 sep=" ")
            else:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                                 "lsst/detrend/" +
                                 str(row.chip) + "/" +
                                 Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                                 sep=" ")
    else:
        try:
            if row.chip < 10:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                                 "lsst/detrend/" +
                                 "0" + str(row.chip) + "/" +
                                 Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                                 sep=" ")
            else:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                                 "lsst/detrend/" +
                                 str(row.chip) + "/" +
                                 Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                                 sep=" ")
        except:
            if row.chip < 10:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                                 "star_list/detrend/" +
                                 "0" + str(row.chip) + "/" +
                                 Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                                 sep=" ")
            else:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                                 "star_list/detrend/" +
                                 str(row.chip) + "/" +
                                 Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                                 sep=" ")
    lc['dys'] = lc.jd.astype(int)
    lc.rename(columns={'err': 'err_nscl'}, inplace=True)
    agg_lc = lc[(lc.mag > 0) & (lc.err_nscl > 0)].groupby('dys').agg(std_err=('err_nscl', Photometry.clipped_median),
                                                                     total_obs=('mag', 'count'))

    # get the median rms, minimum rms on the daily level and the full light curve rms
    med_erms = agg_lc[agg_lc.total_obs >= 6].std_err.median()

    scl_rms = e_rms[idx] / med_erms
    lc['err'] = np.around(lc['err_nscl'] * scl_rms, decimals=4)

    lc = lc[['jd', 'mag', 'err', 'raw', 'err_nscl', 'trd', 'x', 'y']]

    if row.cat_source == 'toros':
        if row.chip < 10:
            lc.to_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                      "star_list/fin/" +
                      "0" + str(row.chip) + "/" +
                      Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                      sep=" ", header=True, index=False)
        else:
            lc.to_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                      "star_list/fin/" +
                      str(row.chip) + "/" +
                      Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                      sep=" ", header=True, index=False)
    else:
        if row.chip < 10:
            lc.to_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                      "lsst/fin/" +
                      "0" + str(row.chip) + "/" +
                      Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                      sep=" ", header=True, index=False)
        else:
            lc.to_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                      "lsst/fin/" +
                      str(row.chip) + "/" +
                      Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                      sep=" ", header=True, index=False)
    del lc

    if idx % 1000 == 0:
        Utils.log(str(len(star_list) - idx - 1) + ' stars remain to have their errors rescaled.', "info")