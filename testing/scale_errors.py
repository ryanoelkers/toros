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

# remove stars near 47 Tuc and the smallcluster
star_list = pd.read_csv("/Volumes/OUMUAMUA/toros/commissioning/master/" + Configuration.FIELD + "/"
                        + Configuration.FIELD + "_star_list.txt", sep=' ', low_memory=False, index_col=0)

# redo the uncertainties?
reydo = 'N'

if reydo == 'Y':
    f = open("/Volumes/OUMUAMUA/toros/commissioning/lc/" + Configuration.FIELD + "_errors.txt", 'w')
    f.write('file mag rms erms full_rms\n')
    for idx, row in star_list.iterrows():

        if row.chip < 10:
            lc = pd.read_csv('/Volumes/OUMUAMUA/toros/commissioning/lc/' + Configuration.FIELD + '/0' +
                             str(row.chip) + '/' + Configuration.FIELD + '_' + row.source_id + '.lc',
                             sep=' ')
        else:
            lc = pd.read_csv('/Volumes/OUMUAMUA/toros/commissioning/lc/' + Configuration.FIELD + '/' +
                             str(row.chip) + '/' + Configuration.FIELD + '_' + row.source_id + '.lc',
                             sep=' ')

        mag, _, full_rms = scs(lc[(lc.mag > 0) & (lc.err > 0)].mag, sigma=2.5)
        lc['dys'] = lc.jd.to_numpy().astype('int')

        rms_vals = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'std'}).to_numpy().flatten()
        num_obs = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'count'}).to_numpy().flatten()

        erms = lc[(lc.mag > 0) & (lc.err > 0)].err.mean()
        try:
            rms = np.median(rms_vals[num_obs >= 6])
        except:
            rms = full_rms
        del lc

        f.write(Configuration.FIELD + '_' + row.source_id + '.lc ' +
                str(np.around(mag, decimals=4)) + ' ' +
                str(np.around(rms, decimals=4)) + ' ' +
                str(np.around(erms, decimals=4)) + ' ' +
                str(np.around(full_rms, decimals=4)) + '\n')

        if idx % 1000 == 0:
            Utils.log(str(len(star_list) - idx - 1) + ' stars remaining for error calculations.', "info")
    f.close()

# read in the error file
errs = pd.read_csv("/Volumes/OUMUAMUA/toros/commissioning/lc/" + Configuration.FIELD + "_errors.txt",
                   sep=" ")

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
e_rms = np.interp(errs.mag, mgs, ers)
scl_rms = e_rms / errs.erms

# re-scale the errors
for idx, row in star_list.iterrows():

    if row.chip < 10:
        lc = pd.read_csv('/Volumes/OUMUAMUA/toros/commissioning/lc/' + Configuration.FIELD + '/detrend/0' +
                         str(row.chip) + '/' + Configuration.FIELD + '_' + row.source_id + '.lc',
                         sep=' ')
    else:
        lc = pd.read_csv('/Volumes/OUMUAMUA/toros/commissioning/lc/' + Configuration.FIELD + '/detrend/' +
                         str(row.chip) + '/' + Configuration.FIELD + '_' + row.source_id + '.lc',
                         sep=' ')

    lc.rename(columns={'err': 'err_nscl'}, inplace=True)

    lc['err'] = np.around(lc['err_nscl'] * scl_rms[idx], decimals=4)

    lc = lc[['jd', 'mag', 'err', 'raw', 'err_nscl', 'trd', 'sky', 'bkg', 'x', 'y', 'nstars', 'airmass']]

    if row.chip < 10:
        lc.to_csv('/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/rescale/0' +
                  str(row.chip) + '/' + Configuration.FIELD + '_' + row.source_id + '.lc',
                  sep=' ', index=False)
    else:
        lc.to_csv('/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/rescale/' +
                  str(row.chip) + '/' + Configuration.FIELD + '_' + row.source_id + '.lc',
                  sep=' ', index=False)
    if idx % 1000 == 0:
        Utils.log(str(len(star_list) - idx - 1) + ' stars remaining for error calculations.', "info")