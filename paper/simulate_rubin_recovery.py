import pandas as pd
import numpy as np
import matplotlib
import logging
import random
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from config import Configuration
from libraries.varstats import Varstats
from scipy.stats import median_abs_deviation as mad

# read in the varstats file
varstats = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + Configuration.FIELD + "_varstats.txt",
                       sep=' ', low_memory=False)

# make the 3 sigma cuts on Jstet and Lstet (these are already appropriately scaled)
mad_jstet = mad(varstats['jstet'])
mdn_jstet = np.median(varstats['jstet'])
mad_lstet = mad(varstats['lstet'])
mdn_lstet = np.median(varstats['lstet'])

jstet_cut = mdn_jstet + 3 * mad_jstet
lstet_cut = mdn_lstet + 3 * mad_lstet

# get the list of passing stars
pass_vars = varstats[(varstats.jstet > jstet_cut) &
                     (varstats.lstet > lstet_cut) &
                     (varstats.object_type == 'LSST')].copy().reset_index(drop=True)
fail_vars = varstats[(varstats.jstet <= jstet_cut) &
                     (varstats.lstet <= lstet_cut) &
                     (varstats.object_type == 'Star')].copy().reset_index(drop=True)

# how many stars do you want to test and how many simulations do you want to run?
nstars = 1000
n_sims = 5
# select the random stars for selection
samp_vars = pass_vars.sample(n=nstars, random_state=1987, replace=False).reset_index(drop=True)
samp_cnst = fail_vars.sample(n=nstars, random_state=1987, replace=False).reset_index(drop=True)

del varstats, pass_vars, fail_vars

# set up the holders for the variable star light curves
vars_mags = np.zeros((nstars, 253))
vars_errs = np.zeros((nstars, 253))
# now read in the lightcurves for the 1000 variables and 1000 constant stars
for idx, row in samp_vars.iterrows():

    # read in the light curve
    if row.chip < 10:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RESCALE_DIRECTORY + '/0' + str(row.chip) + '/' +
                         Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                         sep=" ")
    else:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RESCALE_DIRECTORY + '/' + str(row.chip) + '/' +
                         Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                         sep=" ")
    # dump into holder matrix
    vars_mags[idx, :] = lc.mag.to_numpy()
    vars_errs[idx, :] = lc.err.to_numpy()
    del lc

    if idx % 100 == 0:
        Utils.log(str(int(idx)) + " variable light curves read. Working on next 100.", "info")

# set up the holders for the constant star light curves
cnst_mags = np.zeros((nstars, 253))
cnst_errs = np.zeros((nstars, 253))
# now read in the lightcurves for the 1000 variables and 1000 constant stars
for idx, row in samp_cnst.iterrows():

    # read in the light curve
    if row.chip < 10:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RESCALE_DIRECTORY + '/0' + str(row.chip) + '/' +
                         Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                         sep=" ")
    else:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RESCALE_DIRECTORY + '/' + str(row.chip) + '/' +
                         Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                         sep=" ")

    if idx == 0:
        jd = lc.jd.to_numpy()
    # dump into holder matrix
    cnst_mags[idx, :] = lc.mag.to_numpy()
    cnst_errs[idx, :] = lc.err.to_numpy()
    del lc

    if idx % 100 == 0:
        Utils.log(str(int(idx)) + " constant light curves read. Working on next 100.", "info")

# now we need to sample selections
time_idxs = np.arange(253)
rng = np.random.default_rng(1987)

num_pass = np.zeros((253, n_sims))
baseline = np.zeros((253, n_sims))

min_obs=3
for jdx in np.arange(min_obs, 253):
    fill_idx = jdx - min_obs
    if jdx % 10 == 0:
        # the number of simulated selections
        for idx in np.arange(n_sims):

            # pull the time selections
            times = rng.choice(time_idxs, size=jdx, replace=False)
            times = np.sort(times)  # make sure they are sorted for the Stetson caluclations

            j_v = np.zeros(nstars)
            l_v = np.zeros(nstars)
            j_c = np.zeros(nstars)
            l_c = np.zeros(nstars)

            for kdx in np.arange(nstars):
                j_v[kdx], _, l_v[kdx] = Varstats.stetson_metrics(vars_mags[kdx, times], vars_errs[kdx, times])
                j_c[kdx], _, l_c[kdx] = Varstats.stetson_metrics(cnst_mags[kdx, times], cnst_errs[kdx, times])

            # make the 3 sigma cuts on Jstet and Lstet (these are already appropriately scaled)
            mad_j_sim = mad(j_c)
            mdn_j_sim = np.median(j_c)
            mad_l_sim = mad(l_c)
            mdn_l_sim = np.median(l_c)

            j_cut_sim = mdn_j_sim + 3 * mad_j_sim
            l_cut_sim = mdn_l_sim + 3 * mad_l_sim

            num_pass[fill_idx, idx] = np.around(len(j_v[(j_v > j_cut_sim) & (l_v > l_cut_sim)]) / nstars, decimals=2)
            baseline[fill_idx, idx] = np.around(np.max(jd[times]) - np.min(jd[times]), decimals=6)
    Utils.log("Samples with " + str(jdx) + " lc points finished.", "info")
print('hold')
