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
from astropy.stats import sigma_clip as sc
from astropy.timeseries import LombScargle
import warnings
warnings.simplefilter('error', RuntimeWarning)
from astropy.stats import sigma_clipped_stats

def clipped_median(x, sigma=3):
    mean, median, std = sigma_clipped_stats(x, sigma=sigma)
    return median
def clipped_std(x, sigma=3):
    mean, median, std = sigma_clipped_stats(x, sigma=sigma)
    return std

# remove stars near 47 Tuc and the small cluster
star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)
vary_list = star_list.copy().reset_index(drop=True)

# add new columns to star list
vary_list['mag'] = 0.
vary_list['rms'] = 0.
vary_list['min_rms'] = 0.
vary_list['full_rms'] = 0.
vary_list['out_mag'] = -9
vary_list['out_sct'] = -9
vary_list['jstet'] = -9.9999
vary_list['lstet'] = -9.9999
vary_list['d90'] = -9.9999
vary_list['prd'] = -9.9999
vary_list['pwr'] = -9.9999
vary_list['fap'] = -9.9999
vary_list['prox'] = 0
vary_list['cntm'] = 0
vary_list['edge'] = 0
vary_list['simp'] = 0
vary_list['pnts'] = 0

for idx, row in vary_list.iterrows():

    if row.chip < 10:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RESCALE_DIRECTORY + '/0' + str(row.chip) + '/' +
                         Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                         sep=" ")
    else:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RESCALE_DIRECTORY + '/' + str(row.chip) + '/' +
                         Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                         sep=" ")
    lc['dys'] = lc.jd.to_numpy().astype('int')

    # set up the proximity flag if necessary
    dist = np.sqrt((row.x - star_list.x) ** 2 + (row.y - star_list.y) ** 2)
    prox = len(dist[(dist > 0) & (dist <= 16)])
    if prox > 0:
        vary_list.loc[idx, 'prox'] = 1

    contam = len(dist[(dist > 0) & (dist <= 24)])
    if prox > 0:
        vary_list.loc[idx, 'cntm'] = 1

    # edge of the frame (600 < x < 10560) (0 < y < 9700)
    if (row.x < 700) | (row.x > 10460) | (row.y < 100) | (row.y > 9600):
        vary_list.loc[idx, 'edge'] = 1

    npts = len(lc[lc.mag < 0].mag) / len(lc)
    if npts > 0.1:
        vary_list.loc[idx, 'pnts'] = 1

    # get the rms values
    tmag, _, full_rms = scs(lc[(lc.mag > 0) & (lc.err > 0)].mag, sigma=2.5)
    vary_list.loc[idx, 'mag'] = np.around(tmag, decimals=4)  # get the TOROS magnitude
    vary_list.loc[idx, 'full_rms'] = np.around(full_rms, decimals=4)  # get the rms of the full light curve

    # determine if any days have a magnitude way higher or lower than normal
    mmag_vals = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'mean'}).to_numpy().flatten()
    try:
        clip_mag = sc(mmag_vals - np.mean(mmag_vals))
        vary_list.loc[idx, 'out_mag'] = len(clip_mag.data[clip_mag.mask])
    except:
        vary_list.loc[idx, 'out_mag'] = -1

    # determine if any days have large scatter
    rms_vals = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'std'}).to_numpy().flatten()
    try:
        clip_std = sc(rms_vals[rms_vals > 0] - np.nanmean(rms_vals[rms_vals > 0]))
        vary_list.loc[idx, 'out_sct'] = len(clip_std.data[clip_std.mask])
    except:
        vary_list.loc[idx, 'out_sct'] = -1

    # get the number of observations per day
    num_obs = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'count'}).to_numpy().flatten()

    clipd_std = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys')['mag'].agg(clipped_std)

    try:
        min_rms = np.around(np.min(rms_vals[num_obs >= 6]), decimals=4)
        vary_list.loc[idx, 'min_rms'] = min_rms  # get the minimum rms of the data

        clip_rms = np.median(clipd_std[num_obs >= 6])
        vary_list.loc[idx, 'rms'] = np.around(clip_rms, decimals=4)  # get the typical "daily" rms of the data
    except:
        min_rms = np.around(-9.9999, decimals=4)
        vary_list.loc[idx, 'min_rms'] = min_rms  # get the minimum rms of the data

        clip_rms = np.around(-9.9999, decimals=4)
        vary_list.loc[idx, 'rms'] = np.around(clip_rms, decimals=4)  # get the typical "daily" rms of the data

    if (len(lc[(lc.mag > 0) & (lc.err > 0)]) > 10) & (min_rms > 0):
        mean_rms, rms, std_rms = scs(rms_vals[~np.isnan(rms_vals)], sigma=2.5)
        d90 = (np.percentile(lc[(lc.mag > 0) & (lc.err > 0)].mag, 95) -
               np.percentile(lc[(lc.mag > 0) & (lc.err > 0)].mag, 5))
        vary_list.loc[idx, 'd90'] = np.around(d90, decimals=4)

        # get the top LS period
        ls = LombScargle(lc[(lc.mag > 0) & (lc.err > 0)].jd.to_numpy(),
                         lc[(lc.mag > 0) & (lc.err > 0)].mag.to_numpy(),
                         dy=lc[(lc.mag > 0) & (lc.err > 0)].err.to_numpy())

        # get the power spectrum
        frequency, power = ls.autopower(minimum_frequency=0.02, maximum_frequency=48)
        best_freq = frequency[np.argmax(power)]
        best_power = power[np.argmax(power)]
        best_period = 1. / best_freq

        # get the FAP
        try:
            fap = ls.false_alarm_probability(best_power, minimum_frequency=0.02, maximum_frequency=48)
            vary_list.loc[idx, 'prd'] = best_period
            vary_list.loc[idx, 'pwr'] = best_power
            vary_list.loc[idx, 'fap'] = fap
        except:
            vary_list.loc[idx, 'prd'] = -9.9999
            vary_list.loc[idx, 'pwr'] = -9.9999
            vary_list.loc[idx, 'fap'] = -9.9999

        # get J & L stet
        wk = 1.0  # Weighting Factor

        mg = lc[(lc.mag > 0) & (lc.err > 0)].mag.to_numpy()
        MeanMag = np.mean(mg)
        er = lc[(lc.mag > 0) & (lc.err > 0)].err.to_numpy()
        nms = len(lc[(lc.mag > 0) & (lc.err > 0)].jd.to_numpy())

        Jt = np.zeros(nms)
        Jb = np.zeros(nms)
        Kt = np.zeros(nms)
        Kb = np.zeros(nms)

        for i in range(0, nms-2, 2):

            Sigi = (mg[i] - MeanMag) / (er[i]) * (np.sqrt(nms / (nms - 1)))
            Sigj = (mg[i + 1] - MeanMag) / (er[i + 1]) * (np.sqrt(nms / (nms - 1)))

            Pk = Sigi * Sigj  # pg 853 Stetson 1996 Eq 2 Kinemuchi
            if Pk > 0.0:
                sgnPk = 1.0
            if Pk == 0.0:
                sgnPk = 0.0
            if Pk < 0.0:
                sgnPk = -1.0

            Jt[i] = wk * sgnPk * (np.sqrt(abs(Pk)))  # Kinemuchi eq.1 (Numerator)
            Jb[i] = wk  # Kinemuchi eq.1 (Denominator)
            Kt[i] = np.abs(Sigi)  # Kinemuchi eq.5 (Numerator)
            Kb[i] = np.abs(Sigi ** (2.0))  # Kinemuchi eq.5 (Denominator)

        jstet = np.sum(Jt) / np.sum(Jb)  # Eq 1
        if np.sum(Kb) != 0:
            kstet = ((1.0 / nms) * np.sum(Kt)) / (np.sqrt((1.0 / nms) * np.sum(Kb)))  # Eq 5
        else:
            kstet = 0.
        lstet = jstet * kstet / 0.7908
        vary_list.loc[idx, 'jstet'] = np.around(jstet, decimals=4)
        vary_list.loc[idx, 'lstet'] = np.around(lstet, decimals=4)

    if idx % 1000 == 0:
        Utils.log('Varstats calculated for ' + str(idx + 1) + ' stars. ' +
                  str(len(star_list) - idx - 1) + ' stars remain.',
                  'info')

for idx, row in vary_list.iterrows():
    vary_list.loc[idx, 'simp'] = (len(vary_list[(vary_list.prd == row.prd) & (vary_list.pwr > row.pwr)]) /
                                  len(vary_list[(vary_list.prd == row.prd)]))

vary_list.to_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + Configuration.FIELD + "_varstats.txt",
                 sep=' ', header=True, index=False)
errors = vary_list[['source_id', 'mag', 'rms', 'min_rms', 'full_rms']].copy().reset_index(drop=True)
errors = errors.rename(columns={'source_id': 'name'})

errors[errors.rms > 0].to_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + Configuration.FIELD + "_errors.txt",
                              sep=' ', header=True, index=False)