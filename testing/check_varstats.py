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
                        delimiter=' ',
                        header=0,
                        low_memory=False)
var_list = star_list.copy().reset_index(drop=True)

# add new columns to star list
var_list['tmag'] = 0.
var_list['rms'] = 0.
var_list['min_rms'] = 0.
var_list['full_rms'] = 0.
var_list['jstet'] = 0.
var_list['lstet'] = 0.
var_list['d90'] = 0.
var_list['prd1'] = 0.
var_list['pwr1'] = 0.
var_list['fap1'] = 0.
var_list['prd2'] = 0.
var_list['pwr2'] = 0.
var_list['fap2'] = 0.
var_list['prd3'] = 0.
var_list['pwr3'] = 0.
var_list['fap3'] = 0.

for idx, row in var_list.iterrows():

    lc = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY +
                     Configuration.FIELD + "/detrend/" +
                     Configuration.FIELD + "_" + str(row.source_id) + ".lc", sep=' ')
    lc['dys'] = lc.jd.to_numpy().astype('int')

    tmag, _, full_rms = scs(lc[(lc.mag > 0) & (lc.err > 0)].mag, sigma=2.5)
    var_list.loc[idx, 'tmag'] = np.around(tmag, decimals=4)
    var_list.loc[idx, 'full_rms'] = np.around(full_rms, decimals=4)

    rms_vals = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'std'}).to_numpy().flatten()
    num_obs = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'count'}).to_numpy().flatten()

    try:
        min_rms = np.around(np.min(rms_vals[num_obs >= 6]), decimals=4)
        var_list.loc[idx, 'min_rms'] = min_rms

        daily_rms = np.mean(rms_vals[num_obs >= 6])
        var_list.loc[idx, 'rms'] = np.around(daily_rms, decimals=4)
    except:
        min_rms = np.around(-9.9999, decimals=4)
        var_list.loc[idx, 'min_rms'] = min_rms

        daily_rms = np.around(-9.9999, decimals=4)
        var_list.loc[idx, 'rms'] = daily_rms

    if (len(lc[(lc.mag > 0) & (lc.err > 0)]) > 10) & (min_rms > 0):
        mean_rms, rms, std_rms = scs(rms_vals[~np.isnan(rms_vals)], sigma=2.5)
        d90 = (np.percentile(lc[(lc.mag > 0) & (lc.err > 0)].mag, 95) -
               np.percentile(lc[(lc.mag > 0) & (lc.err > 0)].mag, 5))

        # get the top 3 periods with whitening
        resid = lc[(lc.mag > 0) & (lc.err > 0)].mag.to_numpy()
        for idy in range(3):
            ls = LombScargle(lc[(lc.mag > 0) & (lc.err > 0)].jd.to_numpy(),
                             resid,
                             dy=lc[(lc.mag > 0) & (lc.err > 0)].err.to_numpy())

            # get the power spectrum
            frequency, power = ls.autopower(minimum_frequency=0.02, maximum_frequency=48)
            best_freq = frequency[np.argmax(power)]
            best_power = power[np.argmax(power)]
            best_period = 1. / best_freq

            fap = ls.false_alarm_probability(best_power, minimum_frequency=0.02, maximum_frequency=48)

            model = ls.model(lc[(lc.mag > 0) & (lc.err > 0)].jd.to_numpy(), best_freq)

            var_list.loc[idx, 'prd' + str(idy + 1)] = best_period
            var_list.loc[idx, 'pwr' + str(idy + 1)] = best_power
            var_list.loc[idx, 'fap' + str(idy + 1)] = fap

            resid = lc[(lc.mag > 0) & (lc.err > 0)].mag.to_numpy() - model
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

    if idx % 1000 == 0:
        Utils.log('Varstats calculated for ' + str(idx + 1) + ' stars. ' + str(len(star_list) - idx - 1) + ' stars remain.',
                  'info')
star_list.to_csv("/Users/yuw816/Data/toros/commissioning/lc/" + Configuration.FIELD + "_varstats.txt",
                 sep=' ', header=True, index=False)
