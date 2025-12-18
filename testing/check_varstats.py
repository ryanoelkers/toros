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
import os
from scipy.signal import medfilt
import gc
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip as sc
from random import choices
import astropy.units as u

# remove stars near 47 Tuc and the small cluster
star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)

f = open(Configuration.LIGHTCURVE_DIRECTORY + Configuration.FIELD + '_varstats.txt', 'w')
f.write('name mag rms d90 Jstet Lstet p1 pwr1 fap1 p2 pwr2 fap2 p3 pwr3 fap3 p4 pwr4 fap4 p5 pwr5 fap5 '
        'J1 J2 J3 J4 J5 J6 J7 J8 J9 J10 J11 J12 J13 J14 J15 J16 J17\n')

mgs = np.arange(14, 27)
err_zpt = np.array([56.02083333,  9.6       ,  9.26666667,  4.88235294,  3.34285714,
                    2.29333333,  1.44      ,  1.01856148,  0.66386875,  0.5035769 ,
                    0.39705882,  0.27193705,  0.13189532])

for idx in range(0, len(star_list)):

    lc = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY +
                     Configuration.FIELD + "/" +
                     Configuration.FIELD + "_" + str(star_list.loc[idx].source_id) + ".lc", sep=' ')
    lc['dys'] = lc.jd.to_numpy().astype('int')

    # clip the data based on outlier protection
    clp = sc(lc.mag.to_numpy(), sigma=3)
    clp.mask[np.argwhere(lc.mag <= 0)] = True

    # get the sigma-clipped rms
    tmag, _, full_rms = scs(lc[~clp.mask].mag, sigma=2.5)
    mean_rms, rms, std_rms = scs(lc[~clp.mask].groupby('dys').agg({'mag':'std'}).to_numpy().flatten(), sigma=2.5)

    # scale the error based on initial estimates
    lc['err'] = np.interp(tmag, mgs, err_zpt) * lc.err.to_numpy()
    erms = lc[~clp.mask].err.mean()

    # get the amplitude (d90)
    d90 = np.percentile(lc[~clp.mask].mag, 95) - np.percentile(lc[~clp.mask].mag, 5)

    # get the top 5 periods
    ls = LombScargle(lc[~clp.mask].jd.to_numpy(),
                     lc[~clp.mask].mag.to_numpy(),
                     dy=lc[~clp.mask].err.to_numpy())

    frequency, power = ls.autopower(minimum_frequency=0.02)

    # get the FAP
    shfl_mags = lc[~clp.mask].mag.to_numpy()
    mx_power = np.zeros(1000)
    for ii in range(0, 1000):
        ss = np.random.choice(shfl_mags, len(shfl_mags), replace=True)
        ss_f, ss_p = LombScargle(lc[~clp.mask].jd.to_numpy(),
                                 ss,
                                 dy=lc[~clp.mask].err.to_numpy()).autopower(minimum_frequency=0.02)
        mx_power[ii] = np.max(ss_p)

    pwr5 = power[np.flip(np.argsort(power))][0:5]
    prd5 = 1. / frequency[np.flip(np.argsort(power))][0:5]
    fap5 = np.zeros(5)
    for ii in range(0, 5):
        fap5[ii] = len(mx_power[mx_power >= pwr5[ii]])

    # get J & L stet
    wk = 1.0  # Weighting Factor

    mg = lc[~clp.mask].mag.to_numpy()
    MeanMag = np.mean(mg)
    er = lc[~clp.mask].err.to_numpy()
    nms = len(lc[~clp.mask].jd.to_numpy())

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
    kstet = ((1.0 / nms) * np.sum(Kt)) / (np.sqrt((1.0 / nms) * np.sum(Kb)))  # Eq 5
    lstet = jstet * kstet / 0.7908

    # do Jstet LStet per day
    if idx == 0:
        dys = np.unique(lc[~clp.mask].jd.to_numpy().astype(int))
    jstet_dys = np.zeros(len(dys))

    for jj in range(0, len(dys)):
        # get J stet
        wk = 1.0  # Weighting Factor

        mg = lc[(~clp.mask) & (lc.jd.to_numpy().astype(int) == dys[jj])].mag.to_numpy()
        MeanMag = np.mean(mg)
        er = lc[(~clp.mask) & (lc.jd.to_numpy().astype(int) == dys[jj])].err.to_numpy()
        nms = len(lc[(~clp.mask) & (lc.jd.to_numpy().astype(int) == dys[jj])].jd.to_numpy())

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

            Jt[i] = wk * sgnPk * (np.sqrt(np.abs(Pk)))  # Kinemuchi eq.1 (Numerator)
            Jb[i] = wk  # Kinemuchi eq.1 (Denominator)

        if np.sum(Jb) != 0:
            jstet_dys[jj] = np.sum(Jt) / np.sum(Jb)  # Eq 1
        else:
            jstet_dys[jj] = 0

    f.write(Configuration.FIELD + "_" + str(star_list.loc[idx].source_id) + ".lc" + ' ' +
            str(np.around(tmag, decimals=4)) + ' ' + str(np.around(rms, decimals=4)) + ' ' +
            str(np.around(full_rms, decimals=4)) + ' ' + str(np.around(erms, decimals=4)) + ' ' +
            str(np.around(d90, decimals=4)) + ' ' + str(np.around(jstet, decimals=4)) + ' ' +
            str(np.around(lstet, decimals=4)) + ' ' +
            str(np.around(prd5[0], decimals=4)) + ' ' + str(np.around(pwr5[0], decimals=4)) + ' ' + str(np.around(fap5[0], decimals=4)) + ' ' +
            str(np.around(prd5[1], decimals=4)) + ' ' + str(np.around(pwr5[1], decimals=4)) + ' ' + str(np.around(fap5[1], decimals=4)) + ' ' +
            str(np.around(prd5[2], decimals=4)) + ' ' + str(np.around(pwr5[2], decimals=4)) + ' ' + str(np.around(fap5[2], decimals=4)) + ' ' +
            str(np.around(prd5[3], decimals=4)) + ' ' + str(np.around(pwr5[3], decimals=4)) + ' ' + str(np.around(fap5[3], decimals=4)) + ' ' +
            str(np.around(prd5[4], decimals=4)) + ' ' + str(np.around(pwr5[4], decimals=4)) + ' ' + str(np.around(fap5[4], decimals=4)) + ' ' +
            str(np.around(jstet_dys[0], decimals=4)) + ' ' + str(np.around(jstet_dys[1], decimals=4)) + ' ' +
            str(np.around(jstet_dys[2], decimals=4)) + ' ' + str(np.around(jstet_dys[3], decimals=4)) + ' ' +
            str(np.around(jstet_dys[4], decimals=4)) + ' ' + str(np.around(jstet_dys[5], decimals=4)) + ' ' +
            str(np.around(jstet_dys[6], decimals=4)) + ' ' + str(np.around(jstet_dys[7], decimals=4)) + ' ' +
            str(np.around(jstet_dys[8], decimals=4)) + ' ' + str(np.around(jstet_dys[9], decimals=4)) + ' ' +
            str(np.around(jstet_dys[10], decimals=4)) + ' ' + str(np.around(jstet_dys[11], decimals=4)) + ' ' +
            str(np.around(jstet_dys[12], decimals=4)) + ' ' + str(np.around(jstet_dys[13], decimals=4)) + ' ' +
            str(np.around(jstet_dys[14], decimals=4)) + ' ' + str(np.around(jstet_dys[15], decimals=4)) + ' ' +
            str(np.around(jstet_dys[16], decimals=4)) + '\n')

    if idx % 1000 == 0:
        Utils.log('Varstats calculated for ' + str(idx + 1) + ' stars. ' + str(len(star_list) - 1) + ' stars remain.',
                  'info')
f.close()
