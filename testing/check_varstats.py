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
star_list = pd.read_csv("/Users/oelkerrj/OneDrive - The University of Texas-Rio Grande Valley/Research/TOROS/master/"
                        + Configuration.FIELD + "_star_list_updated.txt", sep=' ', low_memory=False, index_col=0)
bd_star = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                   (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

for idx in range(167940, len(star_list)):

    lc = pd.read_csv("/Users/oelkerrj/OneDrive - The University of Texas-Rio Grande Valley/Research/TOROS/lc/"
                     + Configuration.FIELD + "/" +
                     Configuration.FIELD +"_" + str(star_list.loc[idx].source_id) + ".lc", sep=' ')
    lc['cln'] = lc.mag.to_numpy() - lc.trd.to_numpy()

    clp = sc(lc.cln.to_numpy(), sigma=3)

    ls = LombScargle(lc[~clp.mask].jd.to_numpy(),
                     lc[~clp.mask].mag.to_numpy() - lc[~clp.mask].trd.to_numpy(),
                     dy=lc[~clp.mask].err.to_numpy())

    frequency, power = ls.autopower()

    shfl_mags = lc[~clp.mask].cln.to_numpy()

    mx_power = np.zeros(1000)
    for ii in range(0, 1000):
        ss = np.random.choice(shfl_mags, len(shfl_mags), replace=True)
        ss_f, ss_p = LombScargle(lc[~clp.mask].jd.to_numpy(),
                                 ss,
                                 dy=lc[~clp.mask].err.to_numpy()).autopower()
        mx_power[ii] = np.max(ss_p)

    pwr5 = power[np.flip(np.argsort(power))][0:5]
    prd5 = 1. / frequency[np.flip(np.argsort(power))][0:5]
    fap5 = np.zeros(5)
    for ii in range(0, 5):
        fap5[ii] = len(mx_power[mx_power >= pwr5[ii]])

    # get J stet
    wk = 1.0  # Weighting Factor

    MeanMag = np.mean(clp)

    mg = lc[~clp.mask].cln.to_numpy()
    MeanMag = np.mean(mg)
    er = lc[~clp.mask].err.to_numpy()
    nms = len(lc[~clp.mask].jd.to_numpy())

    Jt = np.arange(nms) * 0.0
    Jb = np.arange(nms) * 0.0
    Kt = np.arange(nms) * 0.0
    Kb = np.arange(nms) * 0.0

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
        Jb[i] = (wk)  # Kinemuchi eq.1 (Denominator)
        Kt[i] = abs(Sigi)  # Kinemuchi eq.5 (Numerator)
        Kb[i] = abs(Sigi ** (2.0))  # Kinemuchi eq.5 (Denominator)

    jstet = sum(Jt) / sum(Jb)  # Eq 1
    kstet = ((1.0 / nms) * sum(Kt)) / (np.sqrt((1.0 / nms) * sum(Kb)))  # Eq 5
    lstet = jstet * kstet / (0.7908)

    dys = np.unique(lc[~clp.mask].jd.to_numpy().astype(int))
    jstet_dys = np.zeros(len(dys))
    lstet_dys = np.zeros(len(dys))
    kstet_dys = np.zeros(len(dys))

    for jj in range(0, len(dys)):
        # get J stet
        wk = 1.0  # Weighting Factor

        MeanMag = np.mean(clp)

        mg = lc[(~clp.mask) & (lc.jd.to_numpy().astype(int) == dys[jj])].cln.to_numpy()
        MeanMag = np.mean(mg)
        er = lc[(~clp.mask) & (lc.jd.to_numpy().astype(int) == dys[jj])].err.to_numpy()
        nms = len(lc[(~clp.mask) & (lc.jd.to_numpy().astype(int) == dys[jj])].jd.to_numpy())

        Jt = np.arange(nms) * 0.0
        Jb = np.arange(nms) * 0.0
        Kt = np.arange(nms) * 0.0
        Kb = np.arange(nms) * 0.0

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
            Jb[i] = (wk)  # Kinemuchi eq.1 (Denominator)
            Kt[i] = abs(Sigi)  # Kinemuchi eq.5 (Numerator)
            Kb[i] = abs(Sigi ** (2.0))  # Kinemuchi eq.5 (Denominator)

        jstet_dys[jj] = sum(Jt) / sum(Jb)  # Eq 1
        kstet_dys[jj] = ((1.0 / nms) * sum(Kt)) / (np.sqrt((1.0 / nms) * sum(Kb)))  # Eq 5
        lstet_dys[jj] = jstet * kstet / (0.7908)

# get the light curve list and then read in light curves and get the rms values
# files = Utils.get_file_list(Configuration.LIGHTCURVE_FIELD_DIRECTORY, '.lc')
# lc = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY + Configuration.FIELD + "/" +
#                  Configuration.FIELD + "_4689821167286415744.lc", sep=' ')
# ph = (lc.jd - np.min(lc.jd)) / 0.59484267 % 1
#
# plt.figure(figsize=[9,6])
# plt.subplot(2, 1, 1)
# plt.scatter(lc[lc.mag > 0].jd - 2460000., lc[lc.mag > 0].mag - lc[lc.mag > 0].trd, c='k', marker='.')
# plt.ylim([15.9, 15.1])
# plt.ylabel('T')
# plt.xlabel('JD - 246000 [d]')
# plt.title('AQ-Tuc P=0.59484d')
# plt.subplot(2, 1, 2)
# plt.errorbar(ph[lc.mag > 0], lc[lc.mag > 0].mag - lc[lc.mag > 0].trd, yerr=lc[lc.mag > 0].err, c='k', fmt='none')
# plt.scatter(ph[lc.mag > 0], lc[lc.mag > 0].mag - lc[lc.mag > 0].trd, c='k', marker='.')
# plt.ylim([15.9, 15.1])
# plt.xlim([0,1])
# plt.ylabel('T')
# plt.xlabel('Phase')
# plt.show()
#
# lc = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY + Configuration.FIELD + "/" +
#                  Configuration.FIELD + "_4689579240377005184.lc", sep=' ')
# ph = (lc.jd - np.min(lc.jd)) / 0.37143 % 1
#
# plt.figure(figsize=[9,6])
# plt.subplot(2, 1, 1)
# plt.scatter(lc[lc.mag > 0].jd - 2460000., lc[lc.mag > 0].mag - lc[lc.mag > 0].trd, c='k', marker='.')
# plt.ylim([19.75, 19.])
# plt.ylabel('T')
# plt.xlabel('JD - 246000 [d]')
# plt.title('CO-Tuc P=0.37143d')
# plt.subplot(2, 1, 2)
# plt.errorbar(ph[lc.mag > 0], lc[lc.mag > 0].mag - lc[lc.mag > 0].trd, yerr=lc[lc.mag > 0].err, c='k', fmt='none')
# plt.scatter(ph[lc.mag > 0], lc[lc.mag > 0].mag - lc[lc.mag > 0].trd, c='k', marker='.')
# plt.ylim([19.75, 19.])
# plt.xlim([0,1])
# plt.ylabel('T')
# plt.xlabel('Phase')
# plt.show()

rms = np.zeros(167940)
mg = np.zeros(167940)
err1 = np.zeros(167940)
err2 = np.zeros(167940)
bin_st = 5
for idx in range(0, 167940, bin_st):
    if os.path.exists(Configuration.LIGHTCURVE_DIRECTORY + Configuration.FIELD + "/" +
                     Configuration.FIELD +"_" + str(star_list.loc[idx].source_id) + ".lc"):
        lc = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY + Configuration.FIELD + "/" +
                         Configuration.FIELD +"_" + str(star_list.loc[idx].source_id) + ".lc", sep=' ')
        _, _, rms[int(idx / bin_st)] = scs(lc[(lc.jd < 2460585.) & (lc.mag > 0)].mag -
                                     lc[(lc.jd < 2460585.) & (lc.mag > 0)].trd, sigma=3)
        flx = 10 ** ((lc[lc.mag > 0].mag.median() - 25 - 2.5*np.log10(300)) / (-2.5))
        err1[int(idx / bin_st)] = np.median(lc[lc.mag > 0].org_err)
        err2[int(idx / bin_st)] = np.sqrt(flx) / flx
        mg[int(idx / bin_st)] = star_list.loc[idx].master_mag

        del lc

    if (idx > 0) & (idx % 1000 == 0):
        Utils.log("1000 stars read in. " + str(len(star_list) - idx - 1) + ".", 'info')
        gc.collect()

mm = mg[mg > 0]
s2 = err2[mg > 0]
s1 = err1[mg > 0]
rms = rms[mg > 0]
ss = np.average([s2, s1], axis=0, weights=[1, 0.25])
tt = medfilt(ss, 25)
rr = medfilt(s2, 25)
kk = medfilt(s1, 25)
plt.scatter(mm+ 0.634924, rms, marker='.', c='k', alpha=0.2)
plt.plot(mm+ 0.634924, kk, marker='.', c='blue')
plt.plot(mm+ 0.634924, tt, marker='.', c='orange')
plt.plot(mm+ 0.634924, rr, marker='.', c='r')
plt.ylabel('rms')
plt.xlabel('T$_G$')
plt.yscale('log')
plt.ylim([0.002, 3])
plt.xlim([8, 22.5])
plt.show()
print('hold')