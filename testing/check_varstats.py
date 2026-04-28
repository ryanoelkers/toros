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

errors = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + '/varstats/' + Configuration.FIELD + '_errors.txt',
                     sep=' ')
errors['z'] = errors.rms / errors.erms

z = np.zeros(10)
mm = np.zeros(10)
for ii in np.arange(20, 30):
    _, z[ii-20], _ = scs(errors[(errors.mag > ii) & (errors.mag < ii + 1)].z.to_numpy(), sigma=1)
    mm[ii-20] = ii + 0.5
pp = np.polyfit(mm, z, 1)
vv = np.poly1d(pp)

f = open(Configuration.LIGHTCURVE_FIELD_DIRECTORY + '/varstats/' + Configuration.FIELD + '_varstats.txt', 'w')
f.write('name object_type var_type var_period mag daily_rms min_rms rms erms d90 Jstet Lstet p1 pwr1 fap1 p2 pwr2 fap2 p3 pwr3 fap3 p4 pwr4 fap4 p5 pwr5 fap5\n')

for idx, row in star_list.iterrows():

    lc = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY +
                     Configuration.FIELD + "/detrend/" +
                     Configuration.FIELD + "_" + str(row.source_id) + ".lc", sep=' ')
    lc['dys'] = lc.jd.to_numpy().astype('int')

    tmag, _, full_rms = scs(lc[(lc.mag > 0) & (lc.err > 0)].mag, sigma=2.5)
    lc['err'] = lc.err.to_numpy() / vv(tmag)
    rms_vals = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'std'}).to_numpy().flatten()
    num_obs = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'count'}).to_numpy().flatten()
    dytes = lc[lc.mag >0]['dys'].unique()

    try:
        min_rms = np.min(rms_vals[num_obs >= 6])
        dy_dte = np.argmin(rms_vals[num_obs >= 6])
        erms = lc[lc.dys == dytes[dy_dte]].err.mean()
    except:
        min_rms = -9.9999
        erms = lc[(lc.mag > 0) & (lc.err > 0)].err.mean()

    if (len(lc[(lc.mag > 0) & (lc.err > 0)]) > 10) & (min_rms > 0):
        mean_rms, rms, std_rms = scs(rms_vals[~np.isnan(rms_vals)], sigma=2.5)
        d90 = np.percentile(lc[(lc.mag > 0) & (lc.err > 0)].mag, 95) - np.percentile(lc[(lc.mag > 0) & (lc.err > 0)].mag, 5)
        # get the top 5 periods
        ls = LombScargle(lc[(lc.mag > 0) & (lc.err > 0)].jd.to_numpy(),
                         lc[(lc.mag > 0) & (lc.err > 0)].mag.to_numpy(),
                         dy=lc[(lc.mag > 0) & (lc.err > 0)].err.to_numpy())

        frequency, power = ls.autopower(minimum_frequency=0.02)

        # get the FAP
        shfl_mags = lc[(lc.mag > 0) & (lc.err > 0)].mag.to_numpy()
        mx_power = np.zeros(100)
        ss = np.random.choice(shfl_mags, size=(100, len(shfl_mags)), replace=True)

        for ii in range(0, 100):
            ss_f, ss_p = LombScargle(lc[(lc.mag > 0) & (lc.err > 0)].jd.to_numpy(),
                                     ss[ii],
                                     dy=lc[(lc.mag > 0) & (lc.err > 0)].err.to_numpy()).autopower(minimum_frequency=0.02)
            mx_power[ii] = np.max(ss_p)

        pwr5 = power[np.flip(np.argsort(power))][0:5]
        prd5 = 1. / frequency[np.flip(np.argsort(power))][0:5]
        fap5 = np.zeros(5)
        for ii in range(0, 5):
            fap5[ii] = len(mx_power[mx_power >= pwr5[ii]])

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

        f.write(Configuration.FIELD + "_" + str(row.source_id) + ".lc" + ' ' +
                row.object_type + " " + row.var_type + " " + str(np.around(row.var_period, decimals=4)) + " " +
                str(np.around(tmag, decimals=4)) + ' ' + str(np.around(rms, decimals=4)) + ' ' +
                str(np.around(min_rms, decimals=4)) + ' ' +
                str(np.around(full_rms, decimals=4)) + ' ' + str(np.around(erms, decimals=4)) + ' ' +
                str(np.around(d90, decimals=4)) + ' ' + str(np.around(jstet, decimals=4)) + ' ' +
                str(np.around(lstet, decimals=4)) + ' ' +
                str(np.around(prd5[0], decimals=4)) + ' ' + str(np.around(pwr5[0], decimals=4)) + ' ' + str(np.around(fap5[0], decimals=4)) + ' ' +
                str(np.around(prd5[1], decimals=4)) + ' ' + str(np.around(pwr5[1], decimals=4)) + ' ' + str(np.around(fap5[1], decimals=4)) + ' ' +
                str(np.around(prd5[2], decimals=4)) + ' ' + str(np.around(pwr5[2], decimals=4)) + ' ' + str(np.around(fap5[2], decimals=4)) + ' ' +
                str(np.around(prd5[3], decimals=4)) + ' ' + str(np.around(pwr5[3], decimals=4)) + ' ' + str(np.around(fap5[3], decimals=4)) + ' ' +
                str(np.around(prd5[4], decimals=4)) + ' ' + str(np.around(pwr5[4], decimals=4)) + ' ' + str(np.around(fap5[4], decimals=4)) + '\n')
    else:
        f.write(Configuration.FIELD + "_" + str(row.source_id) + ".lc" + ' ' +
                row.object_type + " " + row.var_type + " " + str(np.around(row.var_period, decimals=4)) + " " +
                str(np.around(tmag, decimals=4)) + ' ' + str(np.around(rms, decimals=4)) + ' ' +
                str(np.around(min_rms, decimals=4)) + ' ' +
                str(np.around(full_rms, decimals=4)) + ' ' + str(np.around(erms, decimals=4)) + ' ' +
                str(-9.9999) + ' ' + str(-9.9999) + ' ' +
                str(-9.9999) + ' ' +
                str(-9.9999) + ' ' + str(-9.9999) + ' ' + str(-9.9999) + ' ' +
                str(-9.9999) + ' ' + str(-9.9999) + ' ' + str(-9.9999) + ' ' +
                str(-9.9999) + ' ' + str(-9.9999) + ' ' + str(-9.9999) + ' ' +
                str(-9.9999) + ' ' + str(-9.9999) + ' ' + str(-9.9999) + ' ' +
                str(-9.9999) + ' ' + str(-9.9999) + ' ' + str(-9.9999) + '\n')
    if idx % 1000 == 0:
        Utils.log('Varstats calculated for ' + str(idx + 1) + ' stars. ' + str(len(star_list) - idx - 1) + ' stars remain.',
                  'info')
f.close()
