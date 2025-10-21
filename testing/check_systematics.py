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
from scipy.stats import spearmanr
import gc
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip as sc
from random import choices
from sklearn.linear_model import LinearRegression

def jstet(mg, er):

    wk = 1.0  # Weighting Factor

    MeanMag = np.mean(mg)
    nms = len(mg)

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

    return jstet, kstet, lstet


star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)

star_list['bd_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 2, 0)
# add "chip" to the star_list
star_list['chip'] = 1
kk = 1
for idx in range(0, Configuration.AXS_X, Configuration.CHP_X):
    for idy in range(0, Configuration.AXS_Y, Configuration.CHP_Y):
        star_list['chip'] = np.where((star_list.xcen > idx) & (star_list.xcen < idx + 1320) &
                                     (star_list.ycen > idy) & (star_list.ycen < idy + 5280),
                                     kk, star_list.chip)
        kk = kk + 1

dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/"
all_files = np.sort(Utils.get_file_list(dir, '.lc'))
odir = "/Users/yuw816/Library/CloudStorage/OneDrive-TheUniversityofTexas-RioGrandeValley/Research/TOROS/lc/FIELD_0e.001/"
# 28529
for idx, row in star_list[28529:].iterrows():

    if row.bd_star == 0:

        lc = pd.read_csv(dir + Configuration.FIELD + "_" + str(star_list.loc[idx].source_id) + ".lc", sep=' ')
        ok_data = sc(lc.mag, sigma=3)
        olc = pd.read_csv(odir + Configuration.FIELD + "_" + str(star_list.loc[idx].source_id) + ".lc", sep=' ')
        lc['phase'] = (lc.jd - lc.jd.min()) / 0.371452 % 1
        lc['flux'] = 10**((lc.mag-2.5*np.log10(300.)-25)/-2.5)
        lc['tot_flux'] = lc.flux - ((lc.sky + lc.bkg) * np.pi * Configuration.APER_SIZE ** 2)

        # get the similar magnitude bright stars
        star_list['dmag'] = np.abs(row.master_mag - star_list['master_mag'])
        star_list['dist'] = np.sqrt((star_list.y - row.y)**2 + (star_list.x - row.x)**2)
        trend_list = star_list[# (star_list.chip == row.chip) &
                               (star_list.bd_star == 0) &
                               (star_list.dmag > 0) & (star_list.dmag < 2) &
                               # (star_list.dist < 4000) &
                               (star_list.object_type != 'Var') & (star_list.object_type != 'Xray')].copy().reset_index(drop=True)

        cols = {}
        col_nme = []

        kk = 0
        j = np.zeros(len(trend_list))
        k = np.zeros(len(trend_list))
        l = np.zeros(len(trend_list))

        for idy, rw in trend_list.iterrows():

            tr = pd.read_csv(dir + Configuration.FIELD + "_" + str(rw.source_id) + ".lc", sep=' ')
            j[idy], k[idy], l[idy] = jstet(tr[~ok_data.mask].mag.to_numpy(), tr[~ok_data.mask].err.to_numpy())

            # tr['flux'] = 10 ** ((tr.mag - 2.5 * np.log10(300.) - 25) / -2.5)
            # tr['nmag'] = 25 - 2.5 * np.log10((tr.flux - (tr.bkg * np.pi * Configuration.APER_SIZE ** 2)) / 300.)

            cols['mag_'+str(kk)] = tr.mag.to_numpy() - tr.mag.median()
            col_nme.append('mag_'+str(kk))
            kk = kk + 1

        _, jm, js = scs(j, sigma=1)
        trs = pd.DataFrame(cols)
        dys = np.unique(lc.jd.to_numpy().astype(int))

        off = np.zeros(len(lc))

        for idy, t in enumerate(ok_data.mask):
            if not t:
                _, off[idy], _ = scs(trs.loc[idy, col_nme], sigma=1)

        sp = np.zeros(len(trend_list))

        for dy in dys:
            dd = (lc.jd.astype(int) == dy) & (~ok_data.mask)
            plt.scatter(lc.jd[dd], lc.mag[dd], marker='.', c='k')
            scld = np.zeros((len(lc.jd[dd]), len(trend_list)))
            gd = []
            for idy, col in enumerate(col_nme):
                sp[idy], _ = spearmanr(cols[col][dd], lc.mag[dd].to_numpy())
                if np.abs(sp[idy]) > 0.3:
                    gd.append(idy)
                    if sp[idy] > 0:
                        scld[:,idy] = ((cols[col][dd] - np.min(cols[col][dd])) * (np.max(lc.mag[dd]) - np.min(lc.mag[dd]))) / (np.max(cols[col][dd]) - np.min(cols[col][dd]))
                    else:
                        scld[:,idy] = ((cols[col][dd] - np.max(cols[col][dd])) * (np.max(lc.mag[dd]) - np.min(lc.mag[dd]))) / (
                                    np.min(cols[col][dd]) - np.max(cols[col][dd]))
                    plt.plot(lc.jd[dd], cols[col][dd] + lc.mag[dd].mean(), marker='x', c='r', alpha=0.3)
                    plt.plot(lc.jd[dd], scld[:,idy] + lc.mag[dd].min(), marker='x', c='g', alpha=0.3)
            fin_scl = np.median(scld[:, gd], axis=1)
            plt.plot(lc.jd[dd], fin_scl + lc.mag[dd].min(), marker='o', c='r')
            _, _, new = scs(lc.mag[dd] - fin_scl, sigma=2.5)
            _, _, old = scs(lc.mag[dd] - off[dd], sigma=2.5)
            print(new, old, row.master_flux_er / row.master_flux)
            plt.show()

            print('hold')
            # xx = trs[col_nme][dd]
            # xx = lc[['trd']][dd]
            # yy = lc[dd].mag.to_numpy()

            # model = LinearRegression().fit(xx, yy).predict(xx)

            # plt.scatter(lc[dd].jd, lc[dd].mag, c='k')
            # plt.scatter(lc[dd].jd, off[dd] + lc[dd].mag.median(), c='g', marker='x')
            # plt.scatter(lc[dd].jd, model, c='red', marker='x')

            # print(np.std(lc[dd].mag - off[dd]), np.std(lc[dd].mag - model), row.master_flux_er / row.master_flux)
            # plt.scatter(lc[dd].jd, model, c='g', marker='x')
            # plt.show()


        # plt.scatter(lc[~ok_data.mask].phase, lc[~ok_data.mask].mag - off[~ok_data.mask], c='k', marker='.')
        # plt.scatter(lc[~ok_data.mask].phase, lc[~ok_data.mask].mag - lc[~ok_data.mask].trd + 0.2, c='r', marker='.')
        # plt.show()
        print('hold')
