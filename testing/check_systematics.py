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
star_list = pd.read_csv("/Users/oelkerrj/OneDrive - The University of Texas-Rio Grande Valley/Research/TOROS/master/"
                        + Configuration.FIELD + "_star_list_updated.txt", sep=' ', low_memory=False, index_col=0)
star_list['bd_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

dir = "/Users/oelkerrj/OneDrive - The University of Texas-Rio Grande Valley/Research/TOROS/lc/"
all_files = np.sort(Utils.get_file_list(dir, '.lc'))

for idx, row in star_list[25706:].iterrows():

    if row.bd_star == 0:
        lc = pd.read_csv("/Users/oelkerrj/OneDrive - The University of Texas-Rio Grande Valley/Research/TOROS/lc/"
                         + Configuration.FIELD + "/" +
                         Configuration.FIELD + "_" + str(star_list.loc[idx].source_id) + ".lc", sep=' ')
        ok_data = sc(lc.mag, sigma=3)

        # get the similar magnitude bright stars
        star_list['dmag'] = np.abs(star_list['master_mag'] - row.master_mag)
        trend_list = star_list[(star_list.dmag > 0) & (star_list.dmag < .01)].copy()

        trs = pd.DataFrame()
        cols = []
        spr = np.zeros(len(trend_list))
        kk = 0
        for idy, rw in trend_list.iterrows():
            if idy < len(all_files):
                tr = pd.read_csv(dir + all_files[rw.star_id], sep=' ')
                spr[kk], _ = spearmanr(lc[~ok_data.mask].mag, tr[~ok_data.mask].mag)
                cols.append('mag_' + str(kk))
                trs['mag_' + str(kk)] = tr.mag.to_numpy()

            kk = kk + 1
        dys = np.unique(lc.jd.to_numpy().astype(int))

        lc['trd'] = np.zeros(len(lc))
        ok = np.argsort(spr)[-10:]
        ok_trs = ['mag_' + o for o in ok.astype(str)]
        for dy in dys:
            model = (LinearRegression().fit(trs.loc[(~ok_data.mask) & (lc.jd.astype(int) == dy), ok_trs],
                                           lc.loc[(~ok_data.mask) & (lc.jd.astype(int) == dy)].mag).
                     predict(trs.loc[(~ok_data.mask) & (lc.jd.astype(int) == dy), ok_trs]))
            lc.loc[(~ok_data.mask) & (lc.jd.astype(int) == dy), 'trd'] = model

        print('hold')
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