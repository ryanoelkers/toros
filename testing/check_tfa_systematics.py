import pandas as pd
import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from config import Configuration
import numpy as np
from astropy.stats import sigma_clipped_stats as scs
from scipy.stats import spearmanr
from astropy.stats import sigma_clip as sc
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)

star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)
chk_list = star_list[star_list.chip == 1].copy().reset_index(drop=True)

dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001_cln/"
all_files = np.sort(Utils.get_file_list(dir, '.lc'))

# 28529
for idx, row in star_list[28529:].iterrows():

    if row.gc_star == 0:

        lc = pd.read_csv(dir + Configuration.FIELD + "_" + str(star_list.loc[idx].source_id) + ".lc", sep=' ')
        ok_data = sc(lc.mag, sigma=3)

        lc['day'] = lc.jd.astype(int)
        n_tr_stars = 100

        # get the similar magnitude bright stars
        star_list['dmag'] = np.abs(row.master_mag - star_list['master_mag'])
        star_list['dist'] = np.sqrt((star_list.y - row.y) ** 2 + (star_list.x - row.x) ** 2)
        trend_list = star_list[(star_list.chip == row.chip) &
                               (star_list.gc_star == 0) &
                               (star_list.dmag > 0) & (star_list.dmag < 2) &
                               (star_list.dist > Configuration.APER_SIZE) &
                               (star_list.object_type == 'Star')].copy().sort_values(by='master_mag')[0:n_tr_stars].reset_index(drop=True)

        cols = {}
        col_nme = []
        dys = np.unique(lc.jd.to_numpy().astype(int))
        kk = 0
        for idy, rw in trend_list.iterrows():
            tr = pd.read_csv(dir + Configuration.FIELD + "_" + str(rw.source_id) + ".lc", sep=' ')
            if len(tr[tr.mag > 0]) >= len(tr[~ok_data.mask]):
                _, mdn_mag, _ = scs(tr[tr.mag > 0].mag, sigma=2)
                cols['mag_' + str(kk)] = tr.mag.to_numpy() - mdn_mag
                col_nme.append('mag_'+str(kk))
                kk = kk + 1

        n_tr_stars = kk
        df = pd.DataFrame(cols, columns=col_nme)

        # for dy in dys:
        #     dd = (lc.jd.astype(int) == dy) & (~ok_data.mask)
        #
        #     gjk = np.zeros((n_tr_stars, n_tr_stars))
        #
        #     yhat = lc[dd].mag - lc[dd].mag.median()
        #
        #     for idj, xj in enumerate(col_nme):
        #         gjk[idj] = df[dd].mul(df.loc[dd, xj], axis='index').sum(axis=0).to_numpy()
        #
        #     inv_gjk = np.linalg.inv(gjk)
        #
        #     hj = df[dd].mul(yhat, axis='index').sum(axis=0).to_numpy()
        #     cj = np.sum(inv_gjk * hj, axis=1)
        #
        #     lc.loc[dd, 'trd'] = np.sum(cj * df.loc[dd], axis=1)

        dd = ~ok_data.mask

        gjk = np.zeros((n_tr_stars, n_tr_stars))

        yhat = lc[dd].mag - lc[dd].mag.mean()

        for idj, xj in enumerate(col_nme):
            gjk[idj] = df[dd].mul(df.loc[dd, xj], axis='index').sum(axis=0).to_numpy()

        inv_gjk = np.linalg.inv(gjk)

        hj = df[dd].mul(yhat, axis='index').sum(axis=0).to_numpy()
        cj = np.sum(inv_gjk * hj, axis=1)

        lc.loc[dd, 'trd'] = np.sum(cj * df.loc[dd], axis=1)

        lc['ph'] = (lc.jd - lc.jd.min()) / 0.37143 % 1

        plt.subplot(2, 1, 1)
        _, _, rms = scs(lc[~ok_data.mask].mag - lc[~ok_data.mask].trd, sigma=3)
        plt.title(str(np.around(rms, decimals=4)) + ' ' + str(np.around(lc[~ok_data.mask].err.mean(), decimals=4)))
        plt.scatter(lc[~ok_data.mask].ph, lc[~ok_data.mask].mag, c='k')
        # plt.plot(lc[~ok_data.mask].jd, lc[~ok_data.mask].trd + lc[~ok_data.mask].mag.median(), c='r')

        plt.subplot(2, 1, 2)
        plt.errorbar(lc[~ok_data.mask].ph, lc[~ok_data.mask].mag - lc[~ok_data.mask].trd, yerr=lc[~ok_data.mask].err, c='k', fmt='none')
        plt.scatter(lc[~ok_data.mask].ph, lc[~ok_data.mask].mag - lc[~ok_data.mask].trd, c='k')


        plt.show()

