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
from sklearn.model_selection import train_test_split
from astropy.timeseries import LombScargle
star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)

star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)
chk_list = star_list[star_list.chip == 1].copy().reset_index(drop=True)

dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/"
all_files = np.sort(Utils.get_file_list(dir, '.lc'))

# 28529
ntr_stars = 100
f = open(Configuration.LIGHTCURVE_DIRECTORY + Configuration.FIELD + '_errors.txt', 'w')
for idx, row in star_list.iterrows():

    lc = pd.read_csv(dir + Configuration.FIELD + "_" + str(star_list.loc[idx].source_id) + ".lc", sep=' ')
    ok_data = sc(lc.mag, sigma=3)
    ok_data.mask[np.argwhere(lc.mag < 0)] = True

    lc['day'] = lc.jd.astype(int)

    # get the similar magnitude bright stars
    star_list['dmag'] = np.abs(row.master_mag - star_list['master_mag'])
    star_list['dist'] = np.sqrt((star_list.y - row.y) ** 2 + (star_list.x - row.x) ** 2)

    if row.gc_star == 0:
        trend_list = star_list[# (star_list.chip == row.chip) &
                               (star_list.gc_star == 0) &
                               # (star_list.dmag > 0) &
                               (star_list.dist > Configuration.APER_SIZE) &
                               (star_list.object_type == 'Star')].copy().sort_values(by='dmag')[0:ntr_stars].reset_index(drop=True)
    else:
        trend_list = star_list[# (star_list.chip == row.chip) &
                               # (star_list.gc_star == 0) &
                               # (star_list.dmag > 0) &
                               (star_list.dist > Configuration.APER_SIZE) &
                               (star_list.object_type == 'Star')].copy().sort_values(by='dmag')[0:ntr_stars].reset_index(drop=True)

    cols = {}
    col_nme = []
    dys = np.unique(lc.jd.to_numpy().astype(int))
    kk = 0
    mst_mag = []
    for idy, rw in trend_list.iterrows():
        tr = pd.read_csv(dir + Configuration.FIELD + "_" + str(rw.source_id) + ".lc", sep=' ')
        if len(tr[tr.mag > 0]) >= len(tr[~ok_data.mask]):
            cols['mag_' + str(kk)] = tr.mag.to_numpy() - tr[tr.mag > 0].mag.mean()
            col_nme.append('mag_'+str(kk))
            kk = kk + 1

    df = pd.DataFrame(cols, columns=col_nme)
    col_nme = np.array(col_nme)

    _, mdn_mag, mn_std_mag = scs(lc[~ok_data.mask].mag, sigma=2.5)

    for dy in dys:
        dd = (lc.jd.astype(int) == dy) & (~ok_data.mask)

        spr = df[dd].corrwith(lc[dd].mag, axis=0).values

        if len(spr[spr > 0.9]) >= 10:
            trd = df[dd][col_nme[spr >= 0.9]].median(axis=1)
        else:
            if len(spr) >= 10:
                trd = df[dd][col_nme[spr >= np.sort(spr)[-10]]].median(axis=1)
            else:
                trd = df[dd][col_nme].median(axis=1)
        lc.loc[dd, 'trd'] = np.around(trd, decimals=6)
        _, _, std_mag = scs(lc[dd].mag - lc[dd].trd, sigma=2.5)
        if (std_mag < mn_std_mag) & (std_mag > 0):
            mn_std_mag = std_mag

    lc = lc.drop(['org_err', 'zpt', 'day'], axis=1)
    lc = lc.rename(columns={'mag': 'raw'})
    lc['mag'] = np.around(lc['raw'] - lc['trd'], decimals=6)
    lc = lc[['jd', 'mag', 'err', 'raw', 'trd', 'sky', 'bkg']]

    line = (Configuration.FIELD + "_" + str(star_list.loc[idx].source_id) + ".lc" + ' ' +
            str(np.around(mdn_mag, decimals=4)) + ' ' + str(np.around(mn_std_mag, decimals=4)) + ' ' +
            str(np.around(lc[~ok_data.mask].err.mean(), decimals=4)) + "\n")
    f.write(line)
    lc.to_csv(Configuration.LIGHTCURVE_DIRECTORY + Configuration.FIELD + '_cln/' +
              Configuration.FIELD + "_" + str(star_list.loc[idx].source_id) + ".lc", index=False, sep=' ')

    if idx % 1000 == 0:
        Utils.log('IDX: ' + str(idx) + ' cleaned. '+ str(len(star_list) - idx) + ' stars remain to be cleaned.', 'info')

f.close()