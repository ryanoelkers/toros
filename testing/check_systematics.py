import pandas as pd
import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
from config import Configuration
from astropy.stats import sigma_clipped_stats as scs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

star_list = pd.read_csv("/Users/yuw816/Data/toros/commissioning/master/FIELD_0e.001/"
                        + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)

star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/raw/"
dirold = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001_hold/"
all_files = np.sort(Utils.get_file_list(dir, '.lc'))

img_stats = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_image_stats.txt', sep=' ', index_col=0)
img_stats['bd_day'] = np.where(img_stats.nstars < 20000, 1, 0)

ntr_stars = 500
tt = 0
for idx, row in star_list.iterrows():
    if (row.gc_star == 0) & (row.object_type == 'Var'):
        lc = pd.read_csv(dir + Configuration.FIELD + "_" + str(star_list.loc[idx].source_id) + ".lc", sep=' ')

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
                                   (star_list.gc_star == 1) &
                                   # (star_list.dmag > 0) &
                                   (star_list.dist > Configuration.APER_SIZE) &
                                   (star_list.object_type == 'Star')].copy().sort_values(by='dmag')[0:ntr_stars].reset_index(drop=True)

        cols = {}
        col_nme = []
        dys = np.unique(lc.jd.to_numpy().astype(int))
        mgs = np.zeros(len(trend_list))
        mst_mgs = np.zeros(len(trend_list))

        kk = 0
        mst_mag = []
        for idy, rw in trend_list.iterrows():
            tr = pd.read_csv(dir + Configuration.FIELD + "_" + str(rw.source_id) + ".lc", sep=' ')
            if len(tr[tr.mag > 0]) > 0:
                cols['mag_' + str(kk)] = tr.mag.to_numpy() - tr[tr.mag > 0].mag.median()
                mgs[idy] = tr[tr.mag > 0].mag.median()
                col_nme.append('mag_'+str(kk))
                kk = kk + 1

        df = pd.DataFrame(cols, columns=col_nme)
        col_nme = np.array(col_nme)

        lc['trd'] = 0.
        for ii in range(0, len(lc)):
            offsets = df.loc[ii].to_numpy()
            mg = []
            off = []
            off2 = []
            for jj in np.arange(np.min(mgs), np.max(mgs), 0.1):
                if len(offsets[(mgs >= jj) & (mgs < jj + 0.1) & (offsets > -20)]) > 0:
                    mg.append(jj + 0.05)
                    _, mg_mdn, _ = scs(offsets[(mgs >= jj) & (mgs < jj + 0.1) & (offsets > -20)], sigma=2.5)
                    if np.isnan(mg_mdn):
                        off.append(np.median(offsets[(mgs >= jj) & (mgs < jj + 0.1) & (offsets > -20)]))
                    else:
                        off.append(mg_mdn)
            trd = np.interp(lc.mag[ii], mg, off)
            if np.isnan(trd):
                lc.loc[ii, 'trd'] = np.nanmedian(off)
            else:
                lc.loc[ii, 'trd'] = trd

        lc = lc.rename(columns={'mag': 'raw'})
        lc['mag'] = np.around(lc['raw'] - lc['trd'], decimals=6)
        lc = lc[['jd', 'mag', 'err', 'raw', 'trd', 'sky', 'bkg', 'x', 'y', 'nstars', 'day', 'airmass']]

        xx = lc[['trd']]
        yy = lc['raw']
        reg = LinearRegression().fit(xx, yy).predict(xx)

        _, _, new_std = scs(lc.mag[lc.mag > 0] - lc.trd[lc.mag > 0], sigma =2.5)
        _, _, old_std = scs(lc.mag[lc.mag > 0], sigma=2.5)

