import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from config import Configuration
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip as sc


def clipped_median(x, sigma=3):
    mean, median, std = sigma_clipped_stats(x, sigma=sigma)
    return median

def clipped_std(x, sigma=3):
    mean, median, std = sigma_clipped_stats(x, sigma=sigma)
    return std


data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"
tv_zpt = 5.4

# read in the varstats file, and exclude the LSST variables for now
fullstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
fullstats['v'] = fullstats.mag - tv_zpt

lsststats = fullstats[(fullstats.cat_source == 'lsst') &
                      (fullstats.day_pass == 1) &
                      (fullstats.v > 17) &
                      (fullstats.G47T == 0)].copy().reset_index(drop=True)

for idx, row in lsststats.iterrows():

    if row.chip < 10:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                         "lsst/fin/" +
                         "0" + str(row.chip) + "/" +
                         Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                         sep=" ")
    else:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                         "lsst/fin/" +
                         str(row.chip) + "/" +
                         Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                         sep=" ")

    lc['dys'] = lc.jd.to_numpy().astype('int')
    dys = lc.dys.unique()
    agg_lc = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg(mean_mag=('mag', clipped_median),
                                                                std_mag=('mag', clipped_std),
                                                                mag05=('mag', ('q05', lambda x: x.quantile(0.05))),
                                                                mag95=('mag', ('q95', lambda x: x.quantile(0.95))),
                                                                total_obs=('mag', 'count')).reset_index()

    agg_lc = agg_lc[agg_lc.total_obs >=6].copy().reset_index(drop=True)
    # get the index of the clipped mean magnitudes
    clip_mag = sc(
        agg_lc.mean_mag.to_numpy() - agg_lc.mean_mag.mean(),
        sigma=3.0,
        masked=True)
    clip_mag_mask = clip_mag.mask

    # get the index of the clipp std magnitudes
    clip_std = sc(
        agg_lc.std_mag.to_numpy() - agg_lc.std_mag.mean(),
        sigma=3.0,
        masked=True)
    clip_std_mask = clip_std.mask

    # they are both out of bounds
    arg_dx = np.argwhere(clip_mag.mask & clip_std.mask).flatten()
    dy = agg_lc.loc[arg_dx[0]].dys
    tme = lc[lc.dys == dy].jd.mean()

    lc = lc[(lc.mag > 0) & (lc.err < 2)].copy().reset_index(drop=True)

    plt.title(str(row.source_id) + ' ' + str(row.chip) + ' ' +
              str(row.out_mag_std) + ' ' + str(tme))
    plt.errorbar(lc.jd, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k')
    plt.scatter(lc.jd, lc.mag - tv_zpt, marker='.', c='k')

    plt.annotate("", xy=(tme, lc.mag.min() - tv_zpt),
                 xytext=(tme, lc.mag.min() - 1 - tv_zpt),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))

    plt.gca().invert_yaxis()
    plt.show()