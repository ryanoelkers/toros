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

data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"
lsst_dir = "/Users/yuw816/Data/toros/commissioning/lsst/"
tv_zpt = 5.4

# read in the varstats file, and exclude the LSST variables for now
fullstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
fullstats['v'] = fullstats['mag'] - tv_zpt

# get the LSST information
# get the known LSST variables
known = pd.read_csv(lsst_dir + "lsst_data_47tuc_variables.csv",
                    delimiter=',',
                    header=0,
                    low_memory=False,
                    index_col=0)
known_g = known[known.band == 'g'].groupby('diaObjectId').agg(psfFlux_g = ('psfFlux', 'mean')).reset_index()
known_r = known[known.band == 'r'].groupby('diaObjectId').agg(psfFlux_r = ('psfFlux', 'mean')).reset_index()
known_i = known[known.band == 'i'].groupby('diaObjectId').agg(psfFlux_i = ('psfFlux', 'mean')).reset_index()
known_y = known[known.band == 'y'].groupby('diaObjectId').agg(psfFlux_y = ('psfFlux', 'mean')).reset_index()

known_g['g'] = -2.5 * np.log10(known_g.psfFlux_g) + 31.4
known_r['r'] = -2.5 * np.log10(known_r.psfFlux_r) + 31.4
known_i['i'] = -2.5 * np.log10(known_i.psfFlux_i) + 31.4
known_y['y'] = -2.5 * np.log10(known_y.psfFlux_y) + 31.4

merged = (known_g.merge(known_r, on="diaObjectId", how="inner").
          merge(known_i, on="diaObjectId", how="inner").
          merge(known_y, on="diaObjectId", how="inner")
)
# lsst = fullstats[(fullstats.cat_source == 'lsst') & ((fullstats.per_pass == 1) | (fullstats.var_pass == 1) | (fullstats.day_pass == 1))].copy().reset_index(drop=True)
lsst = fullstats[fullstats.cat_source == 'lsst'].copy().reset_index(drop=True)
lsst["source_id"] = lsst["source_id"].astype("int64")
# lsst = lsst[['source_id', 'prd', 'd90', 'rms', 'v', 'per_pass', 'var_pass', 'day_pass', 'jstet', 'lstet']].copy()
lsststats = lsst.merge(merged, left_on='source_id', right_on='diaObjectId', how='inner')

lsststats = lsststats[['star_id', 'prd', 'd90', 'rms', 'jstet', 'lstet', 'out_mag_std', 'v', 'g', 'r', 'i', 'per_pass',
                       'var_pass', 'day_pass', 'cntm', 'prnk', 'otlr', 'grps', 'edge', 'prox', 'G47T', 'N121', 'pvar', 'xray']]
lsststats.to_csv('to_filtergraph.csv', sep=',')
import matplotlib.colors as mcolors
plt.scatter(lsststats.g - lsststats.i, lsststats.g, marker='.', c=lsststats.prd)
plt.xlim([0,2.5])
plt.xlabel('g-i')
plt.ylabel('g')
plt.ylim([15, 22])
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()