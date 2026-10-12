import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import pandas as pd
from config import Configuration
from libraries.utils import Utils
import matplotlib.colors as colors
data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt",
                       delimiter=' ', low_memory=False)

df = varstats[['star_id', 'cat_source', 'prd', 'v', 'rms', 'd90', 'jstet', 'lstet',
               'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'var_pass','per_pass', 'day_pass', 'parallax',
               'parallax_error']].copy()

df.to_csv('df_filtergraph.csv', sep=',')
# plot LSST variables that are in Gaia
toros_gaia = varstats[(varstats.cat_source =='toros') &
                      (varstats.parallax_error / varstats.parallax < 0.1) &
                      (varstats.prd > 0)].copy().reset_index(drop=True)
toros_gaia['abs_g'] = toros_gaia.phot_g_mean_mag - 5 * np.log10(1000. / toros_gaia.parallax) + 5
toros_gaia['bmp'] = toros_gaia.phot_bp_mean_mag - toros_gaia.phot_rp_mean_mag
