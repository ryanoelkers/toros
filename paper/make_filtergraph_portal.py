import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from config import Configuration
import numpy as np
tv_zpt = 5.4
xcen_47tuc = 6853
ycen_47tuc = 5375
rad_47tuc = 270

# get the varstats file
data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt",
                       delimiter=' ', low_memory=False)
varstats['v'] = varstats.apply(lambda x: x.mag - tv_zpt, axis=1)
varstats = varstats[varstats.cat_source == 'toros'].copy().reset_index(drop=True)
varstats['toros_num'] = ''
for idx, row in varstats.iterrows():
    if row.star_id < 10:
        varstats.loc[idx, 'toros_num'] = '00000' + str(row.star_id)
    elif (row.star_id >= 10) & (row.star_id < 100):
        varstats.loc[idx, 'toros_num'] = '0000' + str(row.star_id)
    elif (row.star_id >= 100) & (row.star_id < 1000):
        varstats.loc[idx, 'toros_num'] = '000' + str(row.star_id)
    elif (row.star_id >= 1000) & (row.star_id < 10000):
        varstats.loc[idx, 'toros_num'] = '00' + str(row.star_id)
    elif (row.star_id >= 10000) & (row.star_id < 100000):
        varstats.loc[idx, 'toros_num'] = '0' + str(row.star_id)
    else:
        varstats.loc[idx, 'toros_num'] = str(row.star_id)

varstats['toros_id'] = varstats.apply(lambda x: x.toros_field_id + '_' + x.toros_num, axis=1)
varstats['figure'] = 'Figure File'
varstats['lightcurve'] = 'Light Curve File'
varstats['Simbad'] = 'Simbad Link'
varstats['Aladin'] = 'Aladin Link'

varstats = varstats[['toros_id', 'figure', 'lightcurve', 'Simbad', 'Aladin', 'source_id',
                     'ra', 'dec', 'xcen', 'ycen', 'chip',
                     'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                     'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
                     'v', 'rms', 'min_rms', 'full_rms', 'd90', 'jstet', 'lstet', 'prd', 'pwr', 'fap', 'out_mag_std',
                     'G47T', 'N121', 'lsst', 'edge', 'simp', 'prnk', 'pnts', 'otlr', 'grps', 'cntm',
                     'var_pass', 'per_pass', 'day_pass']].copy().reset_index(drop=True)

varstats.rename(columns={'source_id': 'alternate_id', 'xcen': 'x_pixel', 'ycen': 'y_pixel',
                         'phot_g_mean_mag': 'G', 'phot_bp_mean_mag': 'Bp', 'phot_rp_mean_mag': 'Rp',
                         'v': 'Tv', 'jstet': 'J', 'lstet': 'L', 'prd': 'period', 'pwr': 'power', 'fap': 'FAP',
                         'out_mag_std': 'N_D', 'lsst':'LSST', 'edge':'EDGE', 'simp': 'SIMP',
                         'prnk': 'PRNK', 'pnts':'PNTS', 'otlr': 'OTLR', 'grps': 'GRPS', 'ctnm': 'CNTM'}, inplace=True)

varstats.to_csv('filtergraph_release.csv', sep=',', header=True, index=False)

