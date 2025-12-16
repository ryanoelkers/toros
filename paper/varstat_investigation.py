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
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats as scs
from astropy.stats import sigma_clip as sc
import numpy as np
import statistics

# read in the star list
star_list = pd.read_csv(Configuration.ONE_DRIVE + 'master/' + Configuration.FIELD + '/' + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)
star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

# read in the uncertainties file
error_list = pd.read_csv(Configuration.ONE_DRIVE + 'varstats/' + Configuration.FIELD + '_errors.txt',
                         delimiter=' ',
                         header=0,
                         low_memory=False)

# read in the varstats file
vars_list = pd.read_csv(Configuration.ONE_DRIVE + 'varstats/' + Configuration.FIELD + '_varstats.txt',
                         delimiter=' ',
                         header=0,
                         low_memory=False)

# the calculated zeropoints
zpt_gg = 5.53
zpt_g = 5.15
zpt_r = 5.55
zpt_i = 5.62

# get the cut-off in Jstet coordinates
cnts, binns = np.histogram(vars_list.Jstet.to_numpy(), bins=np.around(np.sqrt(len(vars_list)), decimals=0).astype(int))

plt.figure(figsize=(9, 6))
plt.hist(vars_list.Jstet.to_numpy(), bins=np.around(np.sqrt(len(vars_list)), decimals=0).astype(int),
         histtype='step', color='k', linewidth=3, align='left')

plt.xlabel(r'$J_S$', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Count', fontsize=20)
plt.xlim([-100, 1000])
plt.yticks(fontsize=15)
plt.show()
plt.close()
print('hold')