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
varstats = pd.read_csv(Configuration.ONE_DRIVE + 'varstats/' + Configuration.FIELD + '_varstats.txt',
                       delimiter=' ',
                       header=0,
                       low_memory=False)

# the calculated zeropoints
zpt_gg = 5.53
zpt_g = 5.15
zpt_r = 5.55
zpt_i = 5.62

# Get the J/L cutoff
cnts, binns = np.histogram(varstats[varstats.Jstet > 0].Jstet.to_numpy() / varstats[varstats.Jstet > 0].Lstet.to_numpy(),
                           bins=np.around(np.sqrt(len(varstats[varstats.Jstet > 0])), decimals=0).astype(int))

# get the sigma, mean, median from 3 sigma clipping
comp_mean, comp_median, comp_std = scs(varstats[varstats.Jstet > 0].Jstet.to_numpy() /
                                       varstats[varstats.Jstet > 0].Lstet.to_numpy(),
                                       sigma=3)
comp_max = binns[np.argmax(cnts)]

plt.figure(figsize=(9, 6))
plt.hist(varstats.Jstet[varstats.Jstet > 0].to_numpy() / varstats[varstats.Jstet > 0].Lstet.to_numpy(),
         bins=np.around(np.sqrt(len(varstats[varstats.Jstet > 0])), decimals=0).astype(int),
         histtype='step', color='k', linewidth=3, align='left')
plt.plot((np.around(comp_max + 2 * comp_std, decimals=2), np.around(comp_max + 2 * comp_std, decimals=2)),
         (0, 17000),
         color='r',
         linewidth = 3)
plt.text(np.around(comp_max + 2 * comp_std, decimals=2) + 0.5, np.max(cnts) * 0.75, r'$\frac{J_S}{L_S} >$' +
         str(np.around(comp_max + 2 * comp_std, decimals=2)), fontsize=20)
plt.xlabel(r'$\frac{J_S}{L_S}$', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Count', fontsize=20)
plt.xlim([0.74, 14.5])
plt.ylim([0, 17000])
plt.yticks(fontsize=15)
# plt.show()
plt.close()

# get the periodicity cutoffs
cols_to_combine = ['p1', 'p2', 'p3', 'p4', 'p5']
periods = varstats[cols_to_combine].to_numpy().flatten()
prds, cnts = np.unique(periods, return_counts=True)
aap = cnts / np.max(cnts)  # normalize by the maximum duplicated period

# loop through stars to see if they have good periodicities
for idx, row in varstats.iterrows():
    # period 1
    if (aap[prds == row.p1] <= 0.01) & (row.fap1 <= 0.01):
        lc = pd.read_csv(Configuration.ONE_DRIVE + 'lc/' + Configuration.FIELD + '/' + row['name'], sep=' ', header=0)
        ok_data = sc(lc.mag, sigma=3)
        lc['ph'] = ((lc.jd - lc.jd.min()) / row.p1) % 1
        plt.scatter(lc[~ok_data.mask].ph, lc[~ok_data.mask].mag, marker='.', c='k')
        plt.title('P=' + str(row.p1) + 'd')
        plt.gca().invert_yaxis()
        plt.show()