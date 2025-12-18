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
star_list = pd.read_csv(Configuration.ONE_DRIVE + 'master\\' + Configuration.FIELD + '\\' + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)
star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

# read in the uncertainties file
error_list = pd.read_csv(Configuration.ONE_DRIVE + 'varstats\\' + Configuration.FIELD + '_errors.txt',
                         delimiter=' ',
                         header=0,
                         low_memory=False)

# read in the varstats file
varstats = pd.read_csv(Configuration.ONE_DRIVE + 'varstats\\' + Configuration.FIELD + '_varstats.txt',
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
var_metric_cutoff = comp_max + 2 * comp_std

# plt.figure(figsize=(9, 6))
# plt.hist(varstats.Jstet[varstats.Jstet > 0].to_numpy() / varstats[varstats.Jstet > 0].Lstet.to_numpy(),
#          bins=np.around(np.sqrt(len(varstats[varstats.Jstet > 0])), decimals=0).astype(int),
#          histtype='step', color='k', linewidth=3, align='left')
# plt.plot((np.around(comp_max + 2 * comp_std, decimals=2), np.around(comp_max + 2 * comp_std, decimals=2)),
#          (0, 17000),
#          color='r',
#          linewidth = 3)
# plt.text(np.around(comp_max + 2 * comp_std, decimals=2) + 0.5, np.max(cnts) * 0.75, r'$\frac{J_S}{L_S} >$' +
#          str(np.around(comp_max + 2 * comp_std, decimals=2)), fontsize=20)
# plt.xlabel(r'$\frac{J_S}{L_S}$', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylabel('Count', fontsize=20)
# plt.xlim([0.74, 14.5])
# plt.ylim([0, 17000])
# plt.yticks(fontsize=15)
# # plt.show()
# plt.close()

# get the periodicity cutoffs
cols_to_combine = ['p1', 'p2', 'p3', 'p4', 'p5']
periods = varstats[cols_to_combine].to_numpy().flatten()
cols_to_combine = ['pwr1', 'pwr2', 'pwr3', 'pwr4', 'pwr5']
powers = varstats[cols_to_combine].to_numpy().flatten()
cols_to_combine = ['fap1', 'fap2', 'fap3', 'fap4', 'fap5']
faps = varstats[cols_to_combine].to_numpy().flatten()

# make a plot showing the aliasing that occurs
prds, cnts = np.unique(periods, return_counts=True)
aap = cnts / np.max(cnts)  # normalize by the maximum duplicated period

# plt.figure(figsize=(9, 6))
#
# plt.plot(prds, cnts,
#          color='k', linewidth = 2)
# plt.xlabel('Period [log(d)]', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylabel('Count', fontsize=20)
# plt.xlim([0.05, 51])
# plt.xscale('log')
# plt.yticks(fontsize=15)
# # plt.show()
# plt.close()

# now we loop through the variables only selecting the most variable objects and the strongest periods
for idx, row in varstats.iterrows():

    # set the flags to pass the variability of the star
    var_pass = 0
    p1_pass = 0
    p2_pass = 0
    p3_pass = 0
    p4_pass = 0
    p5_pass = 0

    # determine basic variability testing
    var_metric = row.Jstet / row.Lstet
    if row['name'] == 'FIELD_0e.001_AQ_Tuc.lc':
        print('hold')
    if var_metric > var_metric_cutoff:
        lc = pd.read_csv(Configuration.ONE_DRIVE + "lc\\" + Configuration.FIELD + "\\" + row['name'], sep=' ', header=0)
        ok_data = sc(lc.mag, sigma=3)
        ok_data.mask[np.argwhere(lc.mag < 0)] = True
        ok_data.mask[np.argwhere(lc.mag > 25)] = True

        plt.scatter(lc[~ok_data.mask].jd, lc[~ok_data.mask].mag, c='k')
        plt.gca().invert_yaxis()
        plt.show()

    # loop through the various identified periods
    if (row.p1 < 51) & (row.fap1 < 0.01):
        n_allies = 1 - len(periods[periods == row.p1]) / np.max(cnts)
        n_powers = 1 - len(powers[(periods == row.p1) & (powers > row.pwr1)]) / len(powers[periods == row.p1])

        p1_pass = np.around(n_powers * n_allies, decimals=3)

    if (row.p2 < 51) & (row.fap2 < 0.01):
        n_allies = 1 - len(periods[periods == row.p2]) / np.max(cnts)
        n_powers = 1 - len(powers[(periods == row.p2) & (powers > row.pwr2)]) / len(powers[periods == row.p2])

        p2_pass = np.around(n_powers * n_allies, decimals=3)

    if (row.p3 < 51) & (row.fap3 < 0.01):
        n_allies = 1 - len(periods[periods == row.p3]) / np.max(cnts)
        n_powers = 1 - len(powers[(periods == row.p3) & (powers > row.pwr3)]) / len(powers[periods == row.p3])

        p3_pass = np.around(n_powers * n_allies, decimals=3)

    if (row.p4 < 51) & (row.fap4 < 0.01):
        n_allies = 1 - len(periods[periods == row.p4]) / np.max(cnts)
        n_powers = 1 - len(powers[(periods == row.p4) & (powers > row.pwr4)]) / len(powers[periods == row.p4])

        p4_pass = np.around(n_powers * n_allies, decimals=3)

    if (row.p5 < 51) & (row.fap5 < 0.01):
        n_allies = 1 - len(periods[periods == row.p5]) / np.max(cnts)
        n_powers = 1 - len(powers[(periods == row.p5) & (powers > row.pwr5)]) / len(powers[periods == row.p5])

        p5_pass = np.around(n_powers * n_allies, decimals=3)
