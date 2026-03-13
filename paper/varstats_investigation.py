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
errors = pd.read_csv(Configuration.ONE_DRIVE + 'varstats\\' + Configuration.FIELD + '_errors.txt',
                         delimiter=' ',
                         names=['name', 'mag', 'rms', 'erms'],
                         low_memory=False)
errors['gc_star'] = star_list['gc_star'].to_numpy()
errors['var_type'] = star_list['var_type'].to_numpy()

# read in the varstats file
varstats = pd.read_csv(Configuration.ONE_DRIVE + 'varstats\\' + Configuration.FIELD + '_varstats.txt',
                       delimiter=' ',
                       header=0,
                       low_memory=False)
varstats['gc_star'] = star_list['gc_star'].to_numpy()
varstats['var_type'] = star_list['var_type'].to_numpy()
varstats['x'] = star_list['xcen'].to_numpy()
varstats['y'] = star_list['ycen'].to_numpy()

nonvars = varstats[(varstats.gc_star == 0) & (varstats.rms < 0.2) &
                   (varstats.rms > 0) & (varstats.var_type == '--')].copy().reset_index(drop=True)

# the calculated zeropoints
zpt_gg = 5.53
zpt_g = 5.15
zpt_r = 5.55
zpt_i = 5.62

# Get the J\\L cutoff
cnts, binns = np.histogram(nonvars.Jstet.to_numpy() / nonvars.Lstet.to_numpy(),
                           bins=np.around(np.sqrt(len(nonvars)), decimals=0).astype(int))

# get the sigma, mean, median from 3 sigma clipping
comp_mean, comp_median, comp_std = scs(nonvars.Jstet.to_numpy() /
                                       nonvars.Lstet.to_numpy(),
                                       sigma=3)
comp_max = binns[np.argmax(cnts)]
var_metric_cutoff = comp_max + 2 * comp_std

# plt.figure(figsize=(9, 6))
# plt.hist(nonvars.Jstet.to_numpy() \\ nonvars.Lstet.to_numpy(),
#          bins=np.around(np.sqrt(len(nonvars)), decimals=0).astype(int),
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
# plt.ylim([0, 7000])
# plt.yticks(fontsize=15)
# plt.show()
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

    lc = pd.read_csv(Configuration.ONE_DRIVE + "lc\\" + Configuration.FIELD + "\\" + row['name'], sep=' ', header=0)
    ok_data = sc(lc.mag, sigma=3)
    ok_data.mask[np.argwhere(lc.mag == 0)] = True
    ok_data.mask[np.argwhere(lc.mag > 25)] = True

    lc = lc[~ok_data.mask].copy().reset_index(drop=True)
    lc['days'] = lc.jd.to_numpy().astype('int')
    dys = lc['days'].unique()
    lc_agg = lc.groupby('days')['mag'].agg({'mean', 'count'}).reset_index()
    mn, mdn, sg = scs(lc_agg['mean'], sigma=3)

    chi2 = np.abs(lc_agg['mean'].to_numpy() - mn) / sg
    cntdy = lc_agg['count'].to_numpy()
    
    if len(chi2[(chi2 > 3) & (cntdy > 5)] > 0):
        if (np.argwhere(chi2 > 3)[0] == 1) & (row.mag > 16):
            plt.figure(figsize=(9,6))
            plt.errorbar(lc.jd - 2460580, lc.mag, yerr=lc.err, c='k', fmt='none')
            plt.scatter(lc.jd - 2460580, lc.mag, c='k', marker='.')
            plt.xlabel('JD - 2460580 [d]', fontsize=20)
            plt.ylabel(r'$T_G$', fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.ylim([16.7, 16.51])
            cut_lc = lc[lc['days'].isin(dys[chi2 > 3])]
            plt.errorbar(cut_lc.jd - 2460580, cut_lc.mag, yerr=cut_lc.err, c='r', fmt='none')
            plt.scatter(cut_lc.jd - 2460580, cut_lc.mag, c='r', marker='.')
            # plt.arrow(cut_lc.jd.mean() - 2460580, cut_lc.mag.mean() + 2 * sg, 0, sg, color='r', linewidth=2)
            # plt.gca().invert_yaxis()
            plt.savefig("aas_flare.png", dpi=200, bbox_inches='tight')
            plt.show()
            plt.close()

            plt.figure(figsize=(9,6))

            plt.xlabel('JD - 2460580 [d]', fontsize=20)
            plt.ylabel(r'$T_G$', fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            cut_lc = lc[lc['days'].isin(dys[chi2 > 3])]
            plt.errorbar(cut_lc.jd - 2460580, cut_lc.mag, yerr=cut_lc.err, c='r', fmt='none')
            plt.scatter(cut_lc.jd - 2460580, cut_lc.mag, c='r')
            plt.ylim([16.69, 16.56])
            plt.savefig("aas_flare_zoom.png", dpi=200, bbox_inches='tight')
            plt.show()
            plt.close()


    # # loop through the various identified periods
    # if (row.p1 < 51) & (row.fap1 < 0.01):
    #     n_allies = 1 - len(periods[periods == row.p1]) / np.max(cnts)
    #     n_powers = 1 - len(powers[(periods == row.p1) & (powers > row.pwr1)]) / len(powers[periods == row.p1])
    #
    #     p1_pass = np.around(n_powers * n_allies, decimals=3)
    #
    # if (row.p2 < 51) & (row.fap2 < 0.01):
    #     n_allies = 1 - len(periods[periods == row.p2]) / np.max(cnts)
    #     n_powers = 1 - len(powers[(periods == row.p2) & (powers > row.pwr2)]) / len(powers[periods == row.p2])
    #
    #     p2_pass = np.around(n_powers * n_allies, decimals=3)
    #
    # if (row.p3 < 51) & (row.fap3 < 0.01):
    #     n_allies = 1 - len(periods[periods == row.p3]) / np.max(cnts)
    #     n_powers = 1 - len(powers[(periods == row.p3) & (powers > row.pwr3)]) / len(powers[periods == row.p3])
    #
    #     p3_pass = np.around(n_powers * n_allies, decimals=3)
    #
    # if (row.p4 < 51) & (row.fap4 < 0.01):
    #     n_allies = 1 - len(periods[periods == row.p4]) / np.max(cnts)
    #     n_powers = 1 - len(powers[(periods == row.p4) & (powers > row.pwr4)]) / len(powers[periods == row.p4])
    #
    #     p4_pass = np.around(n_powers * n_allies, decimals=3)
    #
    # if (row.p5 < 51) & (row.fap5 < 0.01):
    #     n_allies = 1 - len(periods[periods == row.p5]) / np.max(cnts)
    #     n_powers = 1 - len(powers[(periods == row.p5) & (powers > row.pwr5)]) / len(powers[periods == row.p5])
    #
    #     p5_pass = np.around(n_powers * n_allies, decimals=3)
