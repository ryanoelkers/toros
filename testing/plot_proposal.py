import pandas as pd
import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
from config import Configuration
import numpy as np
from astropy.stats import sigma_clipped_stats as scs
from astropy.stats import sigma_clip as sc
from libraries.priority import Priority
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)
plt.figure(figsize=(5, 5))
lc = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY + Configuration.FIELD + '_cln/' + Configuration.FIELD + "_" +
                 str(star_list.loc[28529].source_id) + ".lc", sep=' ')
lc['jd'] = lc['jd'] - 2460584
lc = lc[(lc.jd < 13) | (lc.jd > 15)].copy().reset_index(drop=True)

clp = sc(lc.mag.to_numpy(), sigma=2)

ls = LombScargle(lc[~clp.mask].jd.to_numpy(),
                 lc[~clp.mask].mag.to_numpy(),
                 dy=lc[~clp.mask].err.to_numpy())

frequency, power = ls.autopower()
period = 1./frequency[np.argmax(power)]
lc['ph'] = (lc.jd - lc.jd.min()) / period % 1


plt.errorbar(lc[~clp.mask].ph, lc[~clp.mask].mag - lc[~clp.mask].mag.mean() + star_list.loc[28529].phot_g_mean_mag,
             yerr=lc[~clp.mask].err, c='k', fmt='none')
plt.scatter(lc[~clp.mask].ph, lc[~clp.mask].mag - lc[~clp.mask].mag.mean() + star_list.loc[28529].phot_g_mean_mag,
            marker='.', c='k')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('T$_G$', fontsize=15)
plt.xlabel('Phase', fontsize=15)
plt.ylim([14.3, 13.6])
plt.xlim([0, 1])
plt.savefig('/Users/yuw816/Development/toros/testing/short_term.png', bbox_inches='tight', dpi=200)
# plt.show()
plt.close()

plt.figure(figsize=(5, 5))

lc = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY + Configuration.FIELD + '_cln/' + Configuration.FIELD + "_" +
                 str(star_list.loc[901].source_id) + ".lc", sep=' ')
lc['jd'] = lc['jd'] - 2460584
lc = lc[(lc.jd < 13) | (lc.jd > 15)].copy().reset_index(drop=True)

clp = sc(lc.mag.to_numpy(), sigma=3)

ls = LombScargle(lc[~clp.mask].jd.to_numpy(),
                 lc[~clp.mask].mag.to_numpy(),
                 dy=lc[~clp.mask].err.to_numpy())

frequency, power = ls.autopower()
period = 1./frequency[np.argmax(power)] * 2
lc['ph'] = (lc.jd - lc.jd.min()) / period % 1


plt.errorbar(lc[~clp.mask].jd, lc[~clp.mask].mag - lc[~clp.mask].mag.mean() + star_list.loc[901].phot_g_mean_mag,
             yerr=lc[~clp.mask].err, c='k', fmt='none')
plt.scatter(lc[~clp.mask].jd, lc[~clp.mask].mag - lc[~clp.mask].mag.mean() + star_list.loc[901].phot_g_mean_mag,
            marker='.', c='k')
plt.ylabel('T$_G$', fontsize=15)
plt.xlabel('JD - 2460584 [d]', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()
plt.savefig('/Users/yuw816/Development/toros/testing/long_term.png', bbox_inches='tight', dpi=200)
# plt.show()
plt.close()

mags = np.arange(18, 25, 0.5)
g_lsst = np.zeros(len(mags))
r_lsst = np.zeros(len(mags))
i_lsst = np.zeros(len(mags))
z_lsst = np.zeros(len(mags))

g_toros = np.zeros(len(mags))
r_toros = np.zeros(len(mags))
i_toros = np.zeros(len(mags))
z_toros = np.zeros(len(mags))

for idx, mag in enumerate(mags):
    g_lsst[idx], r_lsst[idx], i_lsst[idx], z_lsst[idx], _ = Priority.signal_to_noise(30., mag, 1, 6.42)
    g_toros[idx], r_toros[idx], i_toros[idx], z_toros[idx], _ = Priority.signal_to_noise(300., mag, 1, 0.61)

plt.figure(figsize=(8.5, 2.125))

plt.plot(mags, g_lsst, c='darkgreen', linestyle='--')
plt.plot(mags, r_lsst, c='r', linestyle='--')
plt.plot(mags, i_lsst, c='maroon', linestyle='--')
plt.plot(mags, z_lsst, c='k', linestyle='--')

plt.plot(mags, g_toros, c='darkgreen', label='g')
plt.plot(mags, r_toros, c='r', label='r')
plt.plot(mags, i_toros, c='maroon', label='i')
plt.plot(mags, z_toros, c='k', label='z')
plt.yscale('log')

plt.xlabel('Magnitude', fontsize=15)
plt.ylabel('Expected SNR', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([18, 21.5])
plt.ylim([9,1000])
# ax2.legend()
plt.savefig('/Users/yuw816/Development/toros/testing/snr_comparison.png', bbox_inches='tight', dpi=200)
plt.show()
#
# fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8), height_ratios=[2,1])
#
# ax1.plot(mags, g_lsst, c='darkgreen', linestyle='--')
# ax1.plot(mags, r_lsst, c='r', linestyle='--')
# ax1.plot(mags, i_lsst, c='maroon', linestyle='--')
# ax1.plot(mags, z_lsst, c='k', linestyle='--')
#
# ax1.plot(mags, g_toros, c='darkgreen', label='g')
# ax1.plot(mags, r_toros, c='r', label='r')
# ax1.plot(mags, i_toros, c='maroon', label='i')
# ax1.plot(mags, z_toros, c='k', label='z')
# ax1.set_yscale('log')
#
# ax1.set_xlabel('Magnitude', fontsize=15)
# ax1.set_ylabel('Expected SNR', fontsize=15)
# ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=12)
# ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=12)
# # ax1.legend()
#
# ax2.plot(mags, g_lsst / g_toros, c='darkgreen', label='g')
# ax2.plot(mags, r_lsst / r_toros, c='r', label='r')
# ax2.plot(mags, i_lsst / i_toros, c='maroon', label='i')
# ax2.plot(mags, z_lsst / z_toros, c='k', label='z')
# ax2.set_xlabel('Magnitude', fontsize=15)
# ax2.set_ylabel('LSST / TOROS [SNR]', fontsize=15)
# ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=12)
# ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=12)
# # ax2.legend()
# plt.savefig('/Users/yuw816/Development/toros/testing/snr_comparison.png', bbox_inches='tight', dpi=200)
# plt.show()