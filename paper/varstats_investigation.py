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
from scipy.stats import median_abs_deviation as mad
import numpy as np
import statistics

tv_zpt = 5.4
xcen_47tuc = 6853
ycen_47tuc = 5375
rad_47tuc = 270

# read in the varstats file
varstats = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + Configuration.FIELD + "_varstats.txt",
                       sep=' ', low_memory=False)
varstats['v'] = varstats['master_mag'] - tv_zpt
varstats['gc_star'] = 0
dist_47tuc = np.sqrt((varstats['xcen'] - xcen_47tuc) ** 2 + (varstats['ycen'] - ycen_47tuc) ** 2)
varstats['gc_star'] = np.where(dist_47tuc < rad_47tuc, 1, 0)

# make the 2 sigma cuts on Jstet and Lstet (these are already appropriately scaled)
mad_jstet = mad(varstats['jstet'])
mdn_jstet = np.median(varstats['jstet'])
mad_lstet = mad(varstats['lstet'])
mdn_lstet = np.median(varstats['lstet'])

jstet_cut = mdn_jstet + 3 * mad_jstet
lstet_cut = mdn_lstet + 3 * mad_lstet

n_pass = len(varstats[(varstats.jstet > jstet_cut) & (varstats.lstet > lstet_cut)])
Utils.log("The number of stars passing the Welch-Stetson cuts is: " + str(n_pass), "info")

# plt.figure(figsize=(18,6))
#
# plt.subplot(1, 2, 1)
# plt.hist(varstats['jstet'], bins=30, range=[0,30], histtype='step', color='k')
# plt.plot([jstet_cut, jstet_cut], [0, 54000], c='r', linewidth=3)
# plt.text(jstet_cut + .5, 40000,
#          "J > " + str(np.around(jstet_cut, decimals=2)),
#          fontsize=15, color="k")
# plt.xlabel('J', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylim([0,54000])
# plt.ylabel('Count', fontsize=20)
# plt.yticks(fontsize=15)
#
# plt.subplot(1, 2, 2)
# plt.hist(varstats['lstet'], bins=30, range=[0,30], histtype='step', color='k')
# plt.plot([lstet_cut, lstet_cut], [0, 92500], c='r', linewidth=3)
# plt.text(lstet_cut + .5, 70000,
#          "L > " + str(np.around(lstet_cut, decimals=2)),
#          fontsize=15, color="k")
# plt.xlabel('L', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylim([0,92500])
# plt.ylabel('Count', fontsize=20)
# plt.yticks(fontsize=15)
#
# plt.savefig("toros_jl_cutoffs.png", dpi=200, bbox_inches='tight')
# plt.show()

v_pass = varstats[(varstats.jstet > jstet_cut) & (varstats.lstet > lstet_cut)].copy().reset_index(drop=True)
for idx, row in v_pass.iterrows():

    if row.gc_star == 0:
        if row.chip < 10:
            lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DETREND_DIRECTORY + '/0' + str(row.chip) + '/' +
                             Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                             sep=" ")
        else:
            lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DETREND_DIRECTORY + '/' + str(row.chip) + '/' +
                             Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                             sep=" ")
        print(row.xcen, row.ycen)
        plt.scatter(lc.jd - 2460580, lc.mag - tv_zpt)
        # plt.scatter(lc.ph, lc.mag - tv_zpt)
        plt.xlabel('JD - 2460580')
        plt.gca().invert_yaxis()
        plt.show()

# make a cumulative distribution for the simp
tots = np.zeros(varstats.simp.max())
vls = np.arange(varstats.simp.max())
tot = 0
for idx in np.arange(varstats.simp.max()):
    tots[idx] = len(varstats[(varstats.simp <= idx) & (varstats.fap1 < 0.01)]) / len(varstats) * 100

plt.scatter(vls, tots)
plt.show()
for idx, row in varstats.iterrows():
    if (row.simp == 0) | (row.simp == 1):
        if row.chip < 10:
            lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DETREND_DIRECTORY + '/0' + str(row.chip) + '/' +
                             Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                             sep=" ")
        else:
            lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DETREND_DIRECTORY + '/' + str(row.chip) + '/' +
                             Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                             sep=" ")

        lc['ph'] = (lc.jd - lc.jd.min()) / row.prd1 % 1
        # plt.scatter(lc.jd - 2460580, lc.mag - tv_zpt)
        plt.scatter(lc.ph, lc.mag - tv_zpt)
        plt.xlabel('ph')
        plt.gca().invert_yaxis()
        plt.show()
print('hold')