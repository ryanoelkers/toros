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
from libraries.varstats import Varstats
import numpy as np

tv_zpt = 5.4
xcen_47tuc = 6853
ycen_47tuc = 5375
rad_47tuc = 270

xcen_ngc121 = 1660
ycen_ngc121 = 5000

data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"

# read in the varstats file, and exclude the LSST variables for now
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
varstats = varstats[(varstats.source_id != varstats.lsst_id)].copy().reset_index(drop=True)

varstats['v'] = varstats['master_mag'] - tv_zpt  # correct the V magnitude
varstats['grps'] = 0
varstats['per_ok'] = 0
varstats['var_ok'] = 0
varstats['day_ok'] = 0

dist = np.sqrt((varstats.xcen - xcen_47tuc) ** 2 + (varstats.ycen - ycen_47tuc) ** 2)
varstats['G47T'] = np.where(dist < 1500, 1, 0)

dist = np.sqrt((varstats.xcen - xcen_ngc121) ** 2 + (varstats.ycen - ycen_ngc121) ** 2)
varstats['N121'] = np.where(dist < 150, 1, 0)

varstats['grps'] = 0
varstats.loc[(varstats.xcen > 1010) & (varstats.xcen < 1210) & (varstats.G47T == 0) & (varstats.N121 == 0), 'grps'] = 1
varstats.loc[(varstats.xcen > 1580) & (varstats.xcen < 1810) & (varstats.G47T == 0) & (varstats.N121 == 0), 'grps'] = 1
varstats.loc[(varstats.xcen > 2260) & (varstats.xcen < 2320) & (varstats.G47T == 0) & (varstats.N121 == 0), 'grps'] = 1
varstats.loc[(varstats.xcen > 7740) & (varstats.xcen < 7940) & (varstats.G47T == 0) & (varstats.N121 == 0), 'grps'] = 1
varstats.loc[(varstats.xcen > 8380) & (varstats.xcen < 8440) & (varstats.G47T == 0) & (varstats.N121 == 0), 'grps'] = 1
varstats.loc[(varstats.xcen > 8530) & (varstats.xcen < 8620) & (varstats.G47T == 0) & (varstats.N121 == 0), 'grps'] = 1
varstats.loc[varstats.edge == 1, 'grps'] = 1

# Do the daily metric cutoffs
pass_days = varstats[(varstats.out_mag_std > 0) & (varstats.grps == 0)].copy().reset_index(drop=True)
Utils.log("The number of stars with some kind of outburst between nights is " + str(len(pass_days)), "info")

varstats.loc[(varstats.out_mag_std > 0) & (varstats.grps == 0), 'day_ok'] = 1

# plt.figure(figsize=(9,6))
# plt.scatter(pass_daily.xcen, pass_daily.ycen,
#              c='k', marker='.', label='Stars with Daily Variability')
#
# plt.xlabel('X Pixel', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylabel('Y Pixel', fontsize=20)
# plt.yticks(fontsize=15)
# plt.savefig("toros_xy_daily.png", dpi=200, bbox_inches='tight')
# plt.show()
# plt.close()

# Do the Stetson Metric Cutouffs
jstet_results = Varstats.stetson_j_peak_and_cutoff(varstats.jstet, sigma_method="mirror_std", n_sigma=3.0)
jstet_cut = jstet_results['cutoff']

lstet_results = Varstats.stetson_j_peak_and_cutoff(varstats.lstet, sigma_method="mirror_std", n_sigma=3.0)
lstet_cut = lstet_results['cutoff']

n_var_pass = len(varstats[(varstats.jstet > jstet_cut) & (varstats.lstet > lstet_cut) & (varstats.grps == 0)])
var_pass = varstats[(varstats.jstet > jstet_cut) & (varstats.lstet > lstet_cut) & (varstats.grps == 0)].copy().reset_index(drop=True)

Utils.log("The number of stars passing the Welch-Stetson cuts is: " + str(n_var_pass), "info")

# plt.figure(figsize=(15,6))
#
# plt.subplot(1, 2, 1)
# plt.hist(varstats['jstet'], bins=30, range=[0, 30], histtype='step', color='k')
# plt.plot([jstet_cut, jstet_cut], [0, 62000], c='r', linewidth=3)
# plt.text(jstet_cut + .5, 50000, "J > " + str(np.around(jstet_cut, decimals=2)), fontsize=20, color="k")
# plt.xlabel('J', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylim([0, 60000])
# plt.ylabel('Count', fontsize=20)
# plt.yticks(fontsize=15)
#
# plt.subplot(1, 2, 2)
# plt.hist(varstats['lstet'], bins=30, range=[0, 30], histtype='step', color='k')
# plt.plot([lstet_cut, lstet_cut], [0, 82000], c='r', linewidth=3)
# plt.text(lstet_cut + .5, 65000, "L > " + str(np.around(lstet_cut, decimals=2)), fontsize=20, color="k")
# plt.xlabel('L', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylim([0, 80000])
# plt.ylabel('Count', fontsize=20)
# plt.yticks(fontsize=15)
#
# plt.savefig("toros_jl_cutoffs.png", dpi=200, bbox_inches='tight')
# plt.show()
# plt.close()

pass_vars = varstats[(varstats.jstet > jstet_cut) & (varstats.lstet > lstet_cut) & (varstats.grps == 0)].copy().reset_index(drop=True)

varstats.loc[(varstats.jstet > jstet_cut) & (varstats.lstet > lstet_cut) & (varstats.grps == 0), 'var_ok'] = 1

# plt.figure(figsize=(9,6))
# plt.scatter(pass_vars.xcen, pass_vars.ycen,
#              c='k', marker='.', label='Stars with Statistical Variability')
# #
# plt.xlabel('X Pixel', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylabel('Y Pixel', fontsize=20)
# plt.yticks(fontsize=15)
# plt.savefig("toros_xy_vars.png", dpi=200, bbox_inches='tight')
# plt.show()
# plt.close()


sim_pers = np.zeros(len(varstats))
pwr_rnk = np.zeros(len(varstats))
for idx, row in varstats.iterrows():

    sim_pers[idx] = len(varstats[varstats.prd == row.prd])
    pwr_rnk[idx] = len(varstats[(varstats.pwr > row.pwr) & (varstats.prd == row.prd)])

    # if idx % 1000 == 0:
    #    Utils.log(str(len(varstats) - idx - 1) + " stars to check.", "info")

varstats['simp'] = sim_pers
varstats['prnk'] = pwr_rnk


# coords = np.column_stack([pass_pers.xcen.to_numpy(), pass_pers.ycen.to_numpy()])

# db = DBSCAN(eps=100, min_samples=15).fit(coords)
# labels = db.labels_

varstats.loc[(varstats.prnk < 1) & (varstats.fap < 0.001) & (varstats.grps == 0), 'per_ok'] = 1
pass_pers = varstats[(varstats.prnk < 1) & (varstats.fap < 0.001) & (varstats.grps == 0)]
n_pass_pers = len(pass_pers)

# plt.figure(figsize=(9,6))
# plt.scatter(pass_pers.xcen,
#             pass_pers.ycen,
#             c='k', marker='.', label='Stars with Statistical Periods')
#
# plt.scatter(pass_pers[pass_pers.grps==1].xcen,
#             pass_pers[pass_pers.grps==1].ycen,
#             c='r', marker='.', label='Flagged as in a Structure')
#
# plt.xlabel('X Pixel', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylabel('Y Pixel', fontsize=20)
# plt.yticks(fontsize=15)
# plt.savefig("toros_xy_pers.png", dpi=200, bbox_inches='tight')
# plt.legend(fontsize=15)
# plt.show()
# plt.close()

plt.figure(figsize=(12,8))

all_pass = varstats[(varstats.per_ok == 1) | (varstats.var_ok == 1) | (varstats.day_ok == 1)].copy()

plt.scatter(pass_pers.xcen, pass_pers.ycen, c='orange', label='Periodic Variation', alpha=0.1)
plt.scatter(pass_vars.xcen, pass_vars.ycen, c='r', label='Large Amplitude Variation', alpha=0.1)
plt.scatter(pass_days.xcen, pass_days.ycen, c='k', label='Daily Variation', alpha=0.1)
plt.xlabel('X Pixel', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Y Pixel', fontsize=20)
plt.yticks(fontsize=15)
plt.savefig("toros_xy_all.png", dpi=200, bbox_inches='tight')
plt.legend(fontsize=15)
plt.show()
plt.close()

Utils.log("The number of stars passing the period cuts is: " + str(n_pass_pers), "info")

Utils.log("The total number of stars passing some kind of cut is: " + str(len(varstats[(varstats.per_ok == 1) | (varstats.var_ok == 1) | (varstats.day_ok == 1)])), "info")
Utils.log("The total number of stars passing both cuts is: " + str(len(varstats[(varstats.per_ok == 1) & (varstats.var_ok == 1) & (varstats.day_ok == 1)])), "info")

varstats.to_csv(data_dir + Configuration.FIELD + "_varstats_w_flags.txt",
                sep=' ', header=True, index=False)