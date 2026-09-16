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
from scipy.stats import median_abs_deviation as mad
import numpy as np
import statistics
from sklearn.cluster import DBSCAN


tv_zpt = 5.4
xcen_47tuc = 6853
ycen_47tuc = 5375
rad_47tuc = 270

data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/"
lc_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/rescale/"
# read in the varstats file
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
varstats['v'] = varstats['master_mag'] - tv_zpt  # correct the V magnitude
varstats['GRPS'] = 0
varstats['per_ok'] = 0
varstats['var_ok'] = 0
varstats['flr_ok'] = 0

dist = np.sqrt((varstats.xcen - 6832)**2 + (varstats.ycen - 5396)**2)
varstats['G47T'] = np.where(dist < 1500, 1, 0)



jstet_results = Varstats.stetson_j_peak_and_cutoff(varstats[(varstats.source_id != varstats.lsst_id)].jstet, sigma_method="mirror_std", n_sigma=3.0)
jstet_cut = jstet_results['cutoff']

lstet_results = Varstats.stetson_j_peak_and_cutoff(varstats[(varstats.source_id != varstats.lsst_id)].lstet, sigma_method="mirror_std", n_sigma=3.0)
lstet_cut = lstet_results['cutoff']

n_var_pass = len(varstats[(varstats.jstet > jstet_cut) &
                      (varstats.lstet > lstet_cut) &
                      (varstats.source_id != varstats.lsst_id)])
var_pass = varstats[(varstats.jstet > jstet_cut) &
                      (varstats.lstet > lstet_cut) &
                      (varstats.source_id != varstats.lsst_id) &
                    (varstats.G47T == 0)].copy().reset_index(drop=True)

Utils.log("The number of stars passing the Welch-Stetson cuts is: " + str(n_var_pass), "info")


plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)
plt.hist(varstats[(varstats.source_id != varstats.lsst_id)]['jstet'], bins=30, range=[0,30], histtype='step', color='k')
plt.plot([jstet_cut, jstet_cut], [0, 54000], c='r', linewidth=3)
plt.text(jstet_cut + .5, 14000,
           "J > " + str(np.around(jstet_cut, decimals=2)),
           fontsize=20, color="k")
plt.xlabel('J', fontsize=20)
plt.xticks(fontsize=15)
plt.ylim([0,16000])
plt.ylabel('Count', fontsize=20)
plt.yticks(fontsize=15)

plt.subplot(1, 2, 2)
plt.hist(varstats[(varstats.source_id != varstats.lsst_id)]['lstet'], bins=30, range=[0,30], histtype='step', color='k')
plt.plot([lstet_cut, lstet_cut], [0, 28000], c='r', linewidth=3)
plt.text(lstet_cut + .5, 23000, "L > " + str(np.around(lstet_cut, decimals=2)), fontsize=20, color="k")
plt.xlabel('L', fontsize=20)
plt.xticks(fontsize=15)
plt.ylim([0, 27000])
plt.ylabel('Count', fontsize=20)
plt.yticks(fontsize=15)

plt.savefig("toros_jl_cutoffs.png", dpi=200, bbox_inches='tight')
# plt.show()
plt.close()

pass_vars = varstats[(varstats.jstet > jstet_cut) &
                      (varstats.lstet > lstet_cut) &
                      (varstats.source_id != varstats.lsst_id)].copy().reset_index(drop=True)

varstats.loc[(varstats.jstet > jstet_cut) & (varstats.lstet > lstet_cut) & (varstats.source_id != varstats.lsst_id), 'var_ok'] = 1

plt.figure(figsize=(9,6))
plt.scatter(pass_vars.xcen,
            pass_vars.ycen,
            c='k', marker='.', label='Stars with Statistical Variability')

plt.xlabel('X Pixel', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Y Pixel', fontsize=20)
plt.yticks(fontsize=15)
plt.savefig("toros_xy_vars.png", dpi=200, bbox_inches='tight')
# plt.show()
plt.close()

pass_pers = varstats[(varstats.simp < 0.01) & (varstats.fap < 0.001) &
                     (varstats.source_id != varstats.lsst_id)]

coords = np.column_stack([pass_pers.xcen.to_numpy(),
                          pass_pers.ycen.to_numpy()])

db = DBSCAN(eps=200, min_samples=15).fit(coords)
labels = db.labels_

# add back 47-Tuc
labels[(pass_pers.xcen > 4000) & (pass_pers.xcen < 7600)] = -1
labels[labels > -1] = 1
labels[labels == -1] = 0

varstats.loc[pass_pers.index, 'GRPS'] = labels
pass_pers.loc[pass_pers.index, 'GRPS'] = labels

varstats.loc[(varstats.simp < 0.1) & (varstats.fap < 0.001) &
                     (varstats.source_id != varstats.lsst_id) & (varstats.GRPS == 0), 'per_ok'] = 1


plt.figure(figsize=(9,6))
plt.scatter(pass_pers.xcen,
            pass_pers.ycen,
            c='k', marker='.', label='Stars with Statistical Periods')

plt.scatter(pass_pers.xcen[labels == 1],
            pass_pers.ycen[labels == 1],
            c='r', marker='.', label='Flagged as in a Structure')

plt.xlabel('X Pixel', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Y Pixel', fontsize=20)
plt.yticks(fontsize=15)
plt.savefig("toros_xy_pers.png", dpi=200, bbox_inches='tight')
# plt.show()
plt.close()

# update the groups flag
pass_pers = pass_pers[pass_pers.GRPS == 0].copy().reset_index(drop=True)
n_per_pass = len(pass_pers)

for idx, row in pass_pers.iterrows():

    if row.var_period > 0:

        if row.chip < 10:
            lc = pd.read_csv(lc_dir + '/0' + str(row.chip) + '/' +
                             Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                             sep=" ")
        else:
            lc = pd.read_csv(lc_dir + '/' + str(row.chip) + '/' +
                             Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                             sep=" ")

        lc['ph'] = ((lc.jd - lc.jd.min()) / row.prd) % 1

        plt.errorbar(lc[lc.mag> 0].ph, lc[lc.mag> 0].mag, yerr=lc[lc.mag > 0].err, c='k', fmt='none')
        plt.scatter(lc[lc.mag> 0].ph, lc[lc.mag> 0].mag, c='k')
        plt.gca().invert_yaxis()

        print(row.chip, row.source_id, row.var_period, np.around(row.prd, decimals=6),
              len(varstats[varstats.prd == row.prd]), len(varstats[(varstats.prd == row.prd) & (varstats.fap > 0.001)]))
        plt.show()

Utils.log("The number of stars passing the period cuts is: " + str(n_per_pass), "info")

Utils.log("The total number of stars passing some kind of cut is: " + str(len(varstats[(varstats.per_ok == 1) | (varstats.var_ok == 1) ])), "info")
Utils.log("The total number of stars passing both cuts is: " + str(len(varstats[(varstats.per_ok == 1) & (varstats.var_ok == 1) ])), "info")

varstats = varstats.drop(columns=['out_mag_nsct', 'out_sct_nmag', 'out_mag_sct'])

varstats.to_csv(data_dir + Configuration.FIELD + "_varstats_final.txt",
                sep=' ', header=True, index=False)