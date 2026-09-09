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

data_dir = "/Volumes/OUMUAMUA/toros/commissioning/varstats/FIELD_0e.001/"

# read in the varstats file
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
varstats['v'] = varstats['master_mag'] - tv_zpt  # correct the V magnitude

# make the 3 sigma cuts on Jstet and Lstet (these are already appropriately scaled)
mad_jstet = mad(varstats[(varstats.source_id != varstats.lsst_id)]['jstet'])
mdn_jstet = np.median(varstats[(varstats.source_id != varstats.lsst_id)]['jstet'])
mad_lstet = mad(varstats[(varstats.source_id != varstats.lsst_id)]['lstet'])
mdn_lstet = np.median(varstats[(varstats.source_id != varstats.lsst_id)]['lstet'])

jstet_cut = mdn_jstet + 3 * mad_jstet
lstet_cut = mdn_lstet + 3 * mad_lstet

n_var_pass = len(varstats[(varstats.jstet > jstet_cut) &
                      (varstats.lstet > lstet_cut) &
                      (varstats.source_id != varstats.lsst_id)])

Utils.log("The number of stars passing the Welch-Stetson cuts is: " + str(n_var_pass), "info")

n_per_pass = len(varstats[(varstats.simp < 0.01) & (varstats.fap < 0.001) & (varstats.source_id != varstats.lsst_id)])

Utils.log("The number of stars passing the period cuts is: " + str(n_per_pass), "info")

pass_vars = varstats[(varstats.jstet > jstet_cut) &
                      (varstats.lstet > lstet_cut) &
                      (varstats.source_id != varstats.lsst_id)].copy().reset_index(drop=True)
pass_pers = varstats[(varstats.simp < 0.01) & (varstats.fap < 0.001) &
                     (varstats.source_id != varstats.lsst_id)].copy().reset_index(drop=True)

passes = pd.merge(pass_vars, pass_pers, on='source_id', how='outer')
passes_inner = pd.merge(pass_vars, pass_pers, on='source_id', how='inner')
Utils.log("The total number of stars passing some kind of cut is: " + str(len(passes)), "info")
Utils.log("The total number of stars passing both cuts is: " + str(len(passes_inner)), "info")

plt.scatter(pass_pers.xcen, pass_pers.ycen, marker='+', c='r')
plt.scatter(pass_vars.xcen, pass_vars.ycen, marker='x' , c='k')
plt.show()

# plt.figure(figsize=(15,6))
#
# plt.subplot(1, 2, 1)
# plt.hist(varstats['jstet'], bins=30, range=[0,30], histtype='step', color='k')
# plt.plot([jstet_cut, jstet_cut], [0, 54000], c='r', linewidth=3)
# plt.text(jstet_cut + .5, 40000,
#           "J > " + str(np.around(jstet_cut, decimals=2)),
#           fontsize=15, color="k")
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
# plt.close()

