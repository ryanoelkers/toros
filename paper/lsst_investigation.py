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

# read in the var stats
varstats = pd.read_csv(Configuration.DATA_DIRECTORY + 'varstats/' +
                     Configuration.FIELD + '/' + Configuration.FIELD + '_varstats.txt',
                     delimiter=' ',
                     low_memory=False)

lsstvars = varstats[(varstats.object_type == 'LSST')].copy().reset_index(drop=True)

# Get how many stars pass the J/L cut
j_cut = np.median(varstats.jstet.to_numpy()) + 3 * 1.4826 * mad(varstats.jstet.to_numpy())
l_cut = np.median(varstats.lstet.to_numpy()) + 3 * 1.4826 * mad(varstats.lstet.to_numpy())

pass_vars = varstats[(varstats.jstet > j_cut) & (varstats.lstet > l_cut) & (varstats.out_sct == 0)].copy().reset_index(drop=True)
Utils.log("The number of passing variables is: " + str(int(len(pass_vars))), "info")
lsst_vvars = lsstvars[(lsstvars.jstet > j_cut) & (lsstvars.lstet > l_cut) & (lsstvars.out_sct == 0)].copy().reset_index(drop=True)
Utils.log("The number of passing LSST variables is: " + str(int(len(lsst_vvars))), "info")

# Get how many stars pass the daily variance cut
pass_dvars = varstats[(varstats.out_mag > 0) & (varstats.out_sct == 0)].copy().reset_index(drop=True)
Utils.log("The number of stars with an outburst is: " + str(int(len(pass_dvars))), "info")

lsst_dvars = lsstvars[(lsstvars.out_mag > 0) & (lsstvars.out_sct == 0)].copy().reset_index(drop=True)
Utils.log("The number of stars with an outburst in LSST is: " + str(int(len(lsst_dvars))), "info")

# Get how many stars pass the periodicity cuts
pass_pvars = varstats[(varstats.simp < 0.05) & (varstats.fap < 0.001)].copy().reset_index(drop=True)
Utils.log("The number of stars with a statistical period is: " + str(int(len(pass_pvars))), "info")

lsst_pvars = lsstvars[(lsstvars.simp < 0.05) & (lsstvars.fap < 0.001)].copy().reset_index(drop=True)
Utils.log("The number of stars with a statistical period in LSST is: " + str(int(len(lsst_pvars))), "info")

# make the final plot of sources
plt.scatter(pass_vars.master_mag-5.3, pass_vars.rms, marker='.', c='k', alpha=0.1)
#plt.scatter(lsst_pvars.master_mag-5.3, lsst_pvars.rms, marker='.', c='r', alpha=0.1)
#plt.scatter(lsst_dvars.master_mag-5.3, lsst_dvars.rms, marker='.', c='orange', alpha=0.1)
plt.plot([15.8, 15.8], [0.001, 10])
plt.xlabel(r'V$_T$', fontsize=15)
plt.ylabel('rms', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
plt.ylim([0.004, 4])
plt.show()

print('hold')
