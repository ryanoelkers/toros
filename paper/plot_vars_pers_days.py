import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from config import Configuration

tv_zpt = 5.4
xcen_47tuc = 6853
ycen_47tuc = 5375
rad_47tuc = 270

# directories
data_dir = "/Volumes/OUMUAMUA/toros/commissioning/varstats/FIELD_0e.001/"
lc_dir = "/Volumes/OUMUAMUA/toros/commissioning/lc/FIELD_0e.001/rescale"

# read in the varstats file
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
varstats['v'] = varstats['master_mag'] - tv_zpt

lc = pd.read_csv(lc_dir + '/09/' + Configuration.FIELD + '_4689582435832666752.lc', sep=" ")

plt.figure(figsize=(24,6))

plt.subplot(1, 3, 1)
plt.errorbar(lc.jd - 2460500, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460500, lc.mag - tv_zpt, c='k', s=12)
plt.gca().invert_yaxis()
plt.xlabel('JD - 2460500', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$V_T$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689582435832666752',fontsize=12)

lc = pd.read_csv(lc_dir + '/09/' + Configuration.FIELD + '_4689627137844175232.lc', sep=" ")
plt.subplot(1, 3, 2)
plt.errorbar(lc.jd - 2460500, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460500, lc.mag - tv_zpt, c='k', s=12)
plt.gca().invert_yaxis()
plt.xlabel('JD - 2460500', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$V_T$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689627137844175232',fontsize=12)

lc = pd.read_csv(lc_dir + '/12/' + Configuration.FIELD + '_4689639507355023488.lc', sep=" ")
plt.subplot(1, 3, 3)
plt.errorbar(lc.jd - 2460500, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460500, lc.mag - tv_zpt, c='k', s=12)
plt.gca().invert_yaxis()
plt.xlabel('JD - 2460500', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$V_T$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689639507355023488',fontsize=12)

plt.savefig("toros_variables.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()