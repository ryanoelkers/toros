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

# the star list
star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)

# read in the uncerstainties from the field
errors = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY + Configuration.FIELD + '_errors.txt', sep=' ', names=['name', 'mag', 'rms', 'erms'])
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
