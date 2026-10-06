import pandas as pd
from config import Configuration
import numpy as np
from libraries.utils import Utils
import matplotlib
import logging
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats as scs

tv_zpt = 5.4
xcen_47tuc = 6853
ycen_47tuc = 5375
rad_47tuc = 270

# directories
data_dir = "/Volumes/nomad_vandy/toros/lc_stats/"
lc_dir = "/Volumes/nomad_vandy/toros/fin/"

# read in the varstats file
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats_w_flags.txt", sep=' ', low_memory=False)
perstats = varstats[varstats.per_ok == 1].copy().reset_index(drop=True)
n_outs = np.zeros(len(perstats))
tot = np.zeros(len(perstats))
clp = np.zeros(len(perstats))

for idx, row in perstats.iterrows():

    if row.chip < 10:
        lc = pd.read_csv(lc_dir +
                         "0" + str(row.chip) + "/" +
                         Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                         sep=" ")
    else:
        lc = pd.read_csv(lc_dir +
                         str(row.chip) + "/" +
                         Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                         sep=" ")
    if row.pnts == 0:
        mag = lc[lc.mag > 0].mag.to_numpy()
        mn, md, sg = scs(mag, sigma=3)
        tot[idx] = len(mag)
        clp[idx] = len(mag[(mag < mn + 3*sg) & (mag > mn - 3*sg)])
        n_outs[idx] = tot[idx] - clp[idx]
        ph = (lc[lc.mag > 0].jd.to_numpy() - lc[lc.mag > 0].jd.min()) / row.prd % 1

        # if (n_outs[idx] < 10) & (row.G47T == 0) & (row.prd > 0):
        #      Utils.log(str(len(perstats) - idx -1), "info")
        #      plt.scatter(ph, mag, c='k')
        #      plt.gca().invert_yaxis()
        #      plt.title(str(np.around(row.prd, decimals=6)) + " " + str(row.source_id) + " " + str(n_outs[idx]) + " " + str(tot[idx]))
        #      plt.show()

plt.hist(n_outs, bins=20)
print('hold')