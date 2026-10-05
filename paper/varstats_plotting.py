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
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats_w_flags.txt", sep=' ', low_memory=False)
varstats = varstats[(varstats.source_id != varstats.lsst_id)].copy().reset_index(drop=True)

for idx, row in varstats.iterrows():

    if (row.day_ok == 1) & (row.G47T == 0):
        if row.chip < 10:
            lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                             "star_list/fin/" +
                             "0" + str(row.chip) + "/" +
                             Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                             sep=" ")
        else:
            lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY +
                             "star_list/fin/" +
                             str(row.chip) + "/" +
                             Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                             sep=" ")

        # lc['ph'] = (lc.jd - lc.jd.min()) / row.prd % 1
        plt.scatter(lc[lc.mag > 0].jd, lc[lc.mag > 0].mag, marker='.', c='k')
        plt.title(str(row.source_id) + ' ' + str(row.out_mag_std) + ' ' + str(row.chip))
        plt.gca().invert_yaxis()
        plt.show()

print('hold')