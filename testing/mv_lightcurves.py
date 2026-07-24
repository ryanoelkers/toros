import pandas as pd
import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
import matplotlib.pyplot as plt
from config import Configuration
from config import Configuration
from libraries.utils import Utils
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.stats import sigma_clipped_stats as scs

# remove stars near 47 Tuc and the smallcluster
star_list = pd.read_csv("/Users/yuw816/Data/toros/commissioning/master/FIELD_0e.001/"
                        + Configuration.FIELD + "_star_list.txt", sep=' ', low_memory=False, index_col=0)

# get the flux files
dir = "/Users/yuw816/Data/toros/commissioning/flux/"
files = np.sort(Utils.get_all_files_per_field(dir, 'FIELD_0e.001', 'diff', '.flux')[0])
lc = pd.read_csv('/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/detrend/FIELD_0e.001_AQ_Tuc.lc', sep=' ')

trd = np.zeros(len(files))
for idx, file in enumerate(files):

    # read in the flux file
    flux_df = pd.read_csv(file, sep=',', low_memory=False)

    # get the zeropoint offset for the frame
    _, trd[idx], _ = scs(flux_df[flux_df.mag > 0].mag - flux_df[flux_df.mag > 0].master_mag, sigma=2.5)
    if idx % 10 == 0:
        Utils.log(str(len(files) - idx - 1), "info")

for idx, row in star_list.iterrows():

    # read in the light curve
    lc = pd.read_csv('/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/detrend/FIELD_0e.001_' + row.source_id + '.lc', sep=' ')

    # read in the zeropoint offset
    lc['zpt'] = np.around(trd, decimals=4)

    if row.chip < 10:
        lc.to_csv('/Volumes/OUMUAMUA/toros/commissioning/lc/FIELD_0e.001/0' +
                  str(row.chip) + '/FIELD_0e.001_' + row.source_id + '.lc', sep=' ', header=True, index=False)
    else:
        lc.to_csv('/Volumes/OUMUAMUA/toros/commissioning/lc/FIELD_0e.001/' +
                  str(row.chip) + '/FIELD_0e.001_' + row.source_id + '.lc', sep=' ', header=True, index=False)

    lc = lc[['jd', "mag", "err", "raw", "trd", "zpt", "sky", "bkg", "x", "y", "nstars", "airmass"]]

    if idx % 1000 == 0:
        Utils.log(str(len(star_list) - idx - 1), "info")
print('hold')