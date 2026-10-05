import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import pandas as pd
from config import Configuration
from libraries.utils import Utils
import matplotlib.colors as colors
data_dir = "/Volumes/OUMUAMUA/toros/commissioning/varstats/FIELD_0e.001/"
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt",
                       delimiter=' ', low_memory=False)

# plot LSST variables that are in Gaia
lsst_gaia = varstats[(varstats.source_id != varstats.lsst_id) & (varstats.lsst_id != '--')].copy().reset_index(drop=True)
abs_g = lsst_gaia.phot_g_mean_mag - 5 * np.log10(1000. / lsst_gaia.parallax) + 5
bmp = lsst_gaia.phot_bp_mean_mag - lsst_gaia.phot_rp_mean_mag

plt.scatter(bmp, abs_g, marker='.', alpha=0.1, c=lsst_gaia.prd, norm=colors.LogNorm())
plt.colorbar()
plt.gca().invert_yaxis()
plt.show()