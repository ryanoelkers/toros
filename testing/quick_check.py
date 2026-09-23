import pandas as pd
import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np
from config import Configuration

data_dir = Configuration.LIGHTCURVE_FIELD_DIRECTORY

# read in the uncertainties file
errors_cmp = pd.read_csv(data_dir + Configuration.FIELD + '_errors.txt', delimiter=' ', low_memory=False)

errors = pd.read_csv("/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/error_chk.txt",
                     delimiter=' ', low_memory=False)

sky_bkg = 60.
sky_flux = 3.14 * (16 ** 2) * sky_bkg * 0.38

errors['flux'] = 10 ** (((errors.mag.to_numpy() - 2.5 * np.log10(300.)) - 25.)/(-2.5))
errors['shot'] = np.sqrt(errors.flux) / errors.flux
errors['shotnsky'] = np.sqrt(errors.flux + sky_flux) / errors.flux

mgs = errors.mag.to_numpy() - 5.4
rms = errors.min_rms.to_numpy()
pht_lim = errors['shot'].to_numpy()
pht_sky_lim = errors['shotnsky'].to_numpy()

plt.scatter(errors.mag - 5.4, errors.mean_rms, c='k', marker='.', alpha=0.1)
plt.scatter(errors_cmp.mag - 5.4, errors_cmp.rms, c='r', marker='.', alpha=0.1)
plt.plot(mgs[np.argsort(mgs)], pht_lim[np.argsort(mgs)], c='r', linewidth=3, label='Photon Noise')
plt.plot(mgs[np.argsort(mgs)], pht_sky_lim[np.argsort(mgs)], c='orange', linewidth=3, label='Photon & Sky Noise')
plt.yscale('log')
plt.show()