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

errors = pd.read_csv("/Volumes/OUMUAMUA/toros/commissioning/lc/FIELD_0e.001/error_chk.txt", sep=" ")

sky_bkg = 60.
sky_flux = 3.14 * (16 ** 2) * sky_bkg * 0.38

errors['flux'] = 10 ** (((errors.mag.to_numpy() - 2.5 * np.log10(300.)) - 25.)/(-2.5))
errors['shot'] = np.sqrt(errors.flux) / errors.flux
errors['shotnsky'] = np.sqrt(errors.flux + sky_flux) / errors.flux

mgs = errors.mag.to_numpy() - 5.4
rms = errors.min_rms.to_numpy()
pht_lim = errors['shot'].to_numpy()
pht_sky_lim = errors['shotnsky'].to_numpy()

plt.scatter(errors.mag-5.4, errors.mean_rms, c='k', marker='.', alpha=0.1)
# plt.scatter(errs.mag, errs.mean_cmp, c='r', marker='.', alpha=0.1)
plt.plot(mgs[np.argsort(mgs)], pht_lim[np.argsort(mgs)], c='r', linewidth=3, label='Photon Noise')
plt.plot(mgs[np.argsort(mgs)], pht_sky_lim[np.argsort(mgs)], c='orange', linewidth=3, label='Photon & Sky Noise')
plt.yscale('log')
plt.show()