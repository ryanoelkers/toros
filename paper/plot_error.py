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
import numpy as np
import statistics

# read in the uncertainties file
errors = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY + '/' + Configuration.FIELD + '_hold/varstats/'
                     + Configuration.FIELD + '_errors.txt',
                     delimiter=' ',
                     low_memory=False)

sky_bkg = 60.
sky_flux = np.pi * (Configuration.APER_SIZE ** 2) * sky_bkg

errors['flux'] = 10 ** ((errors.mag.to_numpy() - 25.)/(-2.5)) * 300.
errors['shot'] = np.sqrt(errors.flux) / errors.flux
errors['shotnsky'] = np.sqrt(errors.flux + sky_flux) / errors.flux

mgs = errors[(errors.rms < 3) & (errors.mag > 7)].mag.to_numpy() - 5.53
rms = errors[(errors.rms < 3) & (errors.mag > 7)].rms.to_numpy()
pht_lim = errors[(errors.rms < 3) & (errors.mag > 7)]['shot'].to_numpy()
pht_sky_lim = errors[(errors.rms < 3) & (errors.mag > 7)]['shotnsky'].to_numpy()

# plot for uncertainties
plt.figure(figsize=(9,6))
plt.scatter(errors[(errors.rms < 3) & (errors.mag > 7)].mag - 5.53,
            errors[(errors.rms < 3) & (errors.mag > 7)].rms,
             marker='.', c='k', alpha=0.01)

plt.plot(mgs[np.argsort(mgs)],
          pht_lim[np.argsort(mgs)],
          marker='.', c='r', linewidth=3, label='Photon Noise')
plt.plot(mgs[np.argsort(mgs)],
          pht_sky_lim[np.argsort(mgs)],
          marker='.', c='b', linewidth=3, label='Photon & Sky Noise')

plt.xlabel(r'$T_G$', fontsize=20)
plt.xticks(fontsize=15)
plt.xlim([8, 25])
plt.ylabel('rms', fontsize=20)
plt.yticks(fontsize=15)
plt.ylim([0.001, 10])
plt.yscale('log')
plt.savefig("toros_yy_precision.png", dpi=200, bbox_inches='tight')
plt.show()

