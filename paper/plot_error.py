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

# read in the star list to convert to Vmag
star_list = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + Configuration.FIELD + "_varstats.txt",
                        sep=' ', low_memory=False)
star_list['bmp'] = star_list['phot_bp_mean_mag'] - star_list['phot_rp_mean_mag']
# star_list['v'] = (star_list['phot_g_mean_mag']  + 0.02704 -
#                  0.01424 * star_list['bmp'] +
#                  0.2156 * star_list['bmp'] ** 2 -
#                  0.01426 * star_list['bmp'] ** 3)
star_list['v'] = (star_list['phot_g_mean_mag']
                  + 0.01760
                  + 0.006860 * star_list['bmp']
                  + 0.1732 * star_list['bmp'] ** 2)

plt.scatter(star_list[(star_list.object_type =='Star') & (star_list.prox == 0)]['master_mag'],
            star_list[(star_list.object_type =='Star') & (star_list.prox == 0)]['master_mag'] -
            star_list[(star_list.object_type =='Star') & (star_list.prox == 0)]['v'],
            marker='.', c='k', alpha=0.1)

_, tv_zpt, tv_zpt_std = scs(star_list[(star_list.object_type =='Star') & (star_list.prox == 0) & (star_list.master_mag < 22.5)]['master_mag'] -
                            star_list[(star_list.object_type =='Star') & (star_list.prox == 0) & (star_list.master_mag < 22.5)]['v'], sigma=2.5)
plt.plot([14,24], [tv_zpt, tv_zpt], c='r')
plt.show()
plt.figure(figsize=(9,6))

plt.hist(star_list[(star_list.object_type =='Star') & (star_list.prox == 0)]['master_mag'] -
         star_list[(star_list.object_type =='Star') & (star_list.prox == 0)]['v'],
         bins=40, color='k', histtype='step', linewidth=2)
plt.plot([tv_zpt, tv_zpt], [1700, 1800], color='r', linewidth=3)
plt.text(tv_zpt + 0.05, 1730,
         r"$\tilde{x}$ = " + str(np.around(tv_zpt, decimals=1)) + r" $\pm$ " + str(np.around(tv_zpt_std, decimals=1)),
         fontsize=15, color="k")
plt.xlabel('T - V', fontsize=20)
plt.xticks(fontsize=15)
plt.xlim([2, 7])
plt.ylabel('Count', fontsize=20)
plt.yticks(fontsize=15)
plt.savefig("toros_t2v_offset.png", dpi=200, bbox_inches='tight')
plt.show()

# read in the uncertainties file
errors = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + Configuration.FIELD + '_errors.txt',
                     delimiter=' ',
                     low_memory=False)

sky_bkg = 60.
sky_flux = np.pi * (Configuration.APER_SIZE ** 2) * sky_bkg

errors['flux'] = 10 ** ((errors.mag.to_numpy() - 25.)/(-2.5)) * 300.
errors['shot'] = np.sqrt(errors.flux) / errors.flux
errors['shotnsky'] = np.sqrt(errors.flux + sky_flux) / errors.flux

mgs = errors[(errors.rms < 3) & (errors.mag > 7)].mag.to_numpy() - tv_zpt
rms = errors[(errors.rms < 3) & (errors.mag > 7)].rms.to_numpy()
pht_lim = errors[(errors.rms < 3) & (errors.mag > 7)]['shot'].to_numpy()
pht_sky_lim = errors[(errors.rms < 3) & (errors.mag > 7)]['shotnsky'].to_numpy()

# plot for uncertainties
plt.figure(figsize=(9,6))
plt.scatter(errors[(errors.rms < 3) & (errors.mag > 7)].mag - tv_zpt,
            errors[(errors.rms < 3) & (errors.mag > 7)].rms,
             marker='.', c='k', alpha=0.1)

plt.plot(mgs[np.argsort(mgs)],
          pht_lim[np.argsort(mgs)],
         c='r', linewidth=3, label='Photon Noise')
plt.plot(mgs[np.argsort(mgs)],
          pht_sky_lim[np.argsort(mgs)],
         c='orange', linewidth=3, label='Photon & Sky Noise')

plt.xlabel(r'$V_{TR}$', fontsize=20)
plt.xticks(fontsize=15)
plt.xlim([8, 20.2])
plt.ylabel('rms', fontsize=20)
plt.yticks(fontsize=15)
plt.ylim([0.001, 10])
plt.yscale('log')
plt.legend(loc="upper left", fontsize=15)
plt.savefig("toros_yy_precision.png", dpi=200, bbox_inches='tight')
plt.show()

