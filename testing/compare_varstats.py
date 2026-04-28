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
import numpy as np
from astropy.stats import sigma_clipped_stats as scs
from astropy.timeseries import LombScargle
import warnings
warnings.simplefilter('error', RuntimeWarning)

# remove stars near 47 Tuc and the small cluster
star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                        sep=' ',
                        header=0,
                        low_memory=False)

star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)
errors = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + '/varstats/' + Configuration.FIELD + '_errors.txt',
                     sep=' ',
                     header=0,
                     low_memory=False)
varstats = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + '/varstats/' + Configuration.FIELD + '_varstats.txt',
                       sep=' ',
                       header=0,
                       low_memory=False)

perds = np.concatenate([varstats.p1.to_numpy(),
                        varstats.p2.to_numpy(),
                        varstats.p3.to_numpy(),
                        varstats.p4.to_numpy(),
                        varstats.p5.to_numpy()])

pwrs = np.concatenate([varstats.pwr1.to_numpy(),
                        varstats.pwr2.to_numpy(),
                        varstats.pwr3.to_numpy(),
                        varstats.pwr4.to_numpy(),
                        varstats.pwr5.to_numpy()])

plt.scatter(perds, pwrs)
plt.show()
prd_bins = np.unique(perds)
prd_cnts = np.zeros(len(prd_bins))
for idx, prd in enumerate(prd_bins):
    prd_cnts[idx] = len(perds[perds == prd])

plt.plot(prd_bins, prd_cnts)
plt.yscale('log')
plt.show()
img_stats = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_image_stats.txt', sep=' ', index_col=0)
img_stats['bd_day'] = np.where(img_stats.nstars < 20000, 1, 0)

# set up the magnitude range for determining the 95% window
mag_step = 1
mag_range_1 = np.arange(14, 17, mag_step)
mag_step = 0.25
mag_range_2 = np.arange(17, 26.5, mag_step)

mag_range = np.concatenate([mag_range_1, mag_range_2])
cut_d90 = np.zeros(len(mag_range) - 1)
cut_jstet = np.zeros(len(mag_range) - 1)
cut_lstet = np.zeros(len(mag_range) - 1)

perc = 0.9

for idx, mag in enumerate(mag_range[:-1]):
    cut_d90[idx] = varstats[(varstats.mag > mag) & (varstats.mag < mag_range[idx + 1]) &
                            (varstats.d90 > 0) & (varstats.Lstet > 0) & (varstats.Lstet > 0)].d90.quantile(perc)
    cut_jstet[idx] = varstats[(varstats.mag > mag) & (varstats.mag < mag_range[idx + 1]) &
                              (varstats.d90 > 0) & (varstats.Lstet > 0) & (varstats.Lstet > 0)].Jstet.quantile(perc)
    cut_lstet[idx] = varstats[(varstats.mag > mag) & (varstats.mag < mag_range[idx + 1]) &
                              (varstats.d90 > 0) & (varstats.Lstet > 0) & (varstats.Lstet > 0)].Lstet.quantile(perc)

ok_4_varstats = (varstats.rms > 0) & (varstats.d90 > 0) & (varstats.Lstet > 0) & (varstats.Lstet > 0)
# plt.scatter(varstats[ok_4_varstats].mag, varstats[ok_4_varstats].d90, marker='.', alpha=0.1, c='k')
# plt.plot(mag_range[:-1], cut_d90, c='r', linewidth=2)
# plt.xlabel('TOROS Instrumental Magnitude')
# plt.ylabel(r'$\Delta_{90}$')
# plt.yscale('log')
# plt.show()
# plt.close()

# plt.scatter(varstats[ok_4_varstats].mag, varstats[ok_4_varstats].Jstet, marker='.', alpha=0.1, c='k')
# plt.plot(mag_range[:-1], cut_jstet, c='r', linewidth=2)
# plt.xlabel('TOROS Instrumental Magnitude')
# plt.ylabel(r'$J_S$')
# plt.yscale('log')
# plt.show()
# plt.close()

# plt.scatter(varstats[ok_4_varstats].mag, varstats[ok_4_varstats].Lstet, marker='.', alpha=0.1, c='k')
# plt.plot(mag_range[:-1], cut_lstet, c='r', linewidth=2)
# plt.xlabel('TOROS Instrumental Magnitude')
# plt.ylabel(r'$L_S$')
# plt.yscale('log')
# plt.show()
# plt.close()

full_cut_d90 = np.interp(varstats.mag, mag_range[:-1], cut_d90)
full_cut_jstet = np.interp(varstats.mag, mag_range[:-1], cut_jstet)
full_cut_lstet = np.interp(varstats.mag, mag_range[:-1], cut_lstet)

new_var_list = varstats[(varstats.d90 > full_cut_d90) &
                        (varstats.Jstet > full_cut_jstet) &
                        (varstats.Lstet > full_cut_lstet)].copy().reset_index(drop=True)

## SNR FIGURE ##
# get the magnitude values
inst_mag = varstats.mag.to_numpy()
inst_mag = np.sort(inst_mag)

# set up the flux values
flx = 10 ** (((inst_mag  - 2.5 * np.log10(300.)) - 25) / (-2.5))
e_rms_photon = 1. / np.sqrt(flx)

# now assume some sky values
sky_mean, sky_median, sky_sig = scs(img_stats[img_stats.bd_day == 0].sky, sigma=2.5)
sky_noise = Configuration.GAIN * np.pi * Configuration.APER_SIZE ** 2 * sky_median
e_rms_photon_n_sky = np.sqrt(flx + sky_noise) / flx

plt.scatter(varstats[star_list.gc_star == 1].mag, varstats[star_list.gc_star == 1].min_rms,
            marker='.', c='k', alpha=0.05, label='Likely 47-Tuc Member')
plt.scatter(varstats[star_list.gc_star == 0].mag, varstats[star_list.gc_star == 0].min_rms,
            marker='.', c='b', alpha=0.1, label='Outside 47-Tuc')
plt.plot(inst_mag, e_rms_photon, c='r', label='Photon Noise')
plt.plot(inst_mag, e_rms_photon_n_sky, c='orange', label='Photon + Sky Noise')
plt.yscale('log')
plt.xlabel('TOROS Instrumental Magnitude')
plt.ylabel('Typical Nightly rms')
plt.xlim([14, 28])
plt.ylim([0.001, 10.])
plt.legend()
# plt.show()
plt.close()


print('hold')