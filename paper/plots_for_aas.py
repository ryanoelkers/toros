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

# read in the star list
star_list = pd.read_csv(Configuration.ONE_DRIVE + 'master/' + Configuration.FIELD + '/' + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)
star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

# read in the uncertainties file
errors = pd.read_csv(Configuration.ONE_DRIVE + 'varstats/' + Configuration.FIELD + '_errors.txt',
                         delimiter=' ',
                         names=['name', 'mag', 'rms', 'erms'],
                         low_memory=False)
errors['gc_star'] = star_list['gc_star'].to_numpy()
errors['var_type'] = star_list['var_type'].to_numpy()

# read in the varstats file
varstats = pd.read_csv(Configuration.ONE_DRIVE + 'varstats/' + Configuration.FIELD + '_varstats.txt',
                       delimiter=' ',
                       header=0,
                       low_memory=False)
varstats['gc_star'] = star_list['gc_star'].to_numpy()
varstats['var_type'] = star_list['var_type'].to_numpy()
varstats['var_period'] = star_list['var_period'].to_numpy()
varstats['object_type'] = star_list['object_type'].to_numpy()
varstats['master_mag'] = star_list['master_mag'].to_numpy()
varstats['x'] = star_list['xcen'].to_numpy()
varstats['y'] = star_list['ycen'].to_numpy()
varstats['ra'] = star_list['ra'].to_numpy()
varstats['dec'] = star_list['dec'].to_numpy()

sky_bkg = 60.
sky_flux = np.pi * (Configuration.APER_SIZE ** 2) * sky_bkg

varstats['flux'] = 10 ** ((varstats.mag.to_numpy() - 25.)/(-2.5)) * 300.
varstats['shot'] = np.sqrt(varstats.flux) / varstats.flux
varstats['shotnsky'] = np.sqrt(varstats.flux + sky_flux) / varstats.flux

mgs = varstats[(varstats.rms < 3) & (varstats.mag > 7) & (varstats.gc_star == 0)].mag.to_numpy() - 5.53
rms = varstats[(varstats.rms < 3) & (varstats.mag > 7) & (varstats.gc_star == 0)].rms.to_numpy()
pht_lim = varstats[(varstats.rms < 3) & (varstats.mag > 7) & (varstats.gc_star == 0)]['shot'].to_numpy()
pht_sky_lim = varstats[(varstats.rms < 3) & (varstats.mag > 7) & (varstats.gc_star == 0)]['shotnsky'].to_numpy()

# plot for uncertainties
# plt.figure(figsize=(9,6))
# plt.scatter(varstats[(varstats.rms < 3) & (varstats.mag > 7) & (varstats.gc_star == 0)].mag - 5.53,
#            varstats[(varstats.rms < 3) & (varstats.mag > 7) & (varstats.gc_star == 0)].rms,
#             marker='.', c='k', alpha=0.1)
#
# plt.plot(mgs[np.argsort(mgs)],
#          pht_lim[np.argsort(mgs)],
#          marker='.', c='r', linewidth=3)
# plt.plot(mgs[np.argsort(mgs)],
#          pht_sky_lim[np.argsort(mgs)],
#          marker='.', c='b', linewidth=3)
#
# plt.xlabel(r'$T_G$', fontsize=20)
# plt.xticks(fontsize=15)
# plt.xlim([8, 21])
# plt.ylabel('rms', fontsize=20)
# plt.yticks(fontsize=15)
# plt.ylim([0.004, 1])
# plt.yscale('log')
# plt.savefig("aas_precision.png", dpi=200, bbox_inches='tight')
# plt.show()

cols_to_combine = ['p1', 'p2', 'p3', 'p4', 'p5']
periods = varstats[cols_to_combine].to_numpy().flatten()
cols_to_combine = ['pwr1', 'pwr2', 'pwr3', 'pwr4', 'pwr5']
powers = varstats[cols_to_combine].to_numpy().flatten()
cols_to_combine = ['fap1', 'fap2', 'fap3', 'fap4', 'fap5']

# make a plot showing the aliasing that occurs
prds, cnts = np.unique(periods, return_counts=True)
aap = cnts / np.max(cnts)

var1 = 'FIELD_0e.001_4689637956899105792.lc'
tvar = varstats[varstats['name'] == var1]
lc = pd.read_csv(Configuration.ONE_DRIVE + 'lc/' + Configuration.FIELD + '/' + var1, sep=' ', header=0)
clp = sc(lc.mag.to_numpy(), sigma=3)
clp.mask[np.argwhere(lc.mag <= 0)] = True
lc = lc[~clp.mask].copy().reset_index(drop=True)
lc['jd'] = lc.jd.to_numpy() - 2460580.

lc['ph'] = (lc.jd.to_numpy() - lc.jd.min()) / tvar.p1.values[0] % 1
lc['ph'] = lc['ph'] - .75
lc['ph'] = np.where(lc.ph < 0, lc.ph + 1, lc.ph)

plt.figure(figsize=(9,6))

plt.errorbar(2.0, 18.2 - 5.53, yerr=lc.err.median(), c='r', fmt='none')
plt.scatter(lc.ph, lc.mag - 5.53, marker='.', c='k')
plt.scatter(lc.ph + 1, lc.mag - 5.53, marker='.', c='k')

plt.ylabel(r'$T_G$', fontsize=20)
plt.yticks(fontsize=15)
plt.gca().invert_yaxis()

plt.xticks(fontsize=15)
plt.xlabel('Phase', fontsize=20)

plt.savefig("rr_lyrae_" + str(tvar.p1.values[0]) + ".png", dpi=200, bbox_inches='tight')
plt.close()
# plt.show()

var2 = 'FIELD_0e.001_4689579240377005184.lc'
tvar = varstats[varstats['name'] == var2]
lc = pd.read_csv(Configuration.ONE_DRIVE + 'lc/' + Configuration.FIELD + '/' + var2, sep=' ', header=0)
clp = sc(lc.mag.to_numpy(), sigma=3)
clp.mask[np.argwhere(lc.mag <= 0)] = True
lc = lc[~clp.mask].copy().reset_index(drop=True)
lc['jd'] = lc.jd.to_numpy() - 2460580.

lc['ph'] = (lc.jd.to_numpy() - lc.jd.min()) / tvar.p1.values[0] % 1
lc['ph'] = lc['ph'] - .88
lc['ph'] = np.where(lc.ph < 0, lc.ph + 1, lc.ph)

plt.figure(figsize=(9,6))

plt.errorbar(2.0, 14.1, yerr=lc.err.median(), c='r', fmt='none')
plt.scatter(lc.ph, lc.mag - 5.53, marker='.', c='k')
plt.scatter(lc.ph + 1, lc.mag - 5.53, marker='.', c='k')

plt.ylabel(r'$T_G$', fontsize=20)
plt.yticks(fontsize=15)
plt.ylim([14.2, 13.5])
# plt.gca().invert_yaxis()

plt.xticks(fontsize=15)
plt.xlabel('Phase', fontsize=20)

plt.savefig("rrc_" + str(tvar.p1.values[0]) + ".png", dpi=200, bbox_inches='tight')
plt.close()

var3 = 'FIELD_0e.001_4689637789376112256.lc'

tvar = varstats[varstats['name'] == var3]
lc = pd.read_csv(Configuration.ONE_DRIVE + 'lc/' + Configuration.FIELD + '/' + var3, sep=' ', header=0)
clp = sc(lc.mag.to_numpy(), sigma=3)
clp.mask[np.argwhere(lc.mag <= 0)] = True
lc = lc[~clp.mask].copy().reset_index(drop=True)
lc['jd'] = lc.jd.to_numpy() - 2460580.

plt.figure(figsize=(9,6))

plt.errorbar(55.0, 10.8, yerr=lc.err.median(), c='r', fmt='none')
plt.scatter(lc.jd, lc.mag - 5.53, marker='.', c='k')

plt.ylabel(r'$T_G$', fontsize=20)
plt.yticks(fontsize=15)
plt.gca().invert_yaxis()

plt.xticks(fontsize=15)
plt.xlabel('JD - 2460580', fontsize=20)

plt.savefig("l_var.png", dpi=200, bbox_inches='tight')
plt.close()

xray1 = 'FIELD_0e.001_4689626897325758336.lc'

tvar = varstats[varstats['name'] == xray1]
lc = pd.read_csv(Configuration.ONE_DRIVE + 'lc/' + Configuration.FIELD + '/' + xray1, sep=' ', header=0)
clp = sc(lc.mag.to_numpy(), sigma=3)
clp.mask[np.argwhere(lc.mag <= 0)] = True
lc = lc[~clp.mask].copy().reset_index(drop=True)
lc['jd'] = lc.jd.to_numpy() - 2460580.

lc['ph'] = (lc.jd.to_numpy() - lc.jd.min()) / tvar.p1.values[0] % 1
lc['ph'] = lc['ph'] - .49
lc['ph'] = np.where(lc.ph < 0, lc.ph + 1, lc.ph)

plt.figure(figsize=(9,6))

plt.errorbar(2.0, 13.49, yerr=lc.err.median(), c='r', fmt='none')
plt.scatter(lc.ph, lc.mag - 5.53, marker='.', c='k')
plt.scatter(lc.ph + 1, lc.mag - 5.53, marker='.', c='k')

plt.ylabel(r'$T_G$', fontsize=20)
plt.yticks(fontsize=15)
plt.gca().invert_yaxis()

plt.xticks(fontsize=15)
plt.xlabel('Phase', fontsize=20)

plt.savefig("cv_" + str(tvar.p1.values[0]) + ".png", dpi=200, bbox_inches='tight')
plt.close()

xray2 = 'FIELD_0e.001_4689638407851723392.lc'
tvar = varstats[varstats['name'] == xray2]
lc = pd.read_csv(Configuration.ONE_DRIVE + 'lc/' + Configuration.FIELD + '/' + xray2, sep=' ', header=0)
clp = sc(lc.mag.to_numpy(), sigma=3)
clp.mask[np.argwhere(lc.mag <= 0)] = True
lc = lc[~clp.mask].copy().reset_index(drop=True)
lc['jd'] = lc.jd.to_numpy() - 2460580.

plt.figure(figsize=(9,6))

plt.errorbar(55.0, 13.6, yerr=lc.err.median(), c='r', fmt='none')
plt.scatter(lc.jd, lc.mag - 5.53, marker='.', c='k')

plt.ylabel(r'$T_G$', fontsize=20)
plt.yticks(fontsize=15)
plt.gca().invert_yaxis()

plt.xticks(fontsize=15)
plt.xlabel('JD - 2460580', fontsize=20)

plt.savefig("cv_time.png", dpi=200, bbox_inches='tight')
plt.close()

xray3 = 'FIELD_0e.001_4688807756879485184.lc'

tvar = varstats[varstats['name'] == xray3]
lc = pd.read_csv(Configuration.ONE_DRIVE + 'lc/' + Configuration.FIELD + '/' + xray3, sep=' ', header=0)
clp = sc(lc.mag.to_numpy(), sigma=3)
clp.mask[np.argwhere(lc.mag <= 0)] = True
lc = lc[~clp.mask].copy().reset_index(drop=True)
lc['jd'] = lc.jd.to_numpy() - 2460580.

lc['ph'] = (lc.jd.to_numpy() - lc.jd.min()) / tvar.p1.values[0] % 1
lc['ph'] = lc['ph'] -.37
lc['ph'] = np.where(lc.ph < 0, lc.ph + 1, lc.ph)

plt.figure(figsize=(9,6))

plt.errorbar(2.0, 16.86, yerr=lc.err.median(), c='r', fmt='none')
plt.scatter(lc.ph, lc.mag - 5.53, marker='.', c='k')
plt.scatter(lc.ph + 1, lc.mag - 5.53, marker='.', c='k')

plt.ylabel(r'$T_G$', fontsize=20)
plt.yticks(fontsize=15)
# plt.ylim([14.2, 13.5])
plt.gca().invert_yaxis()

plt.xticks(fontsize=15)
plt.xlabel('Phase', fontsize=20)

plt.savefig("agn_" + str(tvar.p1.values[0]) + ".png", dpi=200, bbox_inches='tight')
plt.close()
