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
varstats['object_type'] = star_list['object_type'].to_numpy()
varstats['master_mag'] = star_list['master_mag'].to_numpy()
varstats['x'] = star_list['xcen'].to_numpy()
varstats['y'] = star_list['ycen'].to_numpy()
varstats['ra'] = star_list['ra'].to_numpy()
varstats['dec'] = star_list['dec'].to_numpy()

# plot for uncertainties
# plt.figure(figsize=(9,6))
# plt.scatter(varstats[(varstats.rms < 3) & (varstats.mag > 7)].mag - 5.53,
#             varstats[(varstats.rms < 3) & (varstats.mag > 7)].rms, marker='.', c='k', alpha=0.1)
# plt.xlabel(r'$T_G$', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylabel('rms', fontsize=20)
# plt.yticks(fontsize=15)
# plt.yscale('log')
# plt.savefig("aas_precision.png", dpi=200, bbox_inches='tight')
# plt.show()

# get the periodicity cutoffs
cols_to_combine = ['p1', 'p2', 'p3', 'p4', 'p5']
periods = varstats[cols_to_combine].to_numpy().flatten()
cols_to_combine = ['pwr1', 'pwr2', 'pwr3', 'pwr4', 'pwr5']
powers = varstats[cols_to_combine].to_numpy().flatten()
cols_to_combine = ['fap1', 'fap2', 'fap3', 'fap4', 'fap5']
faps = varstats[cols_to_combine].to_numpy().flatten()

# make a plot showing the aliasing that occurs
prds, cnts = np.unique(periods, return_counts=True)
aap = cnts / np.max(cnts)  # normalize by the maximum duplicated period

# look for rotation periods on xray sources
xray = varstats[(varstats.gc_star == 0) & (varstats.object_type != '--')].copy().reset_index(drop=True)

# read in the list of LSST variable objects
lsst_vars = pd.read_csv(Configuration.DATA_DIRECTORY + 'lsst/lsst_data_47tuc_variables.csv',
                        header=0,
                        index_col=0,
                        low_memory=False)

# read in the list of LSST transient objects
lsst_trans = pd.read_csv(Configuration.DATA_DIRECTORY + 'lsst/lsst_data_47tuc_transients.csv',
                        header=0,
                        index_col=0,
                        low_memory=False)

# first we want to make sure we only do matching across one object at a time
lsst_vars_obs = lsst_vars.groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean'}).reset_index()
lsst_vars_obs['toros_id'] = 0
lsst_trans_obs = lsst_trans.groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean'}).reset_index()

# convert everything to astropy coordinates
xray_ra = xray.ra.to_numpy() * u.degree
xray_de = xray.dec.to_numpy() * u.degree
xray_coords = SkyCoord(ra=xray_ra, dec=xray_de, frame='icrs')
lsst_vars_ra = lsst_vars_obs.coord_ra.to_numpy() * u.degree
lsst_vars_de = lsst_vars_obs.coord_dec.to_numpy() * u.degree
lsst_vars_coords = SkyCoord(ra=lsst_vars_ra, dec=lsst_vars_de, frame='icrs')
lsst_trans_ra = lsst_trans_obs.coord_ra.to_numpy() * u.degree
lsst_trans_de = lsst_trans_obs.coord_dec.to_numpy() * u.degree
lsst_trans_coords = SkyCoord(ra=lsst_trans_ra, dec=lsst_trans_de, frame='icrs')


for idx, row in varstats.iterrows():

    var_sep = xray_coords[idx].separation(lsst_vars_coords).arcsec.min() / Configuration.PIXEL_SIZE
    # var_sep = xray_coords[idx].separation(lsst_trans_coords).arcsec.min() / Configuration.PIXEL_SIZE

    if var_sep < .5:
        var_idx = np.argmin(xray_coords[idx].separation(lsst_vars_coords).arcsec)
        lsst_obs = lsst_vars[lsst_vars.diaObjectId == lsst_vars_obs.loc[var_idx].diaObjectId.astype(int)].copy().reset_index(drop=True)
        lsst_obs['mag'] = lsst_obs.apply(lambda x: -2.5 * np.log10(x['psfFlux']) + 31.4, axis=1).to_numpy()
        lsst_obs['mag_err'] = lsst_obs.apply(lambda x: np.abs(1.086 * x['psfFluxErr'] / x['psfFlux']), axis=1).to_numpy()
        mean_chk = lsst_obs[lsst_obs.band == 'g'].mag.mean() + 5.15

        # read in the light curve data
        lc = pd.read_csv(Configuration.ONE_DRIVE + 'lc/' + Configuration.FIELD + '/' + row['name'], sep=' ', header=0)
        clp = sc(lc.mag.to_numpy(), sigma=3)
        clp.mask[np.argwhere(lc.mag <= 0)] = True
        lc = lc[~clp.mask].copy().reset_index(drop=True)

        lsst_obs['ph'] = (((lsst_obs.expMidptMJD.to_numpy() - 60630) - np.min(lc.jd.to_numpy() - 2460630)) / row.p1) % 1

        plt.figure(figsize=(9, 6))
        plt.errorbar(lsst_obs[lsst_obs.band =='g'].expMidptMJD.to_numpy() - 60630,
                     lsst_obs[lsst_obs.band =='g'].mag - lsst_obs[lsst_obs.band =='g'].mag.mean(),
                     yerr=lsst_obs[lsst_obs.band =='g'].mag_err, fmt='none', c='dodgerblue')
        plt.scatter(lsst_obs[lsst_obs.band =='g'].expMidptMJD.to_numpy() - 60630,
                    lsst_obs[lsst_obs.band =='g'].mag - lsst_obs[lsst_obs.band =='g'].mag.mean(), marker='.', c='dodgerblue')

        plt.errorbar(lsst_obs[lsst_obs.band =='r'].expMidptMJD.to_numpy() - 60630,
                     lsst_obs[lsst_obs.band =='r'].mag - lsst_obs[lsst_obs.band =='r'].mag.mean(),
                     yerr=lsst_obs[lsst_obs.band =='r'].mag_err, fmt='none', c='orange')
        plt.scatter(lsst_obs[lsst_obs.band =='r'].expMidptMJD.to_numpy() - 60630,
                    lsst_obs[lsst_obs.band =='r'].mag - lsst_obs[lsst_obs.band =='r'].mag.mean(), marker='.', c='orange')

        plt.errorbar(lsst_obs[lsst_obs.band =='i'].expMidptMJD.to_numpy() - 60630,
                     lsst_obs[lsst_obs.band =='i'].mag - lsst_obs[lsst_obs.band =='i'].mag.mean(),
                     yerr=lsst_obs[lsst_obs.band =='i'].mag_err, fmt='none', c='maroon')
        plt.scatter(lsst_obs[lsst_obs.band =='i'].expMidptMJD.to_numpy() - 60630,
                    lsst_obs[lsst_obs.band =='i'].mag - lsst_obs[lsst_obs.band =='i'].mag.mean(), marker='.', c='maroon')

        plt.errorbar(lc.jd - 2460630, lc.mag, yerr=lc.err, fmt='none', c='k', alpha=0.1)
        lc['ph'] = (lc.jd - lc.jd.min()) / row.p1 % 1
        plt.scatter(lc.jd.to_numpy() - 2460630, lc.mag - lc.mag.mean(), marker='.', c='k')
        plt.ylim([.2, -.2])
        plt.ylabel(r'$T_G$', fontsize=20)
        plt.yticks(fontsize=15)
        plt.xlabel('JD-246580 [d]', fontsize=20)
        plt.xticks(fontsize=15)
        # plt.gca().invert_yaxis()

        plt.title(str(np.around(var_sep, decimals=2)) + ' ' + str(row.p1))
        plt.show()
        # # get the period
        # if (cnts[prds == row.p1][0] < 100) & (row.fap1 == 0) & (row.p1 < 50):
        #     lc['ph'] = (lc.jd - lc.jd.min()) / row.p1 % 1
        #     plt.scatter(lc.ph, lc.mag)
        #     plt.title(str(np.around(row.p1, decimals=4)))
        #     plt.gca().invert_yaxis()
        #     plt.show()