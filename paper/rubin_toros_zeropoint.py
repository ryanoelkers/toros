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
import numpy as np

rematch = 'Y'

if rematch == 'N':
    # read in the star list
    star_list = pd.read_csv(Configuration.ONE_DRIVE + 'master\\' + Configuration.FIELD + '\\' + Configuration.FIELD + '_star_list.txt',
                            delimiter=' ',
                            header=0,
                            low_memory=False)
    star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                    (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

    # remove obvious bad stars
    star_list = star_list[(~np.isnan(star_list.phot_g_mean_mag)) &
                          (~np.isnan(star_list.phot_bp_mean_mag)) &
                          (~np.isnan(star_list.phot_rp_mean_mag)) &
                          (star_list.var_id == '--') &
                          (star_list.gc_star == 0)].copy().reset_index(drop=True)

    # convert everything to astropy coordinates
    star_list_ra = star_list.ra.to_numpy() * u.degree
    star_list_de = star_list.dec.to_numpy() * u.degree
    star_list_coords = SkyCoord(ra=star_list_ra, dec=star_list_de, frame='icrs')

    # read in the star list
    rubin_list = pd.read_csv(Configuration.ONE_DRIVE + 'lsst\\lsst_data_47tuc_objects.csv',
                            header=0,
                            index_col=0,
                            low_memory=False)

    rubin_list = rubin_list[(~np.isnan(rubin_list.g_psfMag)) &
                            (~np.isnan(rubin_list.r_psfMag)) &
                            (~np.isnan(rubin_list.i_psfMag))].copy().reset_index(drop=True)
    rubin_list['toros_id'] = -1

    # now link the LSST star list with the TOROS star list
    for idx, row in rubin_list.iterrows():
        lsst_ra = row.coord_ra * u.degree
        lsst_de = row.coord_dec * u.degree

        lsst_coords = SkyCoord(ra=lsst_ra, dec=lsst_de, frame='icrs')

        sep = lsst_coords.separation(star_list_coords).arcsec.min() / Configuration.PIXEL_SIZE
        if sep < 5:
            star_list_idx = np.argmin(lsst_coords.separation(star_list_coords))
            rubin_list.loc[idx,'toros_id'] = star_list.star_id[star_list_idx]

    cross_match = pd.merge(rubin_list, star_list, left_on='toros_id', right_on='star_id', how='inner')

    cross_match.to_csv(Configuration.ONE_DRIVE + 'lsst\\toros_lsst_cross_match.csv')

else:
    cross_match = pd.read_csv(Configuration.ONE_DRIVE + 'lsst\\toros_lsst_cross_match.csv',
                              header=0,
                              index_col=0,
                              delimiter=',')

    cross_match['gc_star'] = np.where((cross_match['xcen'] > 4300) & (cross_match['xcen'] < 9300) &
                                    (cross_match['ycen'] > 3600) & (cross_match['ycen'] < 8200), 1, 0)

    cross_match['gmg'] = cross_match.phot_g_mean_mag - cross_match.g_psfMag
    cross_match['gmi'] = cross_match.phot_g_mean_mag - cross_match.i_psfMag
    cross_match['gmr'] = cross_match.phot_g_mean_mag - cross_match.r_psfMag
    cross_match['bmr'] = cross_match.phot_bp_mean_mag - cross_match.phot_rp_mean_mag
    cross_match['gmt'] = cross_match.master_mag - cross_match.phot_g_mean_mag
    cross_match = cross_match[(~np.isnan(cross_match.phot_g_mean_mag)) &
                          (~np.isnan(cross_match.phot_bp_mean_mag)) &
                          (~np.isnan(cross_match.phot_rp_mean_mag)) &
                          (cross_match.var_id == '--') &
                          (cross_match.gc_star == 0) &
                          (cross_match.master_mag_er < 0.02) & (cross_match.gmt > 5)].copy().reset_index(drop=True)
    # plt.scatter(cross_match.bmr, cross_match.gmt, marker='.')
    # plt.show()

    plt.scatter(cross_match.bmr, cross_match.gmr, marker='.')
    plt.show()

    # read in the star list
    star_list = pd.read_csv(Configuration.ONE_DRIVE + 'master\\' + Configuration.FIELD + '\\' + Configuration.FIELD + '_star_list.txt',
                            delimiter=' ',
                            header=0,
                            low_memory=False)
    star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                    (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

    star_list['gmag_off'] = star_list.master_mag - star_list.phot_g_mean_mag
    star_list['bmr'] = star_list.phot_bp_mean_mag - star_list.phot_rp_mean_mag

    # remove obvious bad stars
    star_list = star_list[(~np.isnan(star_list.phot_g_mean_mag)) &
                          (~np.isnan(star_list.phot_bp_mean_mag)) &
                          (~np.isnan(star_list.phot_rp_mean_mag)) &
                          (star_list.var_id == '--') &
                          (star_list.gc_star == 0) &
                          (star_list.master_mag_er < 0.02) & (star_list.gmag_off > 5)].copy().reset_index(drop=True)

    vv = np.polyfit(star_list.bmr, star_list.gmag_off, 3, w=1./star_list.master_mag_er)
    pp = np.poly1d(vv)

    bmr = np.arange(-.5, 3, 0.1)
    plt.errorbar(star_list.bmr, star_list.gmag_off, yerr=star_list.master_mag_er, fmt='none', c='k')
    plt.scatter(star_list.bmr,
                star_list.gmag_off, marker='.', c='b')
    # plt.scatter(star_list.master_mag,
    #             star_list.gmag_off, marker='.', c='r')
    plt.plot(bmr, pp(bmr), c='orange')
    plt.show()


    print('hold')