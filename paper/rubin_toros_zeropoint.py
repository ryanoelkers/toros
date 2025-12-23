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

rematch = 'N'

if rematch == 'Y':
    # read in the star list
    star_list = pd.read_csv(Configuration.ONE_DRIVE + 'master/' + Configuration.FIELD + '/' + Configuration.FIELD + '_star_list.txt',
                            delimiter=' ',
                            header=0,
                            low_memory=False)
    star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                    (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)
    star_list['tmg'] = star_list.master_mag - star_list.phot_g_mean_mag

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
    rubin_list = pd.read_csv(Configuration.ONE_DRIVE + 'lsst/lsst_data_47tuc_objects.csv',
                            header=0,
                            index_col=0,
                            low_memory=False)

    # remove any rubin star without photometry in each bandpass
    rubin_list = rubin_list[(~np.isnan(rubin_list.g_psfMag)) &
                            (~np.isnan(rubin_list.r_psfMag)) &
                            (~np.isnan(rubin_list.i_psfMag)) &
                            (rubin_list.g_psfMag < 20) &
                            (rubin_list.r_psfMag < 20) &
                            (rubin_list.i_psfMag < 20)].copy().reset_index(drop=True)
    rubin_list['toros_id'] = -1

    Utils.log("Starting the cross match between TOROS and LSST.", "info")
    # now link the LSST star list with the TOROS star list
    for idx, row in rubin_list.iterrows():

        # convert the star's coordiantes to a SKyCoord object for astropy
        lsst_ra = row.coord_ra * u.degree
        lsst_de = row.coord_dec * u.degree
        lsst_coords = SkyCoord(ra=lsst_ra, dec=lsst_de, frame='icrs')

        # get the separation in arcseconds
        sep = lsst_coords.separation(star_list_coords).arcsec

        # let's make sure the stars within 5 arcseconds are reasonable colors based on empirical testing
        if len(sep[sep < 5]) > 0:

            # # find the stars with separations of 5 arcseconds
            sep_idxs = np.argwhere(sep < 5)
            kk = np.zeros(len(sep[sep < 5]))
            #
            # # loop through the stars and calculate the colors
            for idxs_idx, idxs in enumerate(sep_idxs):
                gmr = star_list.loc[idxs, 'phot_g_mean_mag'].values[0] - row.r_psfMag
                gmi = star_list.loc[idxs, 'phot_g_mean_mag'].values[0] - row.i_psfMag
                gmg = star_list.loc[idxs, 'phot_g_mean_mag'].values[0] - row.g_psfMag
            #
                 # if the star passes the empirical color check, then keep it
                if (gmr < 0.2) & (gmr > -0.2) & (gmi < 0.75) & (gmi > 0.0) & (gmg < -0.25) & (gmg > -1.5):
                     kk[idxs_idx] = 1
            #
            if len(np.argwhere(kk == 1).flatten()) == 1:
            #     # if only one star passes the color check, then that is your guy
                 rubin_list.loc[idx,'toros_id'] = (
                     star_list.star_id[sep_idxs[np.argwhere(kk == 1).flatten()].flatten()].values)[0]
            elif len(np.argwhere(kk == 1).flatten()) > 1:
                # if 1+ stars pass the color check, then pick the star closest to your position
                min_seps = sep[sep < 5]
                min_idxs = np.argmin(min_seps)

                rubin_list.loc[idx, 'toros_id'] = \
                star_list.star_id[sep_idxs[np.argmin(min_seps)].flatten()].values[0]
            # rubin_list.loc[idx, 'toros_id'] = star_list.star_id[np.argmin(sep)]
    # merge the two data frames so we can have all the information for each star based on the star ID from TOROS
    cross_match = pd.merge(rubin_list, star_list, left_on='toros_id', right_on='star_id', how='inner')

    # output the crossmatch so you dont have to crossmatch everytime we run the code
    cross_match.to_csv(Configuration.ONE_DRIVE + 'lsst/toros_lsst_cross_match.csv')
    Utils.log("Crossmatch is complete.", "info")

else:
    # read in the star list
    star_list = pd.read_csv(Configuration.ONE_DRIVE + 'master/' + Configuration.FIELD + '/' + Configuration.FIELD + '_star_list.txt',
                            delimiter=' ',
                            header=0,
                            low_memory=False)
    star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                    (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

    # pull in the saved cross-match file
    cross_match = pd.read_csv(Configuration.ONE_DRIVE + 'lsst/toros_lsst_cross_match.csv',
                              header=0,
                              index_col=0,
                              delimiter=',')
# calculate the colors for easier plotting and fitting
cross_match['ggmg'] = cross_match.phot_g_mean_mag - cross_match.g_psfMag  # Gaia - g
cross_match['ggmi'] = cross_match.phot_g_mean_mag - cross_match.i_psfMag  # Gaia - i
cross_match['ggmr'] = cross_match.phot_g_mean_mag - cross_match.r_psfMag  # Gaia - r
cross_match['bmr'] = cross_match.phot_bp_mean_mag - cross_match.phot_rp_mean_mag  # Bp - Rp
cross_match['tmgg'] = cross_match.master_mag - cross_match.phot_g_mean_mag  # TOROS - Gaia
cross_match['tmg'] = cross_match.master_mag - cross_match.g_psfMag  # TOROS - g
cross_match['tmi'] = cross_match.master_mag - cross_match.i_psfMag  # TOROS - i
cross_match['tmr'] = cross_match.master_mag - cross_match.r_psfMag  # TOROS - r

# remove obvious bad stars
clipped_star_list = star_list[(~np.isnan(star_list.phot_g_mean_mag)) &
                              (~np.isnan(star_list.phot_bp_mean_mag)) &
                              (~np.isnan(star_list.phot_rp_mean_mag)) &
                              (star_list.var_id == '--') &
                              (star_list.gc_star == 0)].copy().reset_index(drop=True)

# find the high sky values
mn_sky, mdn_sky, std_sky = scs(clipped_star_list.master_sky, sigma=3)
sky_cut = mn_sky + 3 * std_sky
clipped_cross_match = cross_match[cross_match.master_sky < sky_cut].copy().reset_index(drop=True)

# clip based on crowding or high sky values
clipped_cross_match['mn_dist'] = 0.
for idx, row in clipped_cross_match.iterrows():
    # use the distance formula
    dist = np.sqrt((row.xcen - star_list.xcen.to_numpy()) ** 2 + (row.ycen - star_list.ycen.to_numpy()) ** 2)
    clipped_cross_match.loc[idx, 'mn_dist'] = np.min(dist[dist > 0])

# only use stars with another star at least 75% away from the other star in aperture
clipped_cross_match = clipped_cross_match[clipped_cross_match.mn_dist > Configuration.APER_SIZE].copy().reset_index(drop=True)

# T - G zeropoint
mean_tmgg, median_tmgg, std_tmgg = scs(clipped_cross_match.tmgg, sigma=5)
ok_data = sc(clipped_cross_match.tmgg.to_numpy(), sigma=5)
clipped_cross_match = clipped_cross_match[~ok_data.mask].copy().reset_index(drop=True)

cnts, binns = np.histogram(clipped_cross_match.tmgg,
                           bins=np.around(np.sqrt(len(clipped_cross_match)), decimals=0).astype(int))
tmgg_zpt = binns[np.argmax(cnts)]

plt.figure(figsize=(9, 6))
plt.hist(clipped_cross_match.tmgg, bins=np.around(np.sqrt(len(clipped_cross_match)), decimals=0).astype(int),
         histtype='step', color='k', linewidth=3, align='left')
plt.arrow(np.around(tmgg_zpt, decimals=2), np.max(cnts) + 10, 0, 25, color='r')
#plt.text(np.around(tmgg_zpt, decimals=2) + 0.01, np.max(cnts) + 20, r'$\Delta_G$=' +
#         str(np.around(tmgg_zpt, decimals=2)) + r'$\pm$' +
#         str(np.around(std_tmgg, decimals=2)), fontsize=20)
plt.xlabel('T - G', fontsize=20)
plt.xticks(fontsize=15)
plt.xlim([5, 6.2])
plt.ylabel('Count', fontsize=20)
plt.yticks(fontsize=15)
# plt.show()
plt.close()
Utils.log("G = T - " + str(np.around(tmgg_zpt, decimals=2)) + "+/-" + str(np.around(std_tmgg, decimals=2)), "info")

# T - g zeropoint
mean_tmg, median_tmg, std_tmg = scs(clipped_cross_match.tmg, sigma=5)

cnts, binns = np.histogram(clipped_cross_match.tmg,
                           bins=np.around(np.sqrt(len(clipped_cross_match)), decimals=0).astype(int))
tmg_zpt = binns[np.argmax(cnts)]

plt.figure(figsize=(12, 6))
plt.hist(clipped_cross_match.tmg, bins=np.around(np.sqrt(len(clipped_cross_match)), decimals=0).astype(int),
         histtype='step', color='darkgreen', linewidth=3, align='left')
plt.arrow(np.around(tmg_zpt, decimals=2), np.max(cnts) + 10, 0, 25, color='darkgreen')
#plt.text(np.around(tmg_zpt, decimals=2) + 0.01, np.max(cnts) + 20, r'$\Delta_g$=' +
#         str(np.around(tmg_zpt, decimals=2)) + r'$\pm$' +
#         str(np.around(std_tmg, decimals=2)), fontsize=20)
plt.xlabel(r'T - $\lambda$', fontsize=20)
plt.xticks(fontsize=15)
# plt.xlim([4.6, 5.4])
plt.ylabel('Count', fontsize=20)
plt.yticks(fontsize=15)
# plt.show()
# plt.close()
Utils.log("g = T - " + str(np.around(tmg_zpt, decimals=2)) + "+/-" + str(np.around(std_tmg, decimals=2)), "info")

# T - r zeropoint
mean_tmr, median_tmr, std_tmr = scs(clipped_cross_match.tmr, sigma=5)

cnts, binns = np.histogram(clipped_cross_match.tmr,
                           bins=np.around(np.sqrt(len(clipped_cross_match)), decimals=0).astype(int))
tmr_zpt = binns[np.argmax(cnts)]

# plt.figure(figsize=(9, 6))
plt.hist(clipped_cross_match.tmr, bins=np.around(np.sqrt(len(clipped_cross_match)), decimals=0).astype(int),
         histtype='step', color='r', linewidth=3, align='left')
plt.arrow(np.around(tmr_zpt, decimals=2), np.max(cnts) + 10, 0, 25, color='r')
#plt.text(np.around(tmr_zpt, decimals=2) + 0.01, np.max(cnts) + 20, r'$\Delta_r$=' +
#         str(np.around(tmr_zpt, decimals=2)) + r'$\pm$' +
#         str(np.around(std_tmr, decimals=2)), fontsize=20)
#plt.xlabel('T - r', fontsize=20)
#plt.xticks(fontsize=15)
#plt.xlim([5.1, 6.1])
#plt.ylabel('Count', fontsize=20)
#plt.yticks(fontsize=15)
#plt.show()
#plt.close()
Utils.log("r = T - " + str(np.around(tmr_zpt, decimals=2)) + "+/-" + str(np.around(std_tmr, decimals=2)), "info")

# T - i zeropoint
mean_tmi, median_tmi, std_tmi = scs(clipped_cross_match.tmi, sigma=5)

cnts, binns = np.histogram(clipped_cross_match.tmi,
                           bins=np.around(np.sqrt(len(clipped_cross_match)), decimals=0).astype(int))
tmi_zpt = binns[np.argmax(cnts)]

# plt.figure(figsize=(9, 6))
plt.hist(clipped_cross_match.tmi, bins=np.around(np.sqrt(len(clipped_cross_match)), decimals=0).astype(int),
         histtype='step', color='maroon', linewidth=3, align='left')
plt.arrow(np.around(tmi_zpt, decimals=2), np.max(cnts) + 10, 0, 25, color='maroon')
# plt.text(np.around(tmi_zpt, decimals=2) + 0.01, np.max(cnts) + 20, r'$\Delta_i$=' +
#          str(np.around(tmi_zpt, decimals=2)) + r'$\pm$' +
#          str(np.around(std_tmi, decimals=2)), fontsize=20)
#plt.xlabel('T - i', fontsize=20)
#plt.xticks(fontsize=15)
#plt.xlim([5.2, 6.6])
#plt.ylabel('Count', fontsize=20)
#plt.yticks(fontsize=15)
plt.savefig("zerpoints.png", dpi=200, bbox_inches='tight')
plt.show()

plt.close()
Utils.log("i = T - " + str(np.around(tmi_zpt, decimals=2)) + "+/-" + str(np.around(std_tmi, decimals=2)), "info")

Utils.log("See ya later Alligator!", "info")

## Maybe don't use these? They require Gaia Bp - Rp, which more stars will not have. ##

# now let's calculate the offset based on if you have both Bp & Rp
# vv = np.polyfit(cross_match.bmr, cross_match.gmg, 2)
# pp = np.poly1d(vv)
# tt = np.arange(0.5, 2.2, 0.1)
#
# plt.figure(figsize=(9, 6))
# plt.scatter(cross_match.bmr, cross_match.gmg, c='k', marker='.')
# plt.plot(tt, pp(tt), c='r')
# plt.xlim([0.5, 2.2])
# plt.xlabel(r'B$_p$ - R$_p$', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylabel("G-g", fontsize=20)
# plt.yticks(fontsize=15)
# plt.ylim([-1.5, -0.2])
# plt.show()
# plt.close()
#
# vv = np.polyfit(cross_match.bmr, cross_match.gmi, 2)
# pp = np.poly1d(vv)
# tt = np.arange(0.5, 2.2, 0.1)
#
# plt.figure(figsize=(9, 6))
# plt.scatter(cross_match.bmr, cross_match.gmi, c='k', marker='.')
# plt.plot(tt, pp(tt), c='r')
# plt.xlabel(r'B$_p$ - R$_p$', fontsize=20)
# plt.xticks(fontsize=15)
# plt.xlim([0.5, 2.2])
# plt.ylabel("G-i", fontsize=20)
# plt.yticks(fontsize=15)
# plt.show()
# plt.close()
#
# vv = np.polyfit(cross_match.bmr, cross_match.gmr, 2)
# pp = np.poly1d(vv)
# tt = np.arange(0.5, 2.2, 0.1)
#
# plt.figure(figsize=(9, 6))
# plt.scatter(cross_match.bmr, cross_match.gmr, c='k', marker='.')
# plt.plot(tt, pp(tt), c='r')
# plt.xlabel(r'B$_p$ - R$_p$', fontsize=20)
# plt.xticks(fontsize=15)
# plt.xlim([0.5, 2.2])
# plt.ylabel("G-r", fontsize=20)
# plt.yticks(fontsize=15)
# plt.show()
# plt.close()

# first calculate the offset between G and T
# pull in the star list
# star_list = pd.read_csv(Configuration.ONE_DRIVE + 'master/' + Configuration.FIELD + '/' + Configuration.FIELD + '_star_list.txt',
#                         delimiter=' ',
#                         header=0,
#                         low_memory=False)
# star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
#                                 (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)
# star_list['tmg'] = star_list.master_mag - star_list.phot_g_mean_mag
# star_list['bmr'] = star_list.phot_bp_mean_mag - star_list.phot_rp_mean_mag
#
# # remove obvious bad stars
# clipped_star_list = star_list[(~np.isnan(star_list.phot_g_mean_mag)) &
#                               (~np.isnan(star_list.phot_bp_mean_mag)) &
#                               (~np.isnan(star_list.phot_rp_mean_mag)) &
#                               (star_list.var_id == '--') &
#                               (star_list.gc_star == 0) &
#                               (star_list.phot_g_mean_mag < 19) &
#                               (star_list.master_mag_er < 0.1)].copy().reset_index(drop=True)
#
# # find the high sky values
# mn_sky, mdn_sky, std_sky = scs(clipped_star_list.master_sky, sigma=3)
# sky_cut = mn_sky + 3 * std_sky
# clipped_star_list = clipped_star_list[clipped_star_list.master_sky < sky_cut].copy().reset_index(drop=True)
#
# # clip based on crowding or high sky values
# clipped_star_list['mn_dist'] = 0.
# for idx, row in clipped_star_list.iterrows():
#     # use the distance formula
#     dist = np.sqrt((row.xcen - star_list.xcen.to_numpy()) ** 2 + (row.ycen - star_list.ycen.to_numpy()) ** 2)
#     clipped_star_list.loc[idx, 'mn_dist'] = np.min(dist[dist > 0])
#
# # only use stars with another star at least 75% away from the other star in aperture
# clipped_star_list = clipped_star_list[clipped_star_list.mn_dist > Configuration.APER_SIZE].copy().reset_index(drop=True)
#
# # now let's just get the basic offset if you have no Bp or Rp
# mean_tmg, median_tmg, std_tmg = scs(clipped_star_list.tmg, sigma=5)
# ok_data = sc(clipped_star_list.tmg.to_numpy(), sigma=5)
# clipped_star_list = clipped_star_list[~ok_data.mask].copy().reset_index(drop=True)
#
# cnts, binns = np.histogram(clipped_star_list.tmg,
#                            bins=np.around(np.sqrt(len(clipped_star_list)), decimals=0).astype(int))
# tmg_zpt = binns[np.argmax(cnts)]
#
# plt.figure(figsize=(9, 6))
# plt.hist(clipped_star_list.tmg, bins=np.around(np.sqrt(len(clipped_star_list)), decimals=0).astype(int),
#          histtype='step', color='k', linewidth=3, align='left')
# plt.arrow(np.around(tmg_zpt, decimals=2), np.max(cnts) + 10, 0, 25, color='r')
# plt.text(np.around(tmg_zpt, decimals=2) + 0.05, np.max(cnts) + 20, r'$\mu$=' +
#          str(np.around(tmg_zpt, decimals=2)) + r'$\pm$' +
#          str(np.around(std_tmg, decimals=2)), fontsize=15)
# plt.xlabel('T-G', fontsize=20)
# plt.xticks(fontsize=15)
# plt.xlim([4.25, 6.9])
# plt.ylabel('Count', fontsize=20)
# plt.yticks(fontsize=15)
# plt.show()
# plt.close()
#
# # now let's calculate the offset based on if you have both Bp & Rp
# vv = np.polyfit(clipped_star_list.bmr, clipped_star_list.tmg, 2)
# pp = np.poly1d(vv)
# tt = np.arange(-0.5, 3.5, 0.1)
#
# plt.figure(figsize=(9, 6))
# plt.errorbar(clipped_star_list.bmr, clipped_star_list.tmg, yerr=clipped_star_list.master_mag_er, c='k', fmt='none')
# plt.scatter(clipped_star_list.bmr, clipped_star_list.tmg, c='k', marker='.')
# plt.plot(tt, pp(tt), c='r')
# plt.xlim([-0.5, 3.0])
# plt.xlabel(r'B$_p$ - R$_p$', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylabel("T-G", fontsize=20)
# plt.yticks(fontsize=15)
# plt.ylim([4.25, 6.9])
# plt.show()
# plt.close()
