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
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from astropy.time import Time
import matplotlib
import logging
from astropy.stats import sigma_clipped_stats as scs
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
from astropy.wcs import WCS
import astropy.units as u
from lsst_source_lib import Sourcelib
from astropy.io import fits
import warnings
warnings.simplefilter('error', RuntimeWarning)
import os

# star list from LSST
lsst_list = pd.read_csv(Configuration.ANALYSIS_DIRECTORY + "lsst_sources.csv",
                        delimiter=',',
                        header=0,
                        low_memory=False)

master_skip = 0
if master_skip == 0:
    # star list from TOROS
    star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                            delimiter=' ',
                            header=0,
                            low_memory=False)

    # remove stars in 47-Tuc
    star_list['gc_star'] = np.where((star_list['xcen'] > 5500) & (star_list['xcen'] < 8100) &
                                    (star_list['ycen'] > 4100) & (star_list['ycen'] < 6800), 1, 0)

    # convert from G to V
    star_list['V'] = star_list.apply(lambda x: x.phot_g_mean_mag +
                                               0.0176 -
                                               0.00686 * (x.phot_bp_mean_mag - x.phot_rp_mean_mag) +
                                               0.1732 * (x.phot_bp_mean_mag - x.phot_rp_mean_mag) ** 2, axis=1)

    # clean up columns and drop
    star_list = star_list.drop(columns=['star_id'])
    star_list = star_list.rename(columns={"phot_g_mean_mag": "G",
                                          "phot_bp_mean_mag": "Bp",
                                          "phot_rp_mean_mag": "Rp",
                                          "source_id": "star_id"})

    # make the final star list based on available photometry
    cut_list = star_list[(star_list.gc_star == 0) &
                         (star_list.object_type == 'Star') &
                         (star_list.V > 0) & (star_list.V < 19.5)].copy().reset_index(drop=True)
    cut_list = cut_list[['star_id', 'ra', 'dec', 'V', 'G', 'Bp', 'Rp', 'object_type']].copy().reset_index(drop=True)

    # now concatenate the two lists
    full_list = pd.concat([lsst_list, cut_list], ignore_index=True).reset_index(drop=True)

    # do the master frame photometry
    t_to_v, mst_phot = Sourcelib.master_phot(full_list)

    # get the image list for photometry
    files, date_dirs = Utils.get_all_files_per_field(Configuration.DIFFERENCED_DIRECTORY,
                                                     Configuration.FIELD,
                                                     'diff',
                                                     Configuration.FILE_EXTENSION)

    # do the photometry for all images
    v = Sourcelib.image_phot(files, mst_phot, t_to_v)

# make the light curves for the objects
files = np.sort(os.listdir(Configuration.ANALYSIS_DIRECTORY + "lsst_sources/flux_files/"))

for idx, row in lsst_list.iterrows():

    Utils.log("Working on " + row.star_id + '.', "info")
    f = open(Configuration.ANALYSIS_DIRECTORY + "lsst_sources/lc_files/" + row.star_id + ".lc", "w")
    f.write('jd mag mag_er flux flux_er raw off x y airmass bkg nstars\n')

    for idy, file in enumerate(files):

        flux = pd.read_csv(Configuration.ANALYSIS_DIRECTORY + "lsst_sources/flux_files/" + file, sep=' ')

        if len(flux[flux.star_id == row.star_id]) > 0:
            f.write(str(np.around(flux.loc[flux.star_id == row.star_id, 'jd'].squeeze(), decimals=6)) + ' ' +
                    str(np.around(flux.loc[flux.star_id == row.star_id,'mag'].squeeze(), decimals=4)) + ' ' +
                    str(np.around(flux.loc[flux.star_id == row.star_id,'mag_er'].squeeze(), decimals=4)) + ' ' +
                    str(np.around(flux.loc[flux.star_id == row.star_id,'flux'].squeeze(), decimals=2)) + ' ' +
                    str(np.around(flux.loc[flux.star_id == row.star_id,'flux_er'].squeeze(), decimals=2)) + ' ' +
                    str(np.around(flux.loc[flux.star_id == row.star_id,'raw'].squeeze(), decimals=6)) + ' ' +
                    str(np.around(flux.loc[flux.star_id == row.star_id,'off'].squeeze(), decimals=6)) + ' ' +
                    str(np.around(flux.loc[flux.star_id == row.star_id,'x'].squeeze(), decimals=6)) + ' ' +
                    str(np.around(flux.loc[flux.star_id == row.star_id,'y'].squeeze(), decimals=2)) + ' ' +
                    str(np.around(flux.loc[flux.star_id == row.star_id,'airmass'].squeeze(), decimals=3)) + ' ' +
                    str(np.around(flux.loc[flux.star_id == row.star_id,'bkg'].squeeze(), decimals=0)) + ' ' +
                    str(np.around(flux.loc[flux.star_id == row.star_id,'nstars'].squeeze(), decimals=0)) +'\n')
    f.close()

    lc = pd.read_csv(Configuration.ANALYSIS_DIRECTORY + "lsst_sources/lc_files/" + row.star_id + ".lc", sep=' ')
    lc = lc.sort_values(by='jd')
    lc.to_csv(Configuration.ANALYSIS_DIRECTORY + "lsst_sources/lc_files/" + row.star_id + ".lc", sep=' ', index=False)

    if row.star_id == 'CO_TUC':
        lc['ph'] = (lc.jd - lc.jd[0]) / 0.37143 % 1
        plt.errorbar(lc[lc.mag > 0].ph, lc[lc.mag > 0].mag, yerr=lc[lc.mag > 0].mag_er, fmt='none', c='k')
        plt.scatter(lc[lc.mag > 0].ph, lc[lc.mag > 0].mag, c='k')
        plt.xlabel('Phase')
        plt.ylabel('V$_{TOROS}$')
        plt.gca().invert_yaxis()
        plt.savefig(Configuration.ANALYSIS_DIRECTORY + '/lsst_sources/lc_files/' + str(row.star_id) + '.png',
                    bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        plt.errorbar(lc[lc.mag > 0].jd - 2460584, lc[lc.mag > 0].mag, yerr=lc[lc.mag > 0].mag_er, fmt='none', c='k')
        plt.scatter(lc[lc.mag > 0].jd - 2460584, lc[lc.mag > 0].mag, c='k')

        plt.xlabel('Phase')
        plt.ylabel('V$_{TOROS}$')
        plt.gca().invert_yaxis()
        plt.savefig(Configuration.ANALYSIS_DIRECTORY + '/lsst_sources/lc_files/' + str(row.star_id) + '.png',
                    bbox_inches='tight')
        plt.show()
        plt.close()