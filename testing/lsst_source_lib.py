import pandas as pd
import matplotlib
import logging
from libraries.utils import Utils
from photutils.centroids import centroid_sources
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
from astropy.io import fits
import warnings
warnings.simplefilter('error', RuntimeWarning)
import os

class Sourcelib:

    @staticmethod
    def master_phot(cut_list):

        if os.path.isfile(Configuration.ANALYSIS_DIRECTORY + "/lsst_sources/" + "master_phot.txt"):
            star_list = pd.read_csv(Configuration.ANALYSIS_DIRECTORY + "/lsst_sources/" + "master_phot.txt", sep=' ', low_memory=False)
        else:
            star_list = cut_list.copy().reset_index(drop=True)

            # set up the aperture statistics
            aper_size = 5
            annuli_inner = 7
            annuli_outer = 9

            # read in the master frame
            master, master_header = fits.getdata(Configuration.ANALYSIS_DIRECTORY + "/lsst_sources/" +
                                                 "FIELD_0e.001_master.fits",
                                                 header=True)

            # get the header file and convert to x/y pixel positions
            w = WCS(master_header)
            ra = star_list.ra.to_numpy()
            dec = star_list.dec.to_numpy()

            # convert to x, y
            x, y = w.all_world2pix(ra, dec, 0)

            # add the x/y to the star data frame
            star_list['x'] = x
            star_list['y'] = y

            # centroid the star list
            star_list['xcen'], star_list['ycen'] = centroid_sources(master,
                                                                    star_list.x.to_numpy(),
                                                                    star_list.y.to_numpy(),
                                                                    box_size=5)
            bd_idxs = np.where(np.isnan(star_list.xcen) | np.isnan(star_list.ycen))
            if len(bd_idxs[0]) > 0:
                for bd_idx in bd_idxs[0]:
                    star_list.loc[bd_idx, 'xcen'] = star_list.loc[bd_idx, 'x']
                    star_list.loc[bd_idx, 'ycen'] = star_list.loc[bd_idx, 'y']

            # get the position information
            positions = np.transpose((star_list.xcen.to_numpy(), star_list.ycen.to_numpy()))

            # set up the aperture objects
            aperture = CircularAperture(positions, r=aper_size)
            aperture_area = aperture.area

            # run the photometry to get the data table
            phot_table = aperture_photometry(master, aperture, method='exact')

            # get the background stats
            annulus_aperture = CircularAnnulus(positions, r_in=annuli_inner, r_out=annuli_outer)
            aperstats = ApertureStats(master, annulus_aperture)

            # local sky-subtracted background in ADU
            bkg_mean = aperstats.mean * aperture_area * Configuration.GAIN

            # local background in ADU
            bkg_full = (master_header['sky'] + aperstats.mean) * aperture_area * Configuration.GAIN

            # extract the flux from the table
            star_list['master_flux'] = (np.array(phot_table['aperture_sum']) * Configuration.GAIN) - bkg_mean
            star_list['master_flux_er'] = np.sqrt(np.sqrt(np.abs(star_list['master_flux'])) ** 2 + np.sqrt(bkg_full) ** 2)
            star_list = star_list[star_list.master_flux > 0].copy().reset_index(drop=True)

            # get the master magnitude
            star_list['master_mag'] = 25 - 2.5 * np.log10(star_list['master_flux']) + 2.5 * np.log10(Configuration.EXP_TIME)
            star_list['master_mag_er'] = (np.log(10.) / 2.5) * (star_list['master_flux_er'] / star_list['master_flux'])

            star_list.to_csv(Configuration.ANALYSIS_DIRECTORY + "/lsst_sources/" + "master_phot.txt", sep=' ',
                             index=False)
        _, t_to_v, _ = scs(star_list[star_list.master_mag < 20].master_mag - star_list[star_list.master_mag < 20].V, sigma=3)
        star_list['master_mag'] = star_list['master_mag'] - t_to_v

        return t_to_v, star_list

    @staticmethod
    def image_phot(files, mst_phot, t_to_v):

        star_list = mst_phot.copy().reset_index(drop=True)

        image_stats = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_image_stats.txt',
                                  sep=' ', index_col=0)

        # now loop through the images
        for idx, file in enumerate(files):
            if idx % 10 == 0:
                Utils.log(str(len(files) - 1 - idx) + " files remain to be reduced.", "info")

            image, header = fits.getdata(file, header=True)

            # get the cleaned image for the x/y position
            cln_nme = file.split('diff')[0] + 'clean' + file.split('diff')[1].split('ad.fits')[0] + '.fits'
            cln_header = fits.getheader(cln_nme)
            w_cln = WCS(cln_header)

            x_cln, y_cln = w_cln.all_world2pix(star_list.ra * u.deg, star_list.dec * u.deg, 0)
            star_list['x'] = x_cln
            star_list['y'] = y_cln

            # get the various important header information
            time = Time(header['DATE'], format='isot', scale='utc')
            jd = time.jd

            # set up the aperture statistics
            aper_size = 5
            annuli_inner = 7
            annuli_outer = 9

            # get the position information
            positions = np.transpose((star_list.xcen.to_numpy(), star_list.ycen.to_numpy()))

            # set up the aperture objects
            aperture = CircularAperture(positions, r=aper_size)
            aperture_area = aperture.area

            # run the photometry to get the data table
            phot_table = aperture_photometry(image, aperture, method='exact')

            # get the background stats
            annulus_aperture = CircularAnnulus(positions, r_in=annuli_inner, r_out=annuli_outer)
            aperstats = ApertureStats(image, annulus_aperture)

            # local sky-subtracted background in ADU
            bkg_mean = aperstats.mean * aperture_area * Configuration.GAIN

            # local background in ADU
            bkg_full = (header['sky'] + aperstats.mean) * aperture_area * Configuration.GAIN

            # extract the flux from the table and combine with master flux
            flux = (np.array(phot_table['aperture_sum']) * Configuration.GAIN) - bkg_mean
            star_list['flux'] = star_list.master_flux + flux
            star_list['flux_er'] = np.sqrt(np.sqrt(np.abs(star_list['flux'])) ** 2 + np.sqrt(np.abs(bkg_full)) ** 2)
            star_list['bkg'] = bkg_full

            # get the magnitude
            star_list['raw'] = -9.999999
            star_list.loc[star_list.flux > 0, 'raw'] = (25 - 2.5 * np.log10(star_list.loc[star_list.flux > 0, 'flux']) +
                                                        2.5 * np.log10(Configuration.EXP_TIME)) - t_to_v
            star_list['mag_er'] = (np.log(10.) / 2.5) * (star_list['flux_er'] / star_list['flux'])
            star_list[star_list.mag_er > 1] = 1

            # get the offset
            diff = (star_list[(star_list.raw > 0) & (star_list.raw < 15)].raw -
                    star_list[(star_list.raw > 0) & (star_list.raw < 15)].master_mag)

            _, offset, _ = scs(diff[diff != 0], sigma=2.5)

            # apply the offset and zeropoint
            star_list['mag'] = star_list.raw - offset
            star_list['off'] = offset

            # final data frame
            lsst_list = star_list[star_list.object_type == "LSST"].copy().reset_index(drop=True)
            lsst_list['jd'] = jd
            lsst_list['airmass'] = image_stats[np.around(image_stats.jd, decimals=5) == np.around(jd, decimals=5)].airmass.values[0]
            lsst_list['nstars'] = image_stats[np.around(image_stats.jd, decimals=5) == np.around(jd, decimals=5)].nstars.values[0]
            lsst_list = lsst_list[['star_id', 'ra', 'dec', 'x', 'y', 'airmass', 'nstars',
                                   'xcen', 'ycen', 'jd', 'mag', 'mag_er', 'flux', 'flux_er',
                                   'raw', 'bkg', 'off']].copy()

            lsst_list.to_csv(Configuration.ANALYSIS_DIRECTORY +
                             "/lsst_sources/flux_files/" +
                             file.split('/')[-1].split('fits')[0] + 'flux', index=False, sep=' ')
