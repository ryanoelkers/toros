from config import Configuration
from libraries.utils import Utils
from libraries.preprocessing import Preprocessing
from libraries.photometry import Photometry
import os
from astropy.io import fits
from photutils.centroids import centroid_sources
import pandas as pd
from astroquery.mast import Catalogs
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
from astropy.coordinates import EarthLocation
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import matplotlib.pyplot as plt

class Master:

    @staticmethod
    def pull_master():
        """ This script will generate the master file and photometry file for the data reduction.

        :return - The master frame is returned and the star list is printed
        """

        master, master_header = Master.mk_master()

        star_list = Master.master_phot(master, master_header)

        return master, star_list  # kernel_stars

    @staticmethod
    def master_phot(master, master_header):
        """ This program will generate the star lists for the master frame and provide a photometry file."""

        if os.path.isfile(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt') == 0:

            if os.path.isfile(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_gaia_dump.txt') == 0:
                # create the string useful for query_region
                #field = str(Configuration.RA) + " " + str(Configuration.DEC)
                field = str(master_header['CRVAL1']) + ' ' + str(master_header['CRVAL2'])

                # select the columns we want to import into the data table
                columns = ["toros_field_id", "source_id", "ra", "dec", "phot_g_mean_mag", "phot_bp_mean_mag",
                           "phot_rp_mean_mag", "teff_val", "parallax", "parallax_error", "pmra",
                           "pmra_error", "pmdec", "pmdec_error"]

                # run the query
                Utils.log('Querying MAST for all stars within the toros field: ' + str(Configuration.FIELD),
                          'info')
                catalog_data = Catalogs.query_region(field,
                                                     radius=Configuration.SEARCH_DIST,
                                                     catalog="Gaia").to_pandas()
                Utils.log('Query finished. ' + str(len(catalog_data)) + ' stars found.', 'info')

                # add the toros field to the catalog data
                catalog_data['toros_field_id'] = Configuration.FIELD

                # pull out the necessary columns
                star_list = catalog_data[columns]

                # get the header file and convert to x/y pixel positions
                w = WCS(master_header)
                ra = star_list.ra.to_numpy()
                dec = star_list.dec.to_numpy()

                # convert to x, y
                x, y = w.all_world2pix(ra, dec, 0)

                # add the x/y to the star data frame
                star_list['x'] = x
                star_list['y'] = y

                # dump the list to file, so you don't have to keep querying Gaia
                star_list.to_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_gaia_dump.txt', sep=' ')
            else:
                Utils.log("Reading dumped Gaia file. Delete if you want a new query.", "info")
                # read in the dumped file
                star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_gaia_dump.txt',
                                        sep=' ', index_col=0)

            # check for any "known" transients and variable star files
            if Configuration.KNOWN_VARIABLES == 'Y':
                star_list = Photometry.add_variable_list(star_list, master_header)

            # remove the stars outside the frame
            star_list = star_list[(star_list.x >= 500) & (star_list.x < 10540) &
                                  (star_list.y > 20) & (star_list.y <= 9700)].copy().reset_index(drop=True)

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

            # add "chip" to the star_list
            star_list['chip'] = 1
            kk = 1
            for idx in range(0, Configuration.AXS_X, Configuration.CHP_X):
                for idy in range(0, Configuration.AXS_Y, Configuration.CHP_Y):
                    star_list['chip'] = np.where((star_list.xcen > idx) & (star_list.xcen < idx + 1320) &
                                                 (star_list.ycen > idy) & (star_list.ycen < idy + 5280),
                                                 kk, star_list.chip)
                    kk = kk + 1

            # centroid the positions
            positions = star_list[['xcen', 'ycen']].copy().reset_index(drop=True)  # positions = (x, y)

            # set up the star aperture and sky annuli
            aperture = CircularAperture(positions, r=Configuration.APER_SIZE)

            aperture_area = aperture.area  # the area of the aperture
            annulus_aperture = CircularAnnulus(positions,
                                               r_in=Configuration.ANNULI_INNER,
                                               r_out=Configuration.ANNULI_OUTER)

            # get the background stats
            aperstats = ApertureStats(master, annulus_aperture)
            bkg_mean = aperstats.mean + master_header['MDN_SKY']
            total_bkg = bkg_mean * aperture_area * Configuration.GAIN

            # run the photometry to get the data table
            phot_table = aperture_photometry(master, aperture, method='exact')

            # extract the flux from the table
            star_flux = np.array(phot_table['aperture_sum']) * Configuration.GAIN  # sky was already subtracted

            # calculate the expected photometric error
            star_error = star_flux
            bkg_error = total_bkg

            # combine sky and signal error in quadrature
            star_flux_err = np.sqrt(star_error + bkg_error)

            # convert to magnitude
            mag = 25. - 2.5 * np.log10(star_flux / Configuration.EXP_TIME)
            mag_er = (np.log(10.) / 2.5) * (star_flux_err / star_flux)

            # initialize the light curve data frame
            star_list['master_mag'] = mag
            star_list['master_mag_er'] = mag_er
            star_list['master_flux'] = star_flux
            star_list['master_flux_er'] = star_flux_err
            star_list['master_sky'] = total_bkg

            # only keep stars with reasonable photometry in the master list
            star_list = star_list[(star_list['master_flux'] > 0) | (star_list['object_type'] != 'Star')]

            # index is reset twice to make sure the star ID matches the brightness on the master frame
            star_list = star_list.sort_values(by='master_mag').reset_index(drop=True).reset_index()
            star_list = star_list.rename(columns={'index': 'star_id'})

            star_list.to_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                             sep=' ',
                             index=False)

        else:
            star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                                    delimiter=' ',
                                    header=0)

        return star_list

    @staticmethod
    def mk_master():
        """ This function will make the master frame that will be used for the differencing

        :return - The master frame is returned and written to the master directory
        """

        file_name = Configuration.FIELD +'_master' + Configuration.FILE_EXTENSION

        if os.path.isfile(Configuration.MASTER_DIRECTORY + file_name) == 0:

            chk_tmp_files = Utils.get_file_list(Configuration.MASTER_TMP_DIRECTORY, Configuration.FILE_EXTENSION)

            # get the image list
            full_image_list, dates = Utils.get_all_files_per_field(Configuration.CLEAN_DIRECTORY,
                                                                   Configuration.FIELD,
                                                                   'clean',
                                                                   Configuration.FILE_EXTENSION)
            # get the image statistics
            image_stats = Master.get_image_stats(full_image_list)

            # get statistics on the sky
            _, img_mdn_sky, img_std_sky = sigma_clipped_stats(image_stats.sky.to_numpy(), sigma=2)

            # get the good files
            pass_images = image_stats[(image_stats['nstars'] > Configuration.NSKY_STARS) &
                                     (image_stats['sky'] < img_mdn_sky + 2 * img_std_sky) &
                                     (image_stats['date'] != Configuration.BAD_DATES)].copy().reset_index(drop=True)

            # get the center coordinates for the master frame
            pass_files = pass_images.file.to_list()
            cen_ra = pass_images.ra.median()
            cen_dec = pass_images.dec.median()
            nfiles = len(pass_files)

            # find the closest file to the center
            file_dist = np.sqrt((cen_ra - pass_images.ra) ** 2 + (cen_dec - pass_images.dec) ** 2)
            master_file = pass_files[np.argmin(file_dist)]

            Utils.log("Center found at ra: " + str(np.around(cen_ra, decimals=2)) + " dec: " +
                      str(np.around(cen_dec, decimals=2)), "info")

            # pull in and set the master frame
            master, master_header = fits.getdata(master_file, header=True)

            if len(chk_tmp_files) == 0:
                Utils.log("No temporary Master files found. Generating new ones.", "info")
                # determine the number of loops we need to move through for each image
                nbulk = 20

                # get the integer and remainder for the combination
                full_bulk = nfiles // nbulk
                part_bulk = nfiles % nbulk

                if part_bulk > 0:
                    hold_bulk = full_bulk + 1
                else:
                    hold_bulk = full_bulk

                # here is the 'holder'
                hold_data = np.ndarray(shape=(hold_bulk, Configuration.AXS_Y, Configuration.AXS_X))

                # update the log
                Utils.log("Generating a master frame from multiple files in bulks of " + str(nbulk) +
                          " images. There are " + str(nfiles) + " images to combine, which means there should be " +
                          str(hold_bulk) + " mini-files to median combine.", "info")

                cnt_img = 0
                for kk in range(0, hold_bulk):

                    # loop through the images in sets of nbulk
                    if kk < full_bulk:
                        # generate the image holder
                        block_hold = np.ndarray(shape=(nbulk, Configuration.AXS_Y, Configuration.AXS_X))

                        # generate the max index
                        mx_index = nbulk
                    else:
                        # generate the image holder
                        block_hold = np.ndarray(shape=(part_bulk, Configuration.AXS_Y, Configuration.AXS_X))

                        # generate the max index
                        mx_index = part_bulk

                    # make the starting index
                    loop_start = kk * nbulk
                    idx_cnt = 0

                    Utils.log("Making mini file " + str(kk) + ".", "info")

                    # now loop through the images
                    for jj in range(loop_start, mx_index + loop_start):
                        # read in the image directly into the block_hold

                        master_tmp, master_tmp_head = fits.getdata(pass_files[jj], header=True)
                        master_tmp = master_tmp - master_tmp_head['sky']

                        tmp = Preprocessing.align_img(master_tmp, master_tmp_head, master_header)
                        Utils.log("Mini file " + str(kk) + " image " + str(jj) +
                                  " aligned. " + str(nfiles - cnt_img) + " remain.",
                                  "info")
                        block_hold[idx_cnt] = tmp - master_tmp_head['sky']
                        del tmp
                        del master_tmp

                        cnt_img = cnt_img + 1
                        # increase the iteration
                        idx_cnt += 1

                    # median the data into a single file
                    hold_data[kk] = np.median(block_hold, axis=0)
                    del block_hold
                    if kk < 10:
                        fits.writeto(Configuration.MASTER_TMP_DIRECTORY + "0" + str(kk) + "_tmp_master.fits",
                                     hold_data[kk], master_tmp_head, overwrite=True)
                    else:
                        fits.writeto(Configuration.MASTER_TMP_DIRECTORY + str(kk) + "_tmp_master.fits",
                                     hold_data[kk], master_tmp_head, overwrite=True)
            else:
                Utils.log("Legacy files found. Creating Master frame from these files. "
                          "Delete if you do not want this!", "info")
                hold_bulk = len(chk_tmp_files)
                # here is the 'holder'
                hold_data = np.ndarray(shape=(hold_bulk, Configuration.AXS_Y, Configuration.AXS_X))

                for kk, tmp_file in enumerate(chk_tmp_files):
                    master_tmp, master_tmp_head = fits.getdata(Configuration.MASTER_TMP_DIRECTORY + tmp_file,
                                                               header=True)
                    hold_data[kk] = master_tmp
                    master_header = master_tmp_head

            # median the mini-images into one large image
            master = np.median(hold_data, axis=0)

            master_header['MAST_COMB'] = 'median'
            master_header['NUM_MAST'] = nfiles
            master_header['MDN_SKY'] = img_mdn_sky

            _, master_sky, master_sig = sigma_clipped_stats(master, sigma=2.5)

            # now mask the bad parts of the image #### THIS WILL CHANGE PER FIELD!!!! LIKELY YOU SHOULD REMOVE#####
            master[:, 0:600] = np.random.normal(loc=master_sky,
                                                scale=master_sig,
                                                size=(10560,600))  # fill in the bad x
            master[9699:-1, :] = np.random.normal(loc=master_sky,
                                                  scale=master_sig,
                                                  size=(860,10560))

            if (master_sky < -1 * master_sig) | (master_sky > master_sky):
                master = master - master_sky

            #### END LIKELY REMOVE

            # write the image out to the master directory
            fits.writeto(Configuration.MASTER_DIRECTORY + file_name,
                         master, master_header, overwrite=True)
        else:
            master, master_header = fits.getdata(Configuration.MASTER_DIRECTORY + file_name, header=True)

        return master, master_header

    @staticmethod
    def get_image_stats(file_list):
        """ This function will determine the best set of images based on statistics on each image (airmass, sky level,
        star count, bad dates, etc.).

        :parameter file_list - This is the list of images including their directory information

        :return image_stats - A dataframe with the image statistics is returned and written to file
        """

        if os.path.exists(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_image_stats.txt') == 0:
            Utils.log("No legacy image statistics file found, generating a new one.", "info")

            # loop through the file list getting statistics on each image
            image_stats_list = []

            location = EarthLocation(lat=Configuration.TOROS_LATITUDE * u.deg,
                                     lon=Configuration.TOROS_LONGITUDE * u.deg,
                                     height=Configuration.TOROS_ELEVATION * u.m)


            for idx, file in enumerate(file_list):
                if idx % 10 == 0:
                    Utils.log("Working on statistics for the next 10 files. " +
                              str(len(file_list) - idx) + ' files remain.', 'info')

                # get the image
                img, header = fits.getdata(file, header=True)

                # get the start date of the observation
                date = file.split('/')[-3]

                # get the time of the image
                time_iso = header['Date'].split('T')[0] + ' ' +header['Date'].split('T')[1]
                jd = Time(time_iso, format='iso', scale='utc').jd

                # get the WCS of the position of the center pixel (10560x10560) > 5280/5280
                wcs = WCS(header)
                ra_cen, dec_cen = wcs.all_pix2world(Configuration.AXS_X / 2, Configuration.AXS_Y / 2, 1)

                # get the airmass
                target = SkyCoord(ra=ra_cen * u.deg, dec=dec_cen * u.deg, frame='icrs')
                altaz_frame = AltAz(obstime=time_iso, location=location)
                target_altaz = target.transform_to(altaz_frame)
                airmass = target_altaz.secz

                # get the number of stars in the image and the image background
                sky = header['sky']
                sky_sig = header['sky_sig']
                daofind = DAOStarFinder(fwhm=9.0, threshold=5. * sky_sig)
                sources = daofind(img - sky)
                nstars = len(sources[sources['flux'] > 0])

                image_list_row = {'file': file, 'date': date, 'jd': jd, 'ra': ra_cen, 'dec': dec_cen,
                                  'sky': sky, 'airmass': airmass, 'nstars': nstars}

                image_stats_list.append(image_list_row)

            # convert to data frame
            image_stats = pd.DataFrame(image_stats_list)

            # write to file
            image_stats.to_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_image_stats.txt', sep=' ')
        else:
            # warn about legacy file
            Utils.log("Legacy image statistics file being used. Delete it if you don't want it!", "info")

            image_stats = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_image_stats.txt',
                                      sep=' ', index_col=0)

        return image_stats