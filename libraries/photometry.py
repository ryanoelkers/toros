""" This set of functions is primarily used for photometery."""
from config import Configuration
from libraries.utils import Utils
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
import numpy as np
import pandas as pd
import warnings
from astropy.time import Time
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=Warning)
import matplotlib
import logging
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
import matplotlib.pyplot as plt
from astropy.wcs import WCS


class Photometry:

    @staticmethod
    def add_variable_list(star_list, master_header):
        """ This function will read in the known variable star / transient list for the star field. It will get
        master frame photometry and x/y pixel position so the stars can be written to file.

        :parameter star_list - The data frame with the original star list
        :parameter master_header - The header of the master frame

        :return updated_list - The star list with the transients is returned
        """

        # add a column to link in star_list
        star_list['var_id'] = ' '

        # now get the variable / transient list
        known = pd.read_csv(Configuration.MASTER_DIRECTORY +
                            Configuration.FIELD + "_known_objects.csv", sep=",")

        # update the ra and dec in the known data frame to be in degrees
        known['ra'] = known.apply(lambda x: ((float(x['coords_ra'].split(' ')[0]) / 24) +
                                            (float(x['coords_ra'].split(' ')[1]) / 60 / 24) +
                                            (float(x['coords_ra'].split(' ')[2]) / 60 / 60 / 24)) * 360, axis=1)

        # be careful with negative declinations, you need to subtract and not add
        known['dec'] = known.apply(lambda x: float(x['coords_de'].split(' ')[0]) -
                                            (float(x['coords_de'].split(' ')[1]) / 60) -
                                            (float(x['coords_de'].split(' ')[2]) / 60 / 60) if float(x['coords_de'].split(' ')[0]) < 0 else float(x['coords_de'].split(' ')[0]) +
                                            (float(x['coords_de'].split(' ')[1]) / 60) +
                                            (float(x['coords_de'].split(' ')[2]) / 60 / 60), axis=1)

        # get the header file and convert to x/y pixel positions
        w = WCS(master_header)
        ra = known.ra.to_numpy()
        dec = known.dec.to_numpy()

        # convert to x, y
        x, y = w.all_world2pix(ra, dec, 0)

        # add the x/y to the star data frame
        known['x'] = x
        known['y'] = y

        # add the variable ID to the star list so it can be linked to the table
        for idx, row in known.iterrows():
            dist = np.min(np.sqrt((star_list.x - row.x) ** 2 + (star_list.y - row.y) ** 2))

            if dist < 5. / Configuration.PIXEL_SIZE:
                min_pos = np.argmin(np.sqrt((star_list.x - row.x) ** 2 + (star_list.y - row.y) ** 2))
                star_list.loc[min_pos, 'var_id'] = row.source_id

        # list of known variable stars
        var_list = star_list.var_id.unique().tolist()

        # filter the star list
        known_filtered = known[~known['source_id'].isin(var_list)].copy().reset_index(drop=True)
        known_filtered = known_filtered.drop(['coords_ra', 'coords_de'], axis=1)
        known_filtered['toros_field_id'] = Configuration.FIELD
        known_filtered['var_id'] = known_filtered['source_id']

        star_list = pd.concat([star_list, known_filtered], ignore_index=True)

        return star_list

    @staticmethod
    def single_frame_aperture_photometry(star_list, img_name, fin_name):
        """ This function will find the subtraction stars to use for the differencing, they will be the same stars for
        every frame. This will help in detrending later.

        :parameter star_list - The data frame with the list of stars to use for the subtraction
        :parameter img_name - The file to extract flux from
        :parameter fin_name - The name of the output flux file

        :return nothing is returned, but the flux file is written and output
        """

        # get the image for photometry
        img, header = fits.getdata(img_name, header=True)

        # get the various important header information
        time = Time(header['DATE'], format='isot', scale='utc')
        jd = time.jd

        # get the stellar positions from the master frame
        positions = np.transpose((star_list['xcen'], star_list['ycen']))

        # set up the aperture objects
        aperture = CircularAperture(positions, r=Configuration.APER_SIZE)
        aperture_area = aperture.area
        annulus_aperture = CircularAnnulus(positions,
                                           r_in=Configuration.ANNULI_INNER,
                                           r_out=Configuration.ANNULI_OUTER)

        # get the background stats
        aperstats = ApertureStats(img, annulus_aperture)
        bkg_mean = aperstats.mean
        bkg_total = bkg_mean * aperture.area

        # run the photometry to get the data table
        phot_table = aperture_photometry(img, aperture, method='exact')

        # extract the flux from the table
        # the sky was subtracted during the calibration and differencing steps, the raw photometry should be fine
        star_flux = np.array(phot_table['aperture_sum']) * Configuration.GAIN

        # calculate the expected photometric error
        star_error = np.abs(star_flux.astype(float) + star_list['master_flux'].to_numpy().astype(float))
        bkg_error = np.abs(header['sky'] + bkg_mean) * aperture_area * Configuration.GAIN

        # combine sky and signal error in quadrature
        star_flux_err = np.sqrt(star_error + bkg_error)

        # combine the fluxes
        flux = star_flux.astype(float) + star_list['master_flux'].to_numpy().astype(float)
        flux_er = np.sqrt(star_flux_err.astype(float) ** 2 + star_list['master_flux_er'].to_numpy().astype(float) ** 2)

        # convert to magnitude
        mag = 25. - 2.5 * np.log10(flux)
        mag_nbkg = 25. - 2.5 * np.log10(flux - bkg_total)
        mag_er = (np.log(10.) / 2.5) * (flux_er / flux)

        # initialize the master magnitude of all frames
        m_mag = star_list['master_mag'].to_numpy()

        # the magnitude difference between the science frame magnitude and the master frame
        dmag = mag[~np.isnan(mag)] - m_mag[~np.isnan(mag)]

        # add "chip" to the star_list
        star_list['chip'] = 1
        kk = 1
        for idx in range(0, Configuration.AXS_X, Configuration.CHP_X):
               for idy in range(0, Configuration.AXS_Y, Configuration.CHP_Y):
                    star_list['chip'] = np.where((star_list.xcen > idx) & (star_list.xcen < idx + 1320) &
                                                (star_list.ycen > idy) & (star_list.ycen < idy + 5280),
                                                 kk, star_list.chip)
                    kk = kk + 1

        # initialize the offset vector
        off = np.zeros(len(mag))

        # set up holders for interpolation
        f_mags = np.arange(6, 16) + 0.5
        n_mags = np.zeros(len(f_mags))

        # make sure there is an object type in the file
        try:
            star_list['object_type'] = np.where(star_list['object_type'].isna(), 'star', star_list['object_type'])
        except:
            star_list['object_type'] = 'star'

        # loop through each chip to find the zeropoint offset
        for chp in range(1, 17):
            for mag_idx, m_lw in enumerate(f_mags):
                # get the zeropoint using non-nan stars in the chip between the magnitude range
                dmag_bin = dmag[(mag[~np.isnan(mag)] > m_lw) &
                                (mag[~np.isnan(mag)] < m_lw + 1) &
                                (star_list[~np.isnan(mag)].chip.to_numpy() == chp) &
                                (star_list[~np.isnan(mag)].object_type.to_numpy() == 'star')]

                # sigma clip outliers
                dmag_mn, dmag_md, dmag_sg = sigma_clipped_stats(np.array(dmag_bin, dtype=float), sigma=2)
                n_mags[mag_idx] = dmag_md

            # interpolate to all magnitudes
            off[star_list.chip.to_numpy() == chp] = np.interp(mag[star_list.chip.to_numpy() == chp], f_mags, n_mags)

        # now correct the magnitudes for exposure time
        mag = mag + 2.5 * np.log10(Configuration.EXP_TIME)

        # replace nans with -9.999999
        off = np.where(np.isnan(off), -9.999999, off)
        mag = np.where(np.isnan(mag), -9.999999, mag)

        # generate the final flux file
        flux_file = star_list.copy().reset_index(drop=True)
        flux_file['flux'] = flux
        flux_file['flux_er'] = flux_er
        flux_file['mag'] = mag
        flux_file['mag_nbkg'] = mag_nbkg
        flux_file['mag_er'] = mag_er
        flux_file['sky'] = header['SKY']
        flux_file['bkg'] = bkg_mean
        flux_file['jd'] = jd
        flux_file['zpt'] = off
        flux_file['exp_time'] = Configuration.EXP_TIME

        flux_file.to_csv(fin_name, header=True, index=False)
        return

    @staticmethod
    def combine_flux_files():
        """ Deconstruct the flux files to single files for each light curve.

        :return - Nothing is being returned, but the raw files are output to disk
        """

        # pull in the star list for the photometry
        if Configuration.KNOWN_VARIABLES == 'Y':
            # if there are known variables or transients get the updated star list
            star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list_updated.txt',
                                    delimiter=' ',
                                    header=0)
        else:
            # if there are no known transients then get the old star list
            star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                                    delimiter=' ',
                                    header=0)

        # get the flux files to read in
        files, dates = Utils.get_all_files_per_field(Configuration.FLUX_DIRECTORY,
                                                     Configuration.FIELD,
                                                     'flux',
                                                     '.flux')
        nfiles = len(files)  # the number of flux files to combine
        nstars = len(star_list)  # The number of stars to generate light curves for
        Utils.log(str(nfiles) + " flux files found for " + Configuration.FIELD + ".", "info")

        # make the holders for the light curves
        jd = np.zeros(nfiles)
        mag = np.zeros((nstars, nfiles))
        err = np.zeros((nstars, nfiles))
        err_scl = np.zeros((nstars, nfiles))
        trd = np.zeros((nstars, nfiles)) - 9.999999
        zpt = np.zeros((nstars, nfiles))
        sky = np.zeros((nstars, nfiles))
        bkg = np.zeros((nstars, nfiles))

        for idy, file in enumerate(files):

            # read in the data frame with the flux information
            img_flux = pd.read_csv(file, header=0)

            if idy == 0:
                star_list['chip'] = img_flux['chip'].to_numpy()
                star_list['object_type'] = img_flux['object_type'].to_numpy()

            # set the data to the numpy array
            jd[idy] = img_flux.loc[0, 'jd']
            mag[:, idy] = img_flux['mag'].to_numpy()
            err[:, idy] = img_flux['mag_er'].to_numpy()
            err_scl[:, idy] = img_flux['mag_er'].to_numpy()
            zpt[:, idy] = img_flux['zpt'].to_numpy()
            sky[:, idy] = img_flux['sky'].to_numpy()
            bkg[:, idy] = img_flux['bkg'].to_numpy()

            if (idy % 100 == 0) & (idy > 0):
                Utils.log("100 flux files read. " + str(nfiles - idy - 1) + ' files remain.', "info")

        ## IF THERE IS AN OBJECT IN THE FIELD (LIKE 47-TUC) then block it
        star_list['bd_stars'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                         (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

        # convert the master frame magnitude to e/s
        star_list['master_mag'] = star_list['master_mag'] + 2.5 * np.log10(Configuration.EXP_TIME)
        src_id = star_list.source_id.to_numpy()

        for idy, row in star_list.iterrows():

            # get the distance to all stars
            dd = np.sqrt((row.xcen - star_list.xcen.to_numpy()) ** 2 +
                         (row.ycen - star_list.ycen.to_numpy()) ** 2)

            # get the difference in magnitude
            dmag = np.abs(row.master_mag - star_list.master_mag.to_numpy())

            # only get nearby stars of similar magnitude, on the same chip, aren't variables, and aren't in a bad area
            vv = np.argwhere((dmag < 2) & (dd > 0) & (star_list['chip'] == row.chip) &
                             (star_list['bd_stars'] == 0) & (star_list['object_type'] == 'star')).reshape(-1)
            vv_all = np.argwhere((dmag < 0.5) & (dd > 0) &
                                 (star_list['bd_stars'] == 0) & (star_list['object_type'] == 'star')).reshape(-1)

            if len(vv) > 0:
                # generate the trend for the trend stars
                for idz in range(len(jd)):
                    _, trd[idy, idz], _ = sigma_clipped_stats(mag[vv, idz] -
                                                              star_list.loc[vv].master_mag.to_numpy(), sigma=3)

            else:
                # if no stars exist, then go ahead and use all stars in the frame, not just your chip
                for idz in range(len(jd)):
                    _, trd[idy, idz], _ = sigma_clipped_stats(mag[vv_all, idz] -
                                                              star_list.loc[vv_all].master_mag.to_numpy(), sigma=3)

            # get the updates error based on similar stars
            _, _, ss = sigma_clipped_stats(mag[vv_all] - trd[idy], sigma=3, axis=1)
            m_er, _, _ = sigma_clipped_stats(err[idy], sigma=3)
            if len(ss) > 0:
                sigma_scale = m_er / np.quantile(ss, 0.01)
            else:
                sigma_scale = 1.

            # rescale the errors
            err_scl[idy] = err[idy] / sigma_scale

            if (idy % 1000 == 0) & (idy > 0):
                Utils.log("1000 stars had their trend found. " +
                          str(nstars - idy - 1) + " stars remain.", "info")

        # write out the light curve data
        Photometry.write_light_curves(nstars, jd, mag, err, err_scl, trd, zpt, sky, bkg, src_id)

        return

    @staticmethod
    def write_light_curves(nstars, jd, mag, er, er_scl, trd, zpt, sky, bkg, src_id):
        """ This function will write the flux columns to light curves for each src_id

        :parameters - nstars - the number of stars to have light curves written
        :parameters - jd - the numpy array of julian dates (one per file)
        :parameters - mag - the magnitudes for each star at each jd
        :parameters - er - the photometric error for each star at each time
        :parameters - er_scl - the photometric error for each star with scaling
        :parameters - trd - the trend from nearby similar magnitude stars
        :parameters - zpt - the zeropiont using all stars in the frame
        :parameters - sky - the median sky background
        :parameters - bkg - the local background for the source
        :parameters - src_id - nstars long array of source ids

        :return - nothing is returned, but the light curve files are written
        """

        # initialize the light curve data frame
        lc = pd.DataFrame(columns=['jd', 'mag', 'err', 'trd', 'zpt', 'sky', 'bkg'])

        Utils.log("Starting light curve writing for " + str(nstars) + " stars.", "info")

        for idx in range(0, nstars):
            star_id = str(src_id[idx])

            # add the time, magnitude and error to the data frame
            lc['jd'] = np.around(jd, decimals=6)
            lc['mag'] = np.around(mag[idx, :], decimals=6)
            lc['err'] = np.around(er_scl[idx, :], decimals=6)
            lc['trd'] = np.around(trd[idx, :], decimals=6)
            lc['org_err'] = np.around(er[idx, :], decimals=6)
            lc['zpt'] = np.around(zpt[idx, :], decimals=6)
            lc['sky'] = np.around(sky[idx, :], decimals=0)
            lc['bkg'] = np.around(bkg[idx, :], decimals=0)

            # make sure the data is in order!
            lc = lc.sort_values(by = 'jd').reset_index(drop=True)
            lc['err'] = np.where(lc['mag'] < 0, -9.999999, lc['err'])
            lc['org_err'] = np.where(lc['mag'] < 0, -9.999999, lc['org_err'])
            lc['trd'] = np.where(lc['mag'] < 0, -9.999999, lc['trd'])
            lc['zpt'] = np.where(lc['mag'] < 0, -9.999999, lc['zpt'])
            lc['sky'] = np.where(lc['mag'] < 0, -9.999999, lc['sky'])
            lc['bkg'] = np.where(lc['mag'] < 0, -9.999999, lc['bkg'])
            lc['mag'] = np.where(lc['mag'] < 0, -9.999999, lc['mag'])

            # write the new file
            lc[['jd', 'mag', 'err', 'trd', 'org_err', 'zpt', 'sky', 'bkg']].to_csv(Configuration.LIGHTCURVE_DIRECTORY +
                                                                                   Configuration.FIELD + "/" +
                                                                                   Configuration.FIELD +"_" + star_id + ".lc",
                                                                                   sep=" ", index=False, na_rep='-9.999999')

            if (idx > 0) & (idx / 10000 % 1 == 0):
                Utils.log("10000 stars have had their light curves written. " +
                          str(nstars - idx - 1) + " stars remain. ", "info")

        Utils.log("All light curves written.", "info")

        return
