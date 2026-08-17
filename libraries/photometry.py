""" This set of functions is primarily used for photometery."""
from config import Configuration
from libraries.utils import Utils
from astropy.io import fits
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
import astropy.units as u
from astropy.stats import sigma_clipped_stats as scs

class Photometry:

    @staticmethod
    def remove_systematics(star_list, trend_stars=500):
        """ This function will use a combination of image statistics and ensemble light curves to remove common
        systematics from each light curve.

        :parameter star_list - The data frame with the original star list
        :parameter trend_stars - The number of stars you want to use for the detrending search, default 500

        :return nothing is returned, but the files are written with a new magnitude column
        """

        # first get the zeropoint determination from the flux measurements
        flux_files, flux_dates = Utils.get_all_files_per_field(Configuration.FLUX_DIRECTORY,
                                                     Configuration.FIELD,
                                                     'flux',
                                                     '.flux')

        #set up the zpt holder
        zpt_offset = np.zeros(len(flux_files))
        jd_offset = np.zeros(len(flux_files))
        Utils.log("Pulling zeropoint offset from flux files...", "info")
        for fidx, ffile in enumerate(flux_files):
            flux_df = pd.read_csv(ffile, nrows=1, header=0)
            zpt_offset[fidx] = flux_df['zpt'].values
            jd_offset[fidx] = flux_df['jd'].values

            if fidx % 50 == 0:
                Utils.log("..." + str(int(np.around(fidx / len(flux_files) * 100, decimals=0))) + "% complete.",
                          "info")

        # sort to the correct time
        time_srt = np.argsort(jd_offset)
        zpt_offset = zpt_offset[time_srt]

        # open the error file for writing
        f = open(Configuration.LIGHTCURVE_FIELD_DIRECTORY + Configuration.FIELD + "_errors.txt", "w")
        header = 'name mag rms erms orms x y chip object_type\n'
        f.write(header)

        for idx, row in star_list.iterrows():
            if idx % 1000 == 0:
                Utils.log("Working to detrend the next 100 light curves. " +
                          str(len(star_list) - idx - 1) + " light curves remain.", "info")

            # read in the light curve
            if row.chip < 10:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                                 '0'+ str(row.chip) + '/' + Configuration.FIELD + '_' + row.source_id + '.lc',
                                 sep=' ')
            else:
                lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                                 str(row.chip) + '/' + Configuration.FIELD + '_' + row.source_id + '.lc',
                                 sep=' ')

            # determine the offset in magnitude and distance from the target star
            star_list['dmag'] = np.abs(row.master_mag - star_list['master_mag'])
            star_list['dist'] = np.sqrt((star_list.y - row.y) ** 2 + (star_list.x - row.x) ** 2)

            # grab the list of trends stars based on whether or not the star is in 47-Tuc
            if row.gc_star == 0:
                trend_list = star_list[(star_list.gc_star == 0) &
                                       (star_list.dist > Configuration.APER_SIZE) &
                                       (star_list.object_type == 'Star')].copy().sort_values(by='dmag')[0:trend_stars].reset_index(drop=True)
            else:
                trend_list = star_list[(star_list.gc_star == 1) &
                                       (star_list.dist > Configuration.APER_SIZE) &
                                       (star_list.object_type == 'Star')].copy().sort_values(by='dmag')[0:trend_stars].reset_index(drop=True)

            # set up the empty collection vectors
            cols = {}
            col_nme = []
            mgs = np.zeros(len(trend_list))

            kk = 0  # initialize the star names
            for idy, rw in trend_list.iterrows():
                # read in the trend light curves
                if rw.chip < 10:
                    tr = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                                     '0' + str(rw.chip) + '/' + Configuration.FIELD + '_' + str(rw.source_id) + '.lc',
                                     sep=' ')
                else:
                    tr = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                                     str(rw.chip) + '/' + Configuration.FIELD + '_' + str(rw.source_id) + '.lc',
                                     sep=' ')

                # only accept the star for detrending if it has non-zero values
                if len(tr[tr.mag > 0]) > 0:
                    # subtract the median value from the light curve
                    cols['mag_' + str(kk)] = tr.mag.to_numpy() - tr[tr.mag > 0].mag.median()
                    # grab the value you subtracted for safe keeping
                    mgs[idy] = tr[tr.mag > 0].mag.median()
                    # append the column name list to make the data frame
                    col_nme.append('mag_' + str(kk))
                    # update kk
                    kk = kk + 1
                del tr
            del trend_list

            # make the trend_df
            trend_df = pd.DataFrame(cols, columns=col_nme)

            # set up the trend vector
            lc['trd'] = np.zeros(len(lc))
            lc['zpt'] = zpt_offset

            # loop through each day finding the appropriate offset
            for ii in range(len(lc)):
                # get the offset for the specific day
                offsets = trend_df.loc[ii].to_numpy()

                # initialize the holding lists
                mg = []
                off = []

                # get the offset list in X.X mag chunks to interpolate around outliers
                for jj in np.arange(np.min(mgs), np.max(mgs), 0.1):
                    # ignore any empty space
                    if len(offsets[(mgs >= jj) & (mgs < jj + 0.1) & (offsets > -20)]) > 0:
                        # get the current magnitude bin
                        mg.append(jj + 0.05)
                        # get the median offset with 2.5 sigma clipping
                        _, mg_mdn, _ = scs(offsets[(mgs >= jj) & (mgs < jj + 0.1) & (offsets > -20)], sigma=2.5)
                        # the value shouldn't be nan, but if it is, then just use the median of the whole day
                        if np.isnan(mg_mdn):
                            off.append(np.median(offsets[(mgs >= jj) & (mgs < jj + 0.1) & (offsets > -20)]))
                        else:
                            off.append(mg_mdn)
                # get the trend value for this observations
                trd = np.interp(lc.mag[ii], mg, off)

                # if it is nan, then just use the median for the full day
                if np.isnan(trd):
                    lc.loc[ii, 'trd'] = np.nanmedian(off)
                else:
                    lc.loc[ii, 'trd'] = trd
            del trend_df

            lc = lc.rename(columns={'mag': 'raw'})
            lc['mag'] = lc['raw'] - lc['trd']

            # rearrange the light curve information for better outputs
            lc = lc[['jd', 'mag', 'err', 'raw', 'trd', 'zpt', 'sky', 'bkg', 'x', 'y', 'nstars', 'airmass']]

            # update bad data
            lc.raw = np.where(lc.raw < 0, -9.9999, lc.raw)
            lc.mag = np.where(lc.raw < 0, -9.9999, lc.mag)

            # calculate statistics for the error analysis
            mag, _, full_rms = scs(lc[(lc.mag > 0) & (lc.err > 0)].mag, sigma=2.5)
            lc['dys'] = lc.jd.to_numpy().astype('int')

            rms_vals = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'std'}).to_numpy().flatten()
            num_obs = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'count'}).to_numpy().flatten()

            erms = lc[(lc.mag > 0) & (lc.err > 0)].err.mean()
            try:
                rms = np.median(rms_vals[num_obs >= 6])
            except:
                rms = full_rms

            # output the statistics
            line = (Configuration.FIELD + "_" + str(row.source_id) + ".lc" + " " +
                    str(np.around(row.master_mag, decimals=4)) + " " +
                    str(np.around(rms, decimals=4)) + " " +
                    str(np.around(erms, decimals=4)) + " " +
                    str(np.around(full_rms, decimals=4)) + " " +
                    str(np.around(row.xcen, decimals=2)) + " " +
                    str(np.around(row.ycen, decimals=2)) + " " +
                    str(int(row.chip)) + " " +
                    str(row.object_type) + "\n")
            f.write(line)

            lc = lc.drop(columns=['dys'])

            # update print formats
            lc.mag = lc.mag.map(lambda x: '%0.4f' % x)
            lc.raw = lc.raw.map(lambda x: '%0.4f' % x)
            lc.err = lc.err.map(lambda x: '%0.4f' % x)
            lc.trd = lc.trd.map(lambda x: '%0.4f' % x)
            lc.zpt = lc.zpt.map(lambda x: '%0.4f' % x)
            lc.x = lc.x.map(lambda x: '%d' % x)
            lc.y = lc.y.map(lambda x: '%d' % x)
            lc.nstars = lc.nstars.map(lambda x: '%d' % x)
            lc.airmass = lc.airmass.map(lambda x: '%0.3f' % x)

            # write out lc
            if row.chip < 10:
                lc.to_csv(Configuration.LIGHTCURVE_FIELD_DETREND_DIRECTORY + '/0' + str(row.chip) +
                          '/' + Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                          sep=" ", header=True, index=False)
            else:
                lc.to_csv(Configuration.LIGHTCURVE_FIELD_DETREND_DIRECTORY + '/' + str(row.chip) +
                          '/' + Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                          sep=" ", header=True, index=False)
            del lc

        # now close the file
        f.close()

        return

    @staticmethod
    def add_lsst_list(star_list, master_header):
        """ This function will read a list of objects from an lsst source (like the DIA source/object table) and insert
        the objects into the TOROS star list after linking to the objects within a very close position

        :parameter star_list - The data frame with the original star list
        :parameter master_header - The header of the master frame

        :return updated_list - The star list with the transients is returned
        """

        # now get the variable / transient list
        known = pd.read_csv(Configuration.DATA_DIRECTORY + "/lsst/lsst_data_47tuc_variables.csv",
                            delimiter=',',
                            header=0,
                            low_memory=False,
                            index_col=0)

        # remove mulitple obserations of the same star and rename the ra and dec columns
        # known = known.groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean', 'psfFlux':'max'}).reset_index()
        # known['psfMag'] = known.apply(lambda x: 31.4 - 2.5 * np.log10(x.psfFlux), axis=1)
        known = known.groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean'}).reset_index()
        known = known.rename(columns={'coord_ra': 'ra', 'coord_dec': 'dec', 'diaObjectId': 'source_id'})

        # get the header file and convert to x/y pixel positions
        w = WCS(master_header)
        ra = known.ra.to_numpy()
        dec = known.dec.to_numpy()

        # convert to x, y
        x, y = w.all_world2pix(ra, dec, 0)

        # add the x/y to the lsst variable data frame
        known['x'] = x
        known['y'] = y

        # add the lsst id to all objects in the star list
        star_list['lsst_id'] = '--'

        # add the variable ID to the star list so it can be linked to the table
        for idx, row in known.iterrows():
            dist = np.min(np.sqrt((star_list.x - row.x) ** 2 + (star_list.y - row.y) ** 2))

            if np.min(dist) < 5:
                min_pos = np.argmin(np.sqrt((star_list.x - row.x) ** 2 + (star_list.y - row.y) ** 2))

                if star_list.loc[min_pos, 'object_type'] == 'Star':
                    star_list.loc[min_pos, 'object_type'] = 'LSST'
                    star_list.loc[min_pos, 'lsst_id'] = row.source_id.astype(int)

                if star_list.loc[min_pos, 'object_type'] == 'Var':
                    star_list.loc[min_pos, 'lsst_id'] = row.source_id.astype(int)

                if star_list.loc[min_pos, 'object_type'] == 'Xray':
                    star_list.loc[min_pos, 'lsst_id'] = row.source_id.astype(int)

        # list of known variable stars
        var_star_list = star_list.lsst_id.unique().tolist()

        # filter the star list
        known_filtered = known[~known['source_id'].isin(var_star_list)].copy().reset_index(drop=True)
        known_filtered['toros_field_id'] = Configuration.FIELD
        known_filtered['var_id'] = known_filtered['source_id']
        known_filtered['var_type'] = '--'
        known_filtered['var_period'] = 0
        known_filtered['object_type'] = 'LSST'
        known_filtered['lsst_id'] = known_filtered['source_id']

        star_list = pd.concat([star_list, known_filtered], ignore_index=True)

        return star_list

    @staticmethod
    def add_variable_list(star_list, master_header):
        """ This function will read in the known variable star / transient list for the star field. It will get
        master frame photometry and x/y pixel position so the stars can be written to file.

        :parameter star_list - The data frame with the original star list
        :parameter master_header - The header of the master frame

        :return updated_list - The star list with the transients is returned
        """

        # add a column to link in star_list
        star_list['var_id'] = '--'
        star_list['var_type'] = '--'
        star_list['var_period'] = 0
        star_list['object_type'] = 'Star'

        # now get the variable / transient list
        known = pd.read_csv(Configuration.MASTER_DIRECTORY + "/known_objects/"
                            + Configuration.FIELD +"_known_objects.csv", sep=",")

        # update the ra and dec in the known data frame to be in degrees
        known['ra'] = known.apply(lambda x: ((float(x['coords'].split(' ')[0]) / 24) +
                                            (float(x['coords'].split(' ')[1]) / 60 / 24) +
                                            (float(x['coords'].split(' ')[2]) / 60 / 60 / 24)) * 360, axis=1)

        # be careful with negative declinations, you need to subtract and not add
        known['dec'] = known.apply(lambda x: float(x['coords'].split(' ')[3]) -
                                            (float(x['coords'].split(' ')[4]) / 60) -
                                            (float(x['coords'].split(' ')[5]) / 60 / 60) if float(x['coords'].split(' ')[3]) < 0 else float(x['coords'].split(' ')[3]) +
                                            (float(x['coords'].split(' ')[4]) / 60) +
                                            (float(x['coords'].split(' ')[5]) / 60 / 60), axis=1)

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

            try:
                nme_chk = star_list[star_list.source_id == int(row.source_id)].index.values[0]
            except:
                nme_chk = -99

            if nme_chk >= 0:
                star_list.loc[nme_chk, 'var_id'] = row.source_id
                star_list.loc[nme_chk, 'var_type'] = row.var_type
                star_list.loc[nme_chk, 'var_period'] = row.var_period
                star_list.loc[nme_chk, 'object_type'] = row.object_type

            elif (dist < 5. / Configuration.PIXEL_SIZE) & (nme_chk < 0):
                min_pos = np.argmin(np.sqrt((star_list.x - row.x) ** 2 + (star_list.y - row.y) ** 2))
                star_list.loc[min_pos, 'var_id'] = row.source_id
                star_list.loc[min_pos, 'var_type'] = row.var_type
                star_list.loc[min_pos, 'var_period'] = row.var_period
                star_list.loc[min_pos, 'object_type'] = row.object_type

        # list of known variable stars
        var_list = star_list.var_id.unique().tolist()

        # filter the star list
        known_filtered = known[~known['source_id'].isin(var_list)].copy().reset_index(drop=True)
        known_filtered = known_filtered.drop(['coords'], axis=1)
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

        # get the cleaned image for the x/y position
        cln_nme = img_name.split('diff')[0] + 'clean' + img_name.split('diff')[1].split('ad.fits')[0] + '.fits'
        cln_header = fits.getheader(cln_nme)
        w_cln = WCS(cln_header)
        x_cln, y_cln = w_cln.all_world2pix(star_list.ra * u.deg, star_list.dec * u.deg, 0)

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
        bkg_total = np.abs(header['sky'] + bkg_mean) * aperture_area

        # run the photometry to get the data table
        phot_table = aperture_photometry(img, aperture, method='exact')

        # extract the flux from the table
        # the sky was subtracted during the calibration and differencing steps, the raw photometry should be fine
        # star_flux = np.array(phot_table['aperture_sum'] - bkg_mean * aperture_area) * Configuration.GAIN
        star_flux = np.array(phot_table['aperture_sum']) * Configuration.GAIN

        # calculate the expected photometric error
        star_error = np.abs(star_flux.astype(float) + star_list['master_flux'].to_numpy().astype(float))
        bkg_error = bkg_total * Configuration.GAIN

        # combine sky and signal error in quadrature
        star_flux_err = np.sqrt(star_error + bkg_error)

        # combine the fluxes
        flux = star_flux.astype(float) + star_list['master_flux'].to_numpy().astype(float)
        flux_er = np.sqrt(star_flux_err.astype(float) ** 2 + star_list['master_flux_er'].to_numpy().astype(float) ** 2)

        # convert to magnitude
        mag = 25. - 2.5 * np.log10(flux)
        mag_er = (np.log(10.) / 2.5) * (flux_er / flux)
        # mg = 25 - 2.5 * np.log10(sf + star_list['master_flux'].to_numpy().astype(float))

        # now correct the magnitudes for exposure time
        mag = mag + 2.5 * np.log10(Configuration.EXP_TIME)

        # replace nans with -9.999999
        mag = np.where(np.isnan(mag), -9.999999, mag)

        # zeropoint selection
        zpt_mn, zpt_mdn, zpt_std = scs(mag[(mag > 0) & (mag < 18)] - star_list[(mag > 0) & (mag < 18)].master_mag,
                                       sigma=2.)
        zpt = np.around(zpt_mdn, decimals=6)

        # generate the final flux file
        flux_file = star_list.copy().reset_index(drop=True)
        flux_file['flux'] = flux
        flux_file['flux_er'] = flux_er
        flux_file['mag'] = mag
        flux_file['mag_er'] = mag_er
        flux_file['bkg'] = bkg_error
        flux_file['jd'] = jd
        flux_file['exp_time'] = Configuration.EXP_TIME
        flux_file['x'] = x_cln
        flux_file['y'] = y_cln
        flux_file['zpt'] = np.zeros(len(flux_file)) + zpt
        flux_file.to_csv(fin_name, header=True, index=False)
        return

    @staticmethod
    def combine_flux_files(star_list):
        """ Deconstruct the flux files to single files for each light curve.

        :parameter star_list - The star list to be used for photometry

        :return - Nothing is being returned, but the raw files are output to disk
        """

        # get the flux files to read in
        files, dates = Utils.get_all_files_per_field(Configuration.FLUX_DIRECTORY,
                                                     Configuration.FIELD,
                                                     'flux',
                                                     '.flux')
        nfiles = len(files)  # the number of flux files to combine
        nstars = len(star_list)  # The number of stars to generate light curves for
        Utils.log(str(nfiles) + " flux files found for " + Configuration.FIELD + ".", "info")

        # get the image statisitics
        img_stats = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_image_stats.txt',
                                sep=' ', index_col=0)

        # make the holders for the light curves
        jd = np.zeros(nfiles)
        chip = star_list['chip'].to_numpy()
        mag = np.zeros((nstars, nfiles))
        err = np.zeros((nstars, nfiles))
        sky = np.zeros(nfiles)
        bkg = np.zeros((nstars, nfiles))
        x = np.zeros((nstars, nfiles))
        y = np.zeros((nstars, nfiles))
        airmass = np.zeros(nfiles)
        nstrs = np.zeros(nfiles)

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
            sky[idy] = img_stats.loc[idy, 'sky']
            bkg[:, idy] = img_flux['bkg'].to_numpy()
            x[:, idy] = img_flux['x'].to_numpy()
            y[:, idy] = img_flux['y'].to_numpy()
            airmass[idy] = img_stats.loc[idy, 'airmass']
            nstrs[idy] = img_stats.loc[idy, 'nstars']

            if (idy % 100 == 0) & (idy > 0):
                Utils.log("100 flux files read. " + str(nfiles - idy - 1) + ' files remain.', "info")

        # get the source id list
        src_id = star_list.source_id.to_numpy()

        # write out the light curve data
        Photometry.write_raw_light_curves(nstars, jd, mag, err, sky, bkg, x, y, airmass, nstrs, src_id, chip)

        return

    @staticmethod
    def write_raw_light_curves(nstars, jd, mag, er, sky, bkg, x, y, airmass, nstars_img, src_id, chip):
        """ This function will write the flux columns to light curves for each src_id

        :parameters - nstars - the number of stars to have light curves written
        :parameters - jd - the numpy array of julian dates (one per file)
        :parameters - mag - the magnitudes for each star at each jd
        :parameters - er - the photometric error for each star at each time
        :parameters - sky - the median sky background
        :parameters - bkg - the local background for the source
        :parameters - x - the x position of the star on the detector
        :parameters - y - the y position of the star on the detector
        :parameters - airmass - the airmass of the image
        :parameters - nstars_img - the number of stars on the image
        :parameters - src_id - nstars long array of source ids
        :parameters - chip - the chip number the star is on (to simplify writing and searching)

        :return - nothing is returned, but the light curve files are written
        """

        # initialize the light curve data frame
        lc = pd.DataFrame(columns=['jd', 'mag', 'err', 'bkg', 'x', 'y', 'sky', 'airmass', 'nstars'])

        Utils.log("Starting light curve writing for " + str(nstars) + " stars.", "info")

        for idx in range(0, nstars):
            star_id = str(src_id[idx])

            # add the time, magnitude and error to the data frame
            lc['jd'] = np.around(jd, decimals=6)
            lc['mag'] = np.around(mag[idx, :], decimals=6)
            lc['err'] = np.around(er[idx, :], decimals=6)
            lc['sky'] = np.around(sky, decimals=2)
            lc['bkg'] = np.around(bkg[idx, :], decimals=2)
            lc['x'] = np.around(x[idx, :], decimals=0)
            lc['y'] = np.around(y[idx, :], decimals=0)
            lc['airmass'] = np.around(airmass, decimals=3)
            lc['nstars'] = np.around(nstars_img, decimals=0)

            # make sure the data is in order!
            lc = lc.sort_values(by = 'jd').reset_index(drop=True)
            lc['err'] = np.where(lc['mag'] < 0, -9.999999, lc['err'])
            lc['sky'] = np.where(lc['mag'] < 0, -9.999999, lc['sky'])
            lc['bkg'] = np.where(lc['mag'] < 0, -9.999999, lc['bkg'])
            lc['mag'] = np.where(lc['mag'] < 0, -9.999999, lc['mag'])
            lc['x'] = np.where(lc['mag'] < 0, -9.999999, lc['x'])
            lc['y'] = np.where(lc['mag'] < 0, -9.999999, lc['y'])

            # write the new file
            if chip[idx] < 10:
                lc[['jd', 'mag', 'err', 'bkg', 'x',
                    'y', 'sky', 'airmass', 'nstars']].to_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY
                                                             + '/0' + str(chip[idx]) + '/'
                                                             + Configuration.FIELD + '_' + star_id + '.lc',
                                                             sep=' ', index=False, na_rep='-9.999999')
            else:
                lc[['jd', 'mag', 'err', 'bkg', 'x',
                    'y', 'sky', 'airmass', 'nstars']].to_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY + '/' +
                                                             str(chip[idx]) + '/' +
                                                             Configuration.FIELD + '_' + star_id + '.lc',
                                                             sep=' ', index=False, na_rep='-9.999999')

            if (idx > 0) & (idx / 10000 % 1 == 0):
                Utils.log("10000 stars have had their light curves written. " +
                          str(nstars - idx - 1) + " stars remain. ", "info")

        Utils.log("All light curves written.", "info")

        return
