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
from config import Configuration
from libraries.utils import Utils
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
import numpy as np
import pandas as pd
from astropy.time import Time

# remove stars near 47 Tuc and the smallcluster
star_list = pd.read_csv("/Users/yuw816/OneDrive - The University of Texas-Rio Grande Valley/Research/TOROS/master/"
                        + Configuration.FIELD + "_star_list_updated.txt", sep=' ', low_memory=False, index_col=0)
bd_star = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                   (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)
star_list = star_list[bd_star == 0].copy().reset_index(drop=True)

dir = "/Users/yuw816/Data/toros/commissioning/diff/"
files = np.sort(Utils.get_all_files_per_field(dir, 'FIELD_0e.001', 'diff', '.fits')[0])
nfiles = len(files)

# get the image for photometry
for idx, file in enumerate(files):
        img, header = fits.getdata(file, header=True)

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
        bkg_total = 0 # bkg_mean * aperture.area

        # run the photometry to get the data table
        phot_table = aperture_photometry(img, aperture, method='exact')

        # extract the flux from the table
        # the sky was subtracted during the calibration and differencing steps, the raw photometry should be fine
        star_flux = np.array((phot_table['aperture_sum'] - bkg_total) / Configuration.EXP_TIME) * Configuration.GAIN

        # calculate the expected photometric error
        star_error = np.sqrt(star_flux)
        bkg_error = np.sqrt(bkg_total)

        # combine sky and signal error in quadrature
        star_flux_err = np.sqrt(star_error ** 2 + bkg_error ** 2)

        # convert to magnitude
        mag = 25. - 2.5 * np.log10(star_flux)
        mag_er = (np.log(10.) / 2.5) * (star_flux_err / star_flux)

        # replace nans with -9.999999
        mag = np.where(np.isnan(mag), -9.999999, mag)

        # generate the final flux file
        flux_file = star_list.copy().reset_index(drop=True)
        flux_file['flux'] = star_flux
        flux_file['flux_er'] = star_flux_err
        flux_file['mag'] = mag
        flux_file['mag_er'] = mag_er
        flux_file['sky'] = header['SKY']
        flux_file['bkg'] = bkg_mean
        flux_file['jd'] = jd
        flux_file['exp_time'] = Configuration.EXP_TIME

        flux_file.to_csv(file.split('.fits')[0]+'.flux', header=True, index=False)
