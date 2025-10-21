""" This set of functions is primarily used for photometery."""
from config import Configuration
from libraries.utils import Utils
from libraries.photometry import Photometry
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=Warning)
import matplotlib
matplotlib.use('TkAgg')


class Lightcurves:

    @staticmethod
    def generate_flux_files():
        """ This function will generate light curves for all of the stars in a given star list.

        :return Nothing is returned, but light curve files are generated for all stars
        """

        # pull in the star list for the photometry
        star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                                delimiter=' ',
                                header=0)

        # get the image list to difference
        files, date_dirs = Utils.get_all_files_per_field(Configuration.DIFFERENCED_DIRECTORY,
                                                     Configuration.FIELD,
                                                     'diff',
                                                     Configuration.FILE_EXTENSION)

        # make the output directories for the flux files
        output_dirs = []

        for dte in date_dirs:
            output_dirs.append(Configuration.DATA_DIRECTORY + "flux/" + dte)
            output_dirs.append(Configuration.DATA_DIRECTORY + "flux/" + dte + "/" + Configuration.FIELD)

        Utils.create_directories(output_dirs)

        # begin the algorithm to produce the photometry
        for idx, file in enumerate(files):

            fin_nme = file.split('.fits')[0] + '.flux'
            fin_nme = fin_nme.replace('/diff/', '/flux/')

            if os.path.isfile(fin_nme) == 1:
                Utils.log("Flux file " + fin_nme + " found. Skipping...", "info")

            # check to see if the flux file already exists
            if os.path.isfile(fin_nme) == 0:

                Utils.log("Working to extract flux from " + file + ".", "info")
                Photometry.single_frame_aperture_photometry(star_list,
                                                            file,
                                                            fin_nme)

        Utils.log("Differencing complete for " + Configuration.FIELD + ".", "info")

    @staticmethod
    def mk_raw_lightcurves():
        """ This function will create the individual raw light curve files for each star in the specific star list

        :return nothing is returned, but each light curve is output
        """

        # pull in the star list for the photometry
        star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
                                delimiter=' ',
                                header=0)

        # combine the flux from the flux files, and write the raw light curves
        Photometry.combine_flux_files(star_list)

        return
