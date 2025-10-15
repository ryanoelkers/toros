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
from libraries.photometry import Photometry
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
star_list['bd_stars'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
                                 (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)
star_list = star_list[star_list.bd_stars == 0].copy().reset_index(drop=True)

# add "chip" to the star_list
star_list['chip'] = 1
kk = 1
for idx in range(0, Configuration.AXS_X, Configuration.CHP_X):
    for idy in range(0, Configuration.AXS_Y, Configuration.CHP_Y):
        star_list['chip'] = np.where((star_list.xcen > idx) & (star_list.xcen < idx + 1320) &
                                     (star_list.ycen > idy) & (star_list.ycen < idy + 5280),
                                     kk, star_list.chip)
        kk = kk + 1

dir = "/Users/yuw816/Data/toros/commissioning/diff/2024-09-30/FIELD_0e.001/"
files =  Utils.get_file_list(dir, '.flux')

nfiles = len(files)

nfiles = len(files)  # the number of flux files to combine
nstars = len(star_list)  # The number of stars to generate light curves for
Utils.log(str(nfiles) + " flux files found for " + Configuration.FIELD + ".", "info")

# make the holders for the light curves
jd = np.zeros(nfiles)
mag = np.zeros((nstars, nfiles))
err = np.zeros((nstars, nfiles))
trd = np.zeros((nstars, nfiles))
sky = np.zeros((nstars, nfiles))
bkg = np.zeros((nstars, nfiles))
err_scl = np.zeros((nstars, nfiles))
zpt = np.zeros((nstars, nfiles))
for idy, file in enumerate(files):

    # read in the data frame with the flux information
    img_flux = pd.read_csv(dir + file, header=0)

    # set the data to the numpy array
    jd[idy] = img_flux.loc[0, 'jd']
    mag[:, idy] = img_flux['mag'].to_numpy()
    err[:, idy] = img_flux['mag_er'].to_numpy()
    sky[:, idy] = img_flux['sky'].to_numpy()
    bkg[:, idy] = img_flux['bkg'].to_numpy()

    if (idy % 100 == 0) & (idy > 0):
        Utils.log("100 flux files read. " + str(nfiles - idy - 1) + ' files remain.', "info")

src_id = star_list.source_id.to_numpy()

for idy, row in star_list[385:].iterrows():

    # get the distance to all stars
    dd = np.sqrt((row.xcen - star_list.xcen.to_numpy()) ** 2 +
                 (row.ycen - star_list.ycen.to_numpy()) ** 2)

    # get the difference in magnitude
    dmag = np.abs(row.master_mag - star_list.master_mag.to_numpy())

    # only get nearby stars of similar magnitude, on the same chip, aren't variables, and aren't in a bad area
    vv = np.argwhere((dmag < 2) & (dd > 0)).reshape(-1)

    # generate the trend for the trend stars
    for idz in range(len(jd)):
        _, trd[idy, idz], _ = sigma_clipped_stats(mag[vv, idz] - np.median(mag[vv], axis=1), sigma=3)

    # get the updates error based on similar stars
    _, _, ss = sigma_clipped_stats(mag[vv] - trd[idy], sigma=3, axis=1)
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
