import numpy as np
import matplotlib.pyplot as plt
from libraries.utils import Utils
from config import Configuration

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.modeling import models, fitting


def measure_fwhm(fits_file,
                 threshold_sigma=5,
                 max_stars=50,
                 box_size=15):

    data = fits.getdata(fits_file)
    data[3600:8200, 4300:9300] = 0
    # Estimate background
    mean, median, std = sigma_clipped_stats(data)

    # Detect stars
    daofind = DAOStarFinder(fwhm=3.0, threshold=threshold_sigma * std)
    sources = daofind(data - median)

    if sources is None:
        print("No stars detected.")
        return None

    # Sort by brightness (descending)
    sources.sort('flux')
    sources = sources[::-1]

    fwhm_list = []

    for star in sources[:max_stars]:
        x = star['xcentroid']
        y = star['ycentroid']

        x = int(round(x))
        y = int(round(y))

        half = box_size // 2
        cutout = data[y-half:y+half+1,
                      x-half:x+half+1] - median

        if cutout.shape[0] != box_size or cutout.shape[1] != box_size:
            continue

        yy, xx = np.mgrid[:box_size, :box_size]

        # 2D Gaussian model
        g_init = models.Gaussian2D(
            amplitude=np.max(cutout),
            x_mean=half,
            y_mean=half,
            x_stddev=2,
            y_stddev=2
        )

        fitter = fitting.LevMarLSQFitter()
        g_fit = fitter(g_init, xx, yy, cutout)

        sigma_x = g_fit.x_stddev.value
        sigma_y = g_fit.y_stddev.value

        # Convert sigma → FWHM
        fwhm_x = 2.355 * sigma_x
        fwhm_y = 2.355 * sigma_y

        fwhm_mean = np.mean([fwhm_x, fwhm_y])
        fwhm_list.append(fwhm_mean)

    if len(fwhm_list) == 0:
        return None

    return np.median(fwhm_list)


# Example usage
fwhm = measure_fwhm("/Users/yuw816/Data/toros/commissioning/master/FIELD_0e.001/FIELD_0e.001_master.fits")
print("Image FWHM (pixels):", fwhm)

dir = "/Users/yuw816/Data/toros/commissioning/master/FIELD_0e.001/FIELD_0e.001_master.fits"
