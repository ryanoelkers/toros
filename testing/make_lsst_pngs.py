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
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import warnings
warnings.simplefilter('error', RuntimeWarning)
import os
from astropy.io import fits
from astropy.visualization import LogStretch, ZScaleInterval, ImageNormalize, SqrtStretch
from astropy.time import Time
# star list from LSST
lsst_list = pd.read_csv(Configuration.ANALYSIS_DIRECTORY + "lsst_sources_positions.csv",
                        delimiter=',',
                        header=0,
                        low_memory=False)

# get the directory for the files
data_dir = "/Volumes/OUMUAMUA/toros/commissioning/diff/"

# get the differenced file list
dates = [f for f in os.listdir(data_dir) if not f.startswith('._')]
dates = np.sort(dates)
ndates = len(dates)

files = []
for date in dates:
    files = files + [data_dir + date + '/FIELD_0e.001/' + f for f in os.listdir(data_dir + date + '/FIELD_0e.001/') if not f.startswith('._')]

# +/- size from the source
x_size = 50
y_size = 50

for idx, file in enumerate(files):
    # read in the image and the header
    img, header = fits.getdata(file, header=True)
    time = Time(header['DATE'], format='isot', scale='utc')
    jd = time.jd

    tme_nme = str(np.around(jd, decimals=6)).split('.')
    for idy, row in lsst_list.iterrows():
        # make the output directory
        if os.path.exists(Configuration.ANALYSIS_DIRECTORY + 'lsst_sources_pngs/' + row.star_id) is False:
            os.mkdir(Configuration.ANALYSIS_DIRECTORY + 'lsst_sources_pngs/' +  row.star_id)
            Utils.log(Configuration.ANALYSIS_DIRECTORY + 'lsst_sources_pngs/' +  row.star_id + ' created.', 'info')

        # get the x/y from the ra/dec of the image
        w = WCS(header)
        x, y = w.all_world2pix(row.ra, row.dec, 0)

        cut_img = img[int(y)-y_size:int(y)+y_size, int(x)-x_size:int(x)+x_size]
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(img)
        norm = ImageNormalize(vmin=-27, vmax=28, stretch=SqrtStretch())

        # 3. Create the plot without axes or borders for a clean PNG
        plt.figure(figsize=(10, 10))
        plt.imshow(cut_img, cmap='grey', norm=norm, origin='lower')

        center_y, center_x = cut_img.shape[0] / 2, cut_img.shape[1] / 2
        circle = plt.Circle((center_x, center_y), radius=5, fill=False,
                            edgecolor='red', linewidth=1.5)
        plt.gca().add_patch(circle)

        # 4. Save directly as a PNG
        plt.savefig(Configuration.ANALYSIS_DIRECTORY +
                    'lsst_sources_pngs/' +  row.star_id + '/' +
                    row.star_id + '_' + tme_nme[0] + '.' + tme_nme[1] + '.png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
