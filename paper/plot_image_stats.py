import pandas as pd
import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from config import Configuration
import astropy.units as u
from astropy.coordinates import SkyCoord, get_body, get_sun
from astropy.stats import sigma_clipped_stats as scs
from astropy.stats import sigma_clip as sc
import statistics
from astropy.time import Time
import numpy as np

image_stats = pd.read_csv("/Users/yuw816/Data/toros/commissioning/master/FIELD_0e.001/FIELD_0e.001_image_stats.txt",
                        sep=" ")
image_stats['moon'] = 0
# get the lunar phase
for idx, row in image_stats.iterrows():
    t = Time(row.jd, format='jd')  # or Time("2026-10-14 03:00:00")

    sun = get_sun(t)
    moon = get_body("moon", t)

    # Elongation of the Moon from the Sun
    elong = sun.separation(moon)

    # Phase angle (Sun-Moon-Earth), illuminated fraction
    phase_angle = np.arctan2(sun.distance * np.sin(elong),
                             moon.distance - sun.distance * np.cos(elong))
    image_stats.loc[idx, 'moon'] = (1 + np.cos(phase_angle.value)) / 2

_, img_mdn_sky, img_std_sky = scs(image_stats.sky.to_numpy(), sigma=2)

pass_images = image_stats[(image_stats['nstars'] > Configuration.NSKY_STARS) &
                          (image_stats['date'] != Configuration.BAD_DATES)].copy().reset_index(drop=True)

# get to magnitude to sq. arcsec (flx / pix * Gain (e/ADU) * pix / arcsec)
flux_adu_s_pix = (pass_images.sky.to_numpy() * Configuration.GAIN) / Configuration.EXP_TIME
flux_adu_s_arcsec2 = flux_adu_s_pix / (Configuration.PIXEL_SIZE ** 2)
pass_images['mag_arcsec2'] = 25 - 2.5 * np.log10(flux_adu_s_arcsec2) - 5.4

Utils.log("The minimum darkness was: " + str(np.around(pass_images.mag_arcsec2.max(), decimals=1)), "info")
Utils.log("The average dark time was: "+ str(np.around(pass_images[pass_images.moon <=0.5].mag_arcsec2.mean(), decimals=1)), "info")

plt.figure(figsize=(18,6))
plt.subplot(1, 3, 1)
plt.hist([pass_images[pass_images.moon <= 0.5].mag_arcsec2, pass_images[pass_images.moon > 0.5].mag_arcsec2],
         color=['k', 'r'], histtype='step', linewidth=2,
         label=[r'Lunar Illumination $\leq 0.5$', 'Lunar Illumination $> 0.5$'],
         range=[17.5, 21.5], bins=10)

plt.xlabel('Sky Background [mag / sq. arcsec]', fontsize=15)
plt.xticks(fontsize=12)
plt.ylabel('Count', fontsize=15)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper center')


plt.subplot(1, 3, 2)
plt.hist([pass_images[pass_images.moon <= 0.5].nstars/10000, pass_images[pass_images.moon > 0.5].nstars/10000],
         color=['k', 'r'], histtype='step', linewidth=2,
         label=[r'Lunar Illumination $\leq 0.5$', 'Lunar Illumination $> 0.5$'],
         range=[2, 10], bins=8)
plt.xlabel(r'Number of Stars per Observation [$10^4$]', fontsize=15)
plt.xticks(fontsize=12)
plt.ylabel('Count', fontsize=15)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper center')

plt.subplot(1, 3, 3)
plt.hist([pass_images[pass_images.moon <= 0.5].airmass, pass_images[pass_images.moon > 0.5].airmass],
         color=['k', 'r'], histtype='step', linewidth=2,
         label=[r'Lunar Illumination $\leq 0.5$', 'Lunar Illumination $> 0.5$'],
         range=[1.3, 1.6], bins=6)
plt.xlabel('Airmass of Observations', fontsize=15)
plt.xticks(fontsize=12)
plt.ylabel('Count', fontsize=15)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper center')

plt.savefig("toros_observations_summary.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()
print('hold')
