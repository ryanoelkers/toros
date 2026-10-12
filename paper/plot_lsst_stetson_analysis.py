import pandas as pd
import matplotlib
import logging
import matplotlib.colors as colors
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from config import Configuration
from astropy.stats import sigma_clipped_stats as scs
import numpy as np
from scipy.stats import median_abs_deviation as mad
from libraries.varstats import Varstats

lsst_stet = pd.read_csv("jstet_daily_r.txt", sep=',')

plt.figure(figsize=(9,6))
plt.scatter(lsst_stet.v, lsst_stet.r,
            c=lsst_stet.d90, marker='.', label='TOROS Objects with Daily Variability', alpha=0.5, norm='log')
plt.xlabel(r'$T_V$', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('r', fontsize=20)
plt.yticks(fontsize=15)
plt.colorbar().set_label(r'$\Delta_{90}$ Logarithmic Scale', fontsize=15)
plt.savefig("toros_daily_stetson.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()