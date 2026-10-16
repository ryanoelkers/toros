import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from config import Configuration
import numpy as np
tv_zpt = 5.4
xcen_47tuc = 6853
ycen_47tuc = 5375
rad_47tuc = 270


# directories
lsst_dir = "/Users/yuw816/Data/toros/commissioning/lsst/"
known = pd.read_csv(lsst_dir + "lsst_data_47tuc_variables.csv",
                    delimiter=',',
                    header=0,
                    low_memory=False,
                    index_col=0)
known['mag'] = -2.5 * np.log10(known.psfFlux) + 31.4
known['err'] = (np.log(10.) / 2.5) * (known['psfFluxErr'] / known['psfFlux'])
known = known[(known.mag > 0) & (known.err >0) & (known.err < 1)]
lc_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lsst/fin/"

st_data = known[known.diaObjectId == 579578073926601081].copy().reset_index(drop=True)
# st_data['ph'] = (st_data.expMidptMJD - st_data.expMidptMJD.min()) / 1.103044 % 1
plt.errorbar(st_data.expMidptMJD, st_data.mag, yerr=st_data.err, fmt='none')
plt.scatter(st_data.expMidptMJD, st_data.mag)
plt.gca().invert_yaxis()
plt.show()

st_data = known[known.diaObjectId == 579576699537068708].copy().reset_index(drop=True)
st_data['ph'] = (st_data.expMidptMJD - st_data.expMidptMJD.min()) / 1.103044 % 1
plt.errorbar(st_data.ph, st_data.mag, yerr=st_data.err, fmt='none')
plt.scatter(st_data.ph, st_data.mag)
plt.gca().invert_yaxis()
plt.show()

st_data = known[known.diaObjectId == 579577249292887288].copy().reset_index(drop=True)
plt.errorbar(st_data.expMidptMJD, st_data.mag, yerr=st_data.err, fmt='none')
plt.scatter(st_data.expMidptMJD, st_data.mag)
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(24,6))

# large amplitude
lc = pd.read_csv(lc_dir + '11/' + Configuration.FIELD + '_579577249292887288.lc', sep=" ")
plt.subplot(1, 3, 1)
lc = lc[(lc.mag > 0) & (lc.err > 0) & (lc.err < 1)].copy().reset_index(drop=True)

plt.errorbar(lc.jd - 2460580, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460580, lc.mag - tv_zpt, c='k', s=12)
plt.gca().invert_yaxis()
plt.xlabel('JD - 2460580', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('Large Amplitude Variability',fontsize=12)

# periodic
lc = pd.read_csv(lc_dir + '10/' + Configuration.FIELD + '_579576699537068708.lc', sep=" ")
plt.subplot(1, 3, 2)
lc = lc[(lc.mag > 0) & (lc.err > 0) & (lc.err < 1)].copy().reset_index(drop=True)
ph = ((lc.jd - lc.jd.min()) / 1.103044) % 1

plt.errorbar(ph, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(ph, lc.mag - tv_zpt, c='k', s=12)
plt.errorbar(ph + 1, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(ph + 1, lc.mag - tv_zpt, c='k', s=12)
plt.xlabel('Phase', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()
plt.title('Periodic Variability P=1.103044',fontsize=12)

# daily
lc = pd.read_csv(lc_dir + '14/' + Configuration.FIELD + '_579578073926601081.lc', sep=" ")
plt.subplot(1, 3, 3)
lc = lc[(lc.mag > 0) & (lc.err > 0) & (lc.err < 1)].copy().reset_index(drop=True)

plt.annotate("", xy=(2460626.574483 - 2460580, 16.9),
             xytext=(2460626.574483 - 2460580, 16.5),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))

plt.errorbar(lc.jd - 2460580, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460580, lc.mag - tv_zpt, c='k', s=12)
plt.xlabel('JD - 2460580', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()

plt.title('Sudden Onset Variability',fontsize=12)

plt.savefig("lsst_vars.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()
