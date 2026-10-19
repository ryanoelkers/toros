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

# get the varstats file
data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt",
                       delimiter=' ', low_memory=False)
varstats = varstats[((varstats.var_pass == 1) | (varstats.per_pass == 1) | (varstats.day_pass == 1)) &
                    (varstats.cat_source == 'lsst')].copy().reset_index(drop=True)


# directories
lsst_dir = "/Users/yuw816/Data/toros/commissioning/lsst/"
known = pd.read_csv(lsst_dir + "lsst_data_47tuc_variables.csv",
                    delimiter=',',
                    header=0,
                    low_memory=False,
                    index_col=0)
known['mag'] = -2.5 * np.log10(known.psfFlux) + 31.4
known['err'] = (np.log(10.) / 2.5) * (known['psfFluxErr'] / known['psfFlux'])
known = known[(known.mag > 0) & (known.err > 0) & (known.err < 1)]

known_agg = known.groupby(['band', 'diaObjectId']).agg(mag = ('mag', 'mean'), rms = ('mag', 'std')).reset_index()

lc_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lsst/fin/"

# large amplitude
st_data = known[known.diaObjectId == 579578073926601081].copy().reset_index(drop=True)
st_data['jd'] = st_data.expMidptMJD + 2400000.5
st_data['dys'] = st_data['jd'].astype(int)

lsst_lc = st_data.groupby(['band', 'dys']).agg(mag = ('mag', 'mean'),
                                               err = ('mag', 'std'),
                                               jd = ('jd', 'mean')).reset_index()
plt.figure(figsize=(8, 6))
# plt.subplot(2, 1, 1)

# large amplitude
lc = pd.read_csv(lc_dir + '11/' + Configuration.FIELD + '_579577249292887288.lc', sep=" ")
# plt.subplot(2, 1, 2)
lc = lc[(lc.mag > 0) & (lc.err > 0) & (lc.err < 1)].copy().reset_index(drop=True)

plt.errorbar(lc.jd - 2460580, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460580, lc.mag - tv_zpt, c='k', s=12, label=r'$T_V$')

g_off = lsst_lc[lsst_lc.band == 'g'].mag.mean() - (lc.mag.mean() - tv_zpt)
r_off = lsst_lc[lsst_lc.band == 'r'].mag.mean() - (lc.mag.mean() - tv_zpt)
i_off = lsst_lc[lsst_lc.band == 'i'].mag.mean() - (lc.mag.mean() - tv_zpt)
y_off = lsst_lc[lsst_lc.band == 'y'].mag.mean() - (lc.mag.mean() - tv_zpt)

plt.errorbar(lsst_lc[lsst_lc.band == 'g'].jd - 2460580, lsst_lc[lsst_lc.band == 'g'].mag - g_off,
             yerr=lsst_lc[lsst_lc.band == 'g'].err, fmt='none', c='teal')
plt.scatter(lsst_lc[lsst_lc.band == 'g'].jd - 2460580, lsst_lc[lsst_lc.band == 'g'].mag - g_off, c='teal', label='g')

plt.errorbar(lsst_lc[lsst_lc.band == 'r'].jd - 2460580, lsst_lc[lsst_lc.band == 'r'].mag - r_off,
             yerr=lsst_lc[lsst_lc.band == 'r'].err, fmt='none', c='orange')
plt.scatter(lsst_lc[lsst_lc.band == 'r'].jd - 2460580, lsst_lc[lsst_lc.band == 'r'].mag - r_off, c='orange', label='r')

plt.errorbar(lsst_lc[lsst_lc.band == 'i'].jd - 2460580, lsst_lc[lsst_lc.band == 'i'].mag - i_off,
             yerr=lsst_lc[lsst_lc.band == 'i'].err, fmt='none', c='r')
plt.scatter(lsst_lc[lsst_lc.band == 'i'].jd - 2460580, lsst_lc[lsst_lc.band == 'i'].mag - i_off, c='r', label='i')

plt.errorbar(lsst_lc[lsst_lc.band == 'y'].jd - 2460580, lsst_lc[lsst_lc.band == 'y'].mag - y_off,
             yerr=lsst_lc[lsst_lc.band == 'y'].err, fmt='none', c='maroon')
plt.scatter(lsst_lc[lsst_lc.band == 'y'].jd - 2460580, lsst_lc[lsst_lc.band == 'y'].mag - y_off, c='maroon', label='y')

plt.xlabel('JD - 2460580', fontsize=15)
plt.xticks(fontsize=12)
plt.xlim([0, 75])

plt.ylabel('Magnitude + Offset', fontsize=15)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()

plt.title('Large Amplitude Variability',fontsize=15)
plt.legend(fontsize=12, loc='lower left')

plt.savefig("lsst_large_amplitude.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()

st_data = known[known.diaObjectId == 579576699537068708].copy().reset_index(drop=True)
lc = pd.read_csv(lc_dir + '10/' + Configuration.FIELD + '_579576699537068708.lc', sep=" ")
lc_min = lc.jd.min() - 2400000.5
lc = lc[(lc.mag > 0) & (lc.err > 0) & (lc.err < 1)].copy().reset_index(drop=True)

ph = ((lc.jd - lc.jd.min()) / 1.103044) % 1
st_data['ph'] = (st_data.expMidptMJD - lc_min) / 1.103044 % 1

plt.figure(figsize=(8, 6))

plt.errorbar(ph, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(ph, lc.mag - tv_zpt, c='k', s=12, label=r'$T_V$')
plt.errorbar(ph + 1, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(ph + 1, lc.mag - tv_zpt, c='k', s=12)

g_off = st_data[st_data.band == 'g'].mag.mean() - (lc.mag.mean() - tv_zpt)
r_off = st_data[st_data.band == 'r'].mag.mean() - (lc.mag.mean() - tv_zpt)

plt.errorbar(st_data[st_data.band == 'g'].ph, st_data[st_data.band == 'g'].mag - g_off,
             yerr=st_data[st_data.band == 'g'].err, fmt='none', c='teal')
plt.scatter(st_data[st_data.band == 'g'].ph, st_data[st_data.band == 'g'].mag - g_off, c='teal', label='g')

plt.errorbar(st_data[st_data.band == 'g'].ph + 1, st_data[st_data.band == 'g'].mag - g_off,
             yerr=st_data[st_data.band == 'g'].err, fmt='none', c='teal')
plt.scatter(st_data[st_data.band == 'g'].ph + 1, st_data[st_data.band == 'g'].mag - g_off, c='teal')

plt.errorbar(st_data[st_data.band == 'r'].ph, st_data[st_data.band == 'r'].mag - r_off,
             yerr=st_data[st_data.band == 'r'].err, fmt='none', c='orange')
plt.scatter(st_data[st_data.band == 'r'].ph, st_data[st_data.band == 'r'].mag - r_off, c='orange', label='r')

plt.errorbar(st_data[st_data.band == 'r'].ph + 1, st_data[st_data.band == 'r'].mag - r_off,
             yerr=st_data[st_data.band == 'r'].err, fmt='none', c='orange')
plt.scatter(st_data[st_data.band == 'r'].ph + 1, st_data[st_data.band == 'r'].mag - r_off, c='orange')

plt.xlabel('Phase', fontsize=15)
plt.xticks(fontsize=12)
plt.xlim([0, 2])

plt.title('Periodic Variability P=1.103044',fontsize=15)
plt.legend(fontsize=12, loc='lower left')

plt.ylabel('Magnitude + Offset', fontsize=15)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()

plt.savefig("lsst_periodic.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()

## Sudden Onset Variabilty ##

lc = pd.read_csv(lc_dir + '14/' + Configuration.FIELD + '_579578073926601081.lc', sep=" ")
plt.figure(figsize=(8, 6))
lc = lc[(lc.mag > 0) & (lc.err > 0) & (lc.err < 1)].copy().reset_index(drop=True)
plt.annotate("", xy=(2460626.574483 - 2460580, 16.9),
             xytext=(2460626.574483 - 2460580, 16.5),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))

plt.errorbar(lc.jd - 2460580, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460580, lc.mag - tv_zpt, c='k', s=12, label=r'$T_V$')

st_data = known[known.diaObjectId == 579577249292887288].copy().reset_index(drop=True)

st_data['jd'] = st_data.expMidptMJD + 2400000.5

g_off = st_data[st_data.band == 'g'].mag.min() - (lc.mag.mean() - tv_zpt)
r_off = st_data[st_data.band == 'r'].mag.min() - (lc.mag.mean() - tv_zpt)
i_off = st_data[st_data.band == 'i'].mag.min() - (lc.mag.mean() - tv_zpt)
y_off = st_data[st_data.band == 'y'].mag.min() - (lc.mag.mean() - tv_zpt)

plt.errorbar(st_data[st_data.band == 'g'].jd - 2460580, st_data[st_data.band == 'g'].mag - g_off,
             yerr=st_data[st_data.band == 'g'].err, fmt='none', c='teal')
plt.scatter(st_data[st_data.band == 'g'].jd - 2460580, st_data[st_data.band == 'g'].mag - g_off, c='teal', label='g')

plt.errorbar(st_data[st_data.band == 'r'].jd - 2460580, st_data[st_data.band == 'r'].mag - r_off,
             yerr=st_data[st_data.band == 'r'].err, fmt='none', c='orange')
plt.scatter(st_data[st_data.band == 'r'].jd - 2460580, st_data[st_data.band == 'r'].mag - r_off, c='orange', label='r')

plt.errorbar(st_data[st_data.band == 'i'].jd - 2460580, st_data[st_data.band == 'i'].mag - i_off,
             yerr=st_data[st_data.band == 'i'].err, fmt='none', c='r')
plt.scatter(st_data[st_data.band == 'i'].jd - 2460580, st_data[st_data.band == 'i'].mag - i_off, c='r', label='i')

plt.errorbar(st_data[st_data.band == 'y'].jd - 2460580, st_data[st_data.band == 'y'].mag - y_off,
             yerr=st_data[st_data.band == 'y'].err, fmt='none', c='maroon')
plt.scatter(st_data[st_data.band == 'y'].jd - 2460580, st_data[st_data.band == 'y'].mag - y_off, c='maroon', label='y')

plt.xlabel('JD - 2460580', fontsize=15)
plt.xticks(fontsize=12)
plt.xlim([0, 75])
plt.ylabel('Magnitude + Offset', fontsize=15)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()

plt.legend(fontsize=12, loc='lower left')
plt.title('Sudden Onset Variability',fontsize=15)

plt.savefig("lsst_sudden_onset.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()


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
