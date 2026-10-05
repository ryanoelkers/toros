import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from config import Configuration

tv_zpt = 5.4
xcen_47tuc = 6853
ycen_47tuc = 5375
rad_47tuc = 270

# directories
data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"
lc_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/star_list/fin/"

# read in the varstats file
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
varstats['v'] = varstats['master_mag'] - tv_zpt

plt.figure(figsize=(24,6))

# first 2 days
lc = pd.read_csv(lc_dir + '12/' + Configuration.FIELD + '_4689819822959171200.lc', sep=" ")
plt.subplot(1, 3, 1)

plt.annotate("", xy=(6.6, 11.57), xytext=(6.6, 11.52), arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.annotate("", xy=(4.6, 11.67), xytext=(4.6, 11.62), arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.errorbar(lc[lc.mag > 0].jd - 2460580, lc[lc.mag > 0].mag - tv_zpt, yerr=lc[lc.mag > 0].err, fmt='none', c='k', linewidth=2)
plt.scatter(lc[lc.mag > 0].jd - 2460580, lc[lc.mag > 0].mag - tv_zpt, c='k', s=12)
plt.ylim([12, 11.5])
plt.xlabel('JD - 2460580', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689819822959171200',fontsize=12)

# first day
lc = pd.read_csv(lc_dir + '13/' + Configuration.FIELD + '_4689650738715992064.lc', sep=" ")
plt.subplot(1, 3, 2)
plt.annotate("", xy=(4.7, 15.65), xytext=(4.7, 15.95), arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.errorbar(lc[lc.mag > 0].jd - 2460580, lc[lc.mag > 0].mag - tv_zpt, yerr=lc[lc.mag > 0].err, fmt='none', c='k', linewidth=2)
plt.scatter(lc[lc.mag > 0].jd - 2460580, lc[lc.mag > 0].mag - tv_zpt, c='k', s=12)
plt.ylim([16, 13.5])
plt.xlabel('JD - 2460580', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689650738715992064',fontsize=12)

lc = pd.read_csv(lc_dir + '09/' + Configuration.FIELD + '_4689628958921773184.lc', sep=" ")
plt.subplot(1, 3, 3)
plt.annotate("", xy=(46.5, 14.5), xytext=(46.5, 14.54), arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.errorbar(lc[lc.mag > 0].jd - 2460580, lc[lc.mag > 0].mag - tv_zpt, yerr=lc[lc.mag > 0].err, fmt='none', c='k', linewidth=2)
plt.scatter(lc[lc.mag > 0].jd - 2460580, lc[lc.mag > 0].mag - tv_zpt, c='k', s=12)
plt.ylim([14.55, 14.25])
plt.xlabel('JD - 2460580', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689628958921773184',fontsize=12)

plt.savefig("toros_daily.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(24,6))

lc = pd.read_csv(lc_dir + '13/' + Configuration.FIELD + '_4689651151009833728.lc', sep=" ")
lc['ph'] = (lc.jd - lc.jd.min()) / 5.654137 % 1
plt.subplot(1, 3, 2)
plt.errorbar(lc.ph, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph, lc.mag - tv_zpt, c='k', s=12)
plt.errorbar(lc.ph + 1, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph + 1, lc.mag - tv_zpt, c='k', s=12)
plt.ylim([12.04, 12.14])
plt.gca().invert_yaxis()
plt.xlabel('Phase', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689651151009833728 P=' + str(5.654137) + 'd',fontsize=12)

lc = pd.read_csv(lc_dir + '09/' + Configuration.FIELD + '_4689579240377005184.lc', sep=" ")
lc['ph'] = (lc.jd - lc.jd.min()) / 0.371668 % 1
plt.subplot(1, 3, 1)
plt.errorbar(lc.ph, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph, lc.mag - tv_zpt, c='k', s=12)
plt.errorbar(lc.ph + 1, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph + 1, lc.mag - tv_zpt, c='k', s=12)
plt.gca().invert_yaxis()
plt.xlabel('Phase', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689579240377005184 P=' + str(0.371668) + 'd',fontsize=12)

lc = pd.read_csv(lc_dir + '12/' + Configuration.FIELD + '_4689638059950658816.lc', sep=" ")
lc['ph'] = (lc.jd - lc.jd.min()) / 41.803326 % 1
plt.subplot(1, 3, 3)
plt.errorbar(lc.ph, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph, lc.mag - tv_zpt, c='k', s=12)
plt.errorbar(lc.ph + 1, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph + 1, lc.mag - tv_zpt, c='k', s=12)
plt.gca().invert_yaxis()
plt.xlabel('JD - 2460580', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689638059950658816 P=' + str(41.803326) + 'd', fontsize=12)

plt.savefig("toros_periodics.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()


plt.figure(figsize=(24,6))

lc = pd.read_csv(lc_dir + '09/' + Configuration.FIELD + '_4689582435832666752.lc', sep=" ")
plt.subplot(1, 3, 1)
plt.errorbar(lc.jd - 2460580, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460580, lc.mag - tv_zpt, c='k', s=12)
plt.gca().invert_yaxis()
plt.xlabel('JD - 2460580', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689582435832666752',fontsize=12)

lc = pd.read_csv(lc_dir + '09/' + Configuration.FIELD + '_4689627137844175232.lc', sep=" ")
plt.subplot(1, 3, 2)
plt.errorbar(lc.jd - 2460580, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460580, lc.mag - tv_zpt, c='k', s=12)
plt.gca().invert_yaxis()
plt.xlabel('JD - 2460580', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689627137844175232',fontsize=12)

lc = pd.read_csv(lc_dir + '12/' + Configuration.FIELD + '_4689639507355023488.lc', sep=" ")
plt.subplot(1, 3, 3)
plt.errorbar(lc.jd - 2460580, lc.mag - tv_zpt, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460580, lc.mag - tv_zpt, c='k', s=12)
plt.gca().invert_yaxis()
plt.xlabel('JD - 2460580', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.title('4689639507355023488',fontsize=12)

plt.savefig("toros_variables.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()