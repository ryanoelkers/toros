import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import Configuration
from matplotlib.ticker import FormatStrFormatter

lc_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/star_list/fin/"
data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt",
                       delimiter=' ', low_memory=False)
varstats = varstats[((varstats.var_pass == 1) | (varstats.per_pass == 1) | (varstats.day_pass == 1)) &
                    (varstats.cat_source == 'toros')].copy().reset_index(drop=True)

df = varstats[['star_id', 'source_id', 'chip', 'cat_source', 'prd', 'v', 'rms', 'd90', 'jstet', 'lstet',
               'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'var_pass','per_pass', 'day_pass', 'parallax',
               'parallax_error', 'G47T', 'N121', 'pvar', 'xray', 'lsst', 'var_type', 'var_id']].copy()
df = df[(df.parallax_error / df.parallax < 0.5)]

df['abs_g'] = df.phot_g_mean_mag.to_numpy() - 5 * np.log10(1000. / df.parallax.to_numpy()) + 5
df['bmp'] = df.phot_bp_mean_mag.to_numpy() - df.phot_rp_mean_mag.to_numpy()

## BINARY CANDIDATE ##
star = df[df.source_id == '4689622396209979136'].copy().reset_index(drop=True)

plt.figure(figsize=(21,6))
plt.subplot(1, 3, 1)

plt.scatter(df.bmp, df.abs_g, c='k', s=20)
plt.scatter(star.bmp, star.abs_g, c='r')
plt.xlim([-0.5, 3])
plt.xlabel(r'$B_p-R_p$', fontsize=15)
plt.xticks(fontsize=12)
plt.ylim([16, -5])
plt.ylabel('G', fontsize=15)
plt.yticks(fontsize=12)

if star.chip[0] >= 10:
    lc = pd.read_csv(lc_dir + str(star.chip[0]) + '/' + Configuration.FIELD + '_' + str(star.source_id[0]) + '.lc', sep=" ")
else:
    lc = pd.read_csv(lc_dir + '0' + str(star.chip[0]) + '/' + Configuration.FIELD + '_' + str(star.source_id[0]) + '.lc', sep=" ")

lc = lc[(lc.mag > 0) & (lc.err > 0) & (lc.err < 1)].copy().reset_index(drop=True)
lc['ph'] = (lc.jd.to_numpy() - lc.jd.min()) / star.prd[0] % 1
plt.subplot(1, 3, 2)

plt.errorbar(lc.ph, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph, lc.mag - 5.4, c='k', s=12)

plt.errorbar(lc.ph + 1, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph + 1, lc.mag - 5.4, c='k', s=12)

plt.xlabel('Phase', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.ylim([14.37, 14.24])
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.title(str(star.source_id[0]) + " Rubin-LSST Variable P="+str(np.around(star.prd[0], decimals=4)) + 'd')

plt.subplot(1, 3, 3)
p_old = star.prd[0] * 2
lc['ph_old'] = (lc.jd.to_numpy() - lc.jd.min()) / p_old % 1

plt.errorbar(lc.ph_old, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph_old, lc.mag - 5.4, c='k', s=12)

plt.errorbar(lc.ph_old + 1, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph_old + 1, lc.mag - 5.4, c='k', s=12)

plt.annotate("", xy=(0.3, 14.26),
             xytext=(0.3, 14.25),
             arrowprops=dict(arrowstyle="->", color="b", lw=2))
plt.annotate("", xy=(1.3, 14.26),
             xytext=(1.3, 14.25),
             arrowprops=dict(arrowstyle="->", color="b", lw=2))

plt.annotate("", xy=(0.8, 14.26),
             xytext=(0.8, 14.25),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.annotate("", xy=(1.8, 14.26),
             xytext=(1.8, 14.25),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))

plt.xlabel('Phase', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.ylim([14.37, 14.24])
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.title(str("Phased on P="+str(np.around(p_old, decimals=4))+'d, Semi-Detached Binary Candidate'))
plt.savefig("new_binary.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()


## MISMATCH PERIOD ##
star = df[df.source_id == '4689844360109159552'].copy().reset_index(drop=True)

plt.figure(figsize=(21,6))
plt.subplot(1, 3, 1)

plt.scatter(df.bmp, df.abs_g, c='k', s=20)
plt.scatter(star.bmp, star.abs_g, c='r')
plt.xlim([-0.5, 3])
plt.xlabel(r'$B_p-R_p$', fontsize=15)
plt.xticks(fontsize=12)
plt.ylim([16, -5])
plt.ylabel('G', fontsize=15)
plt.yticks(fontsize=12)

if star.chip[0] >= 10:
    lc = pd.read_csv(lc_dir + str(star.chip[0]) + '/' + Configuration.FIELD + '_' + str(star.source_id[0]) + '.lc', sep=" ")
else:
    lc = pd.read_csv(lc_dir + '0' + str(star.chip[0]) + '/' + Configuration.FIELD + '_' + str(star.source_id[0]) + '.lc', sep=" ")

lc = lc[(lc.mag > 0) & (lc.err > 0) & (lc.err < 1)].copy().reset_index(drop=True)
lc['ph'] = (lc.jd.to_numpy() - lc.jd.min()) / star.prd[0] % 1
plt.subplot(1, 3, 2)

plt.errorbar(lc.ph, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph, lc.mag - 5.4, c='k', s=12)

plt.errorbar(lc.ph + 1, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph + 1, lc.mag - 5.4, c='k', s=12)

plt.xlabel('Phase', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.ylim([12.53, 12.41])
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.title(str(star.source_id[0]) + " Xray Source P="+str(np.around(star.prd[0], decimals=4)) + 'd')

plt.subplot(1, 3, 3)
p_old = 3.9533
lc['ph_old'] = (lc.jd.to_numpy() - lc.jd.min()) / p_old % 1

plt.errorbar(lc.ph_old, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph_old, lc.mag - 5.4, c='k', s=12)

plt.errorbar(lc.ph_old + 1, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.ph_old + 1, lc.mag - 5.4, c='k', s=12)

plt.xlabel('Phase', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.ylim([12.53, 12.41])
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.title(str("Phased on P="+str(p_old)+'d'))

plt.savefig("mismatch_period_star.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()

## LONG TERM VARIABLE STAR ##
star = df[df.source_id == '4689637789376139264'].copy().reset_index(drop=True)
plt.figure(figsize=(21,6))
plt.subplot(1, 3, 1)

plt.scatter(df.bmp, df.abs_g, c='k', s=20)
plt.scatter(star.bmp, star.abs_g, c='r')
plt.xlim([-0.5, 3])
plt.xlabel(r'$B_p-R_p$', fontsize=15)
plt.xticks(fontsize=12)
plt.ylim([16, -5])
plt.ylabel('G', fontsize=15)
plt.yticks(fontsize=12)

if star.chip[0] >= 10:
    lc = pd.read_csv(lc_dir + str(star.chip[0]) + '/' + Configuration.FIELD + '_' + str(star.source_id[0]) + '.lc', sep=" ")
else:
    lc = pd.read_csv(lc_dir + '0' + str(star.chip[0]) + '/' + Configuration.FIELD + '_' + str(star.source_id[0]) + '.lc', sep=" ")

lc = lc[(lc.mag > 0) & (lc.err > 0) & (lc.err < 1)].copy().reset_index(drop=True)

plt.subplot(1, 3, 2)

plt.annotate("", xy=(4.5, 11.21),
             xytext=(4.5, 11.20),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.annotate("", xy=(19.5, 11.21),
             xytext=(19.5, 11.20),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.annotate("", xy=(20.5, 11.21),
             xytext=(20.5, 11.20),
             arrowprops=dict(arrowstyle="->", color="b", lw=2))
plt.annotate("", xy=(21.5, 11.21),
             xytext=(21.5, 11.20),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.annotate("", xy=(34.5, 11.21),
             xytext=(34.5, 11.20),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.annotate("", xy=(37.5, 11.21),
             xytext=(37.5, 11.20),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.annotate("", xy=(43.5, 11.21),
             xytext=(43.5, 11.20),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.annotate("", xy=(46.5, 11.21),
             xytext=(46.5, 11.20),
             arrowprops=dict(arrowstyle="->", color="red", lw=2))
plt.errorbar(lc.jd - 2460580, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460580, lc.mag - 5.4, c='k', s=12)

plt.xlabel('JD - 2460580', fontsize=12)
plt.xticks(fontsize=12)

plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.title(str(star.source_id[0]) + " Long Term Variable")
plt.ylim([11.38, 11.2])
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.subplot(1, 3, 3)
plt.errorbar(lc.jd - 2460600, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
plt.scatter(lc.jd - 2460600, lc.mag - 5.4, c='k', s=12)

plt.xlabel('JD - 2460600', fontsize=12)
plt.xticks(fontsize=12)
plt.xlim([0.52, 0.66])
plt.ylabel(r'$T_V$', fontsize=12)
plt.yticks(fontsize=12)
plt.ylim([11.3, 11.24])
plt.title('Zoom-in on JD=2460600 Variability')

plt.savefig("flaring_l_star.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()