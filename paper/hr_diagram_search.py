import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import Configuration


lc_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/star_list/fin/"
data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt",
                       delimiter=' ', low_memory=False)
varstats = varstats[((varstats.var_pass == 1) | (varstats.per_pass == 1) | (varstats.day_pass == 1)) &
                    (varstats.cat_source == 'toros')].copy().reset_index(drop=True)

df = varstats[['star_id', 'source_id', 'chip', 'cat_source', 'prd', 'v', 'rms', 'd90', 'jstet', 'lstet',
               'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'var_pass','per_pass', 'day_pass', 'parallax',
               'parallax_error', 'G47T', 'N121', 'pvar', 'xray', 'lsst', 'var_type', 'var_id', 'ra', 'dec']].copy()
df = df[(df.parallax_error / df.parallax < 0.5)]

df['abs_g'] = df.phot_g_mean_mag.to_numpy() - 5 * np.log10(1000. / df.parallax.to_numpy()) + 5
df['bmp'] = df.phot_bp_mean_mag.to_numpy() - df.phot_rp_mean_mag.to_numpy()

chk_stars = df[(df.lsst == 1)].copy().reset_index(drop=True)

for idx, row in chk_stars.iterrows():

    if ~np.isnan(row.abs_g) & (row.per_pass == 1) & ~np.isnan(row.bmp):
        plt.figure(figsize=(21,6))
        plt.subplot(1, 3, 1)

        plt.scatter(df.bmp, df.abs_g, c='k', s=20)
        plt.scatter(row.bmp, row.abs_g, c='r')
        plt.xlim([-0.5, 3])
        plt.xlabel(r'$B_p-R_p$', fontsize=15)
        plt.xticks(fontsize=12)
        plt.ylim([15, -5])
        plt.ylabel('G', fontsize=15)
        plt.yticks(fontsize=12)

        if row.chip >= 10:
            lc = pd.read_csv(lc_dir + str(row.chip) + '/' + Configuration.FIELD + '_' + str(row.source_id) + '.lc', sep=" ")
        else:
            lc = pd.read_csv(lc_dir + '0' + str(row.chip) + '/' + Configuration.FIELD + '_' + str(row.source_id) + '.lc', sep=" ")

        lc = lc[(lc.mag > 0) & (lc.err > 0) & (lc.err < 1)].copy().reset_index(drop=True)

        plt.subplot(1, 3, 2)
        ph = (lc.jd.to_numpy() - lc.jd.min()) / row.prd % 1
        plt.errorbar(ph, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
        plt.scatter(ph, lc.mag - 5.4, c='k', s=12)
        plt.gca().invert_yaxis()
        plt.xlabel('Phase', fontsize=12)
        plt.xticks(fontsize=12)
        plt.ylabel(r'$T_V$', fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(str(row.source_id) + " P=" + str(row.prd)+ ' ' + str(row.var_type) + ' ' + str(row.var_id))

        plt.subplot(1, 3, 3)
        ph = (lc.jd.to_numpy() - lc.jd.min()) / (2*row.prd) % 1
        plt.errorbar(ph, lc.mag - 5.4, yerr=lc.err, fmt='none', c='k', linewidth=2)
        plt.scatter(ph, lc.mag - 5.4, c='k', s=12)
        plt.gca().invert_yaxis()
        plt.xlabel('JD - 2460580', fontsize=12)
        plt.xticks(fontsize=12)

        plt.ylabel(r'$T_V$', fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(str(row.source_id) + ' ' + str(row.ra) + ' '+ str(row.dec) + ' ' + str(row.var_type))
        plt.show()
        plt.close()