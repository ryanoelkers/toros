import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import Configuration


data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt",
                       delimiter=' ', low_memory=False)
varstats = varstats[((varstats.var_pass == 1) | (varstats.per_pass == 1) | (varstats.day_pass == 1)) & 
                    (varstats.cat_source == 'toros')].copy().reset_index(drop=True)

df = varstats[['star_id', 'source_id', 'chip', 'cat_source', 'prd', 'v', 'rms', 'd90', 'jstet', 'lstet',
               'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'var_pass','per_pass', 'day_pass', 'parallax',
               'parallax_error', 'G47T', 'N121', 'pvar', 'xray', 'lsst']].copy()
df = df[(df.parallax_error / df.parallax < 0.5)]

df['abs_g'] = df.phot_g_mean_mag - 5 * np.log10(1000. / df.parallax) + 5
df['bmp'] = df.phot_bp_mean_mag - df.phot_rp_mean_mag

plt.figure(figsize=(9,6))
plt.scatter(df[df.G47T == 1].bmp, df[df.G47T == 1].abs_g, c='k', label='47 Tuc', s=6)
plt.scatter(df[df.lsst == 1].bmp, df[df.lsst == 1].abs_g, c='grey', label='Rubin-LSST', s=6)

plt.scatter(df[df.N121 == 1].bmp, df[df.N121 == 1].abs_g, c='darkorange', label='NGC 121', s=20)
plt.scatter(df[df.pvar == 1].bmp, df[df.pvar == 1].abs_g, c='blue', label='AAVSO VSX', s=20)
plt.scatter(df[df.xray == 1].bmp, df[df.xray == 1].abs_g, c='red', label='X-Ray eROSITA', s=20)
plt.scatter(df[(df.xray == 0) & (df.G47T == 0) & (df.pvar == 0) & (df.N121 == 0) & (df.lsst == 0)].bmp,
            df[(df.xray == 0) & (df.G47T == 0) & (df.pvar == 0) & (df.N121 == 0) & (df.lsst == 0)].abs_g,
            c='teal', label='All Others', s=20)

plt.legend(fontsize=12, loc='upper left')
plt.xlim([-0.5, 3])
plt.xlabel(r'$B_p-R_p$', fontsize=15)
plt.xticks(fontsize=12)
plt.ylim([15, -5])
plt.ylabel('G', fontsize=15)
plt.yticks(fontsize=12)
plt.savefig("toros_hr_diagram.png", dpi=200, bbox_inches='tight')
plt.show()

