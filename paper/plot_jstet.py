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
from astropy.stats import sigma_clipped_stats as scs
import numpy as np
from scipy.stats import median_abs_deviation as mad


data_dir = Configuration.LIGHTCURVE_FIELD_DIRECTORY

full_list = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
dys = np.array([2460584, 2460586, 2460599, 2460600, 2460601, 2460614, 2460617,
                2460619, 2460620, 2460621, 2460622, 2460623, 2460626, 2460635])

# daily analysis
stetson_daily = pd.read_csv(data_dir + "stetson_metrics_daily.txt", sep=' ', low_memory=False)

counts = np.zeros((14, len(dys)))
c_rms = np.zeros((3, len(dys)))
per_pass_lsst = np.zeros(len(dys))
per_pass_other = np.zeros(len(dys))

for idx, dy in enumerate(dys):

    # get the daily stetson cutoff
    mad_jstet = mad(stetson_daily[str(dy) + '_j'], nan_policy='omit')
    mdn_jstet = np.nanmedian(stetson_daily[str(dy) + '_j'])
    mad_lstet = mad(stetson_daily[str(dy) + '_l'], nan_policy='omit')
    mdn_lstet = np.nanmedian(stetson_daily[str(dy) + '_l'])

    jstet_cut = mdn_jstet + 3 * mad_jstet
    lstet_cut = mdn_lstet + 3 * mad_lstet

    n_pass_lsst = len(stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                                    (stetson_daily[str(dy) + '_l'] > lstet_cut) &
                                    (stetson_daily['object_type'] == 'LSST')])

    per_pass_lsst[idx] = np.around((n_pass_lsst / len(stetson_daily[stetson_daily['object_type'] == 'LSST'])) * 100, decimals=2)

    mags = stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                         (stetson_daily[str(dy) + '_l'] > lstet_cut) &
                         (stetson_daily['object_type'] == 'LSST')].mag.to_numpy() - 5.4
    rms = stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                         (stetson_daily[str(dy) + '_l'] > lstet_cut) &
                         (stetson_daily['object_type'] == 'LSST')].d90.to_numpy()

    total_rms = stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                              (stetson_daily[str(dy) + '_l'] > lstet_cut)].d90.to_numpy()

    c_rms[0, idx] = np.around(len(rms[rms < 0.01]) / len(total_rms[total_rms < 0.01]) * 100, decimals=2)
    c_rms[1, idx] = np.around(len(rms[(rms >= 0.01) & (rms <= 0.1)]) /
                              len(total_rms[(total_rms >= 0.01) & (total_rms <= 0.1)]) * 100, decimals=2)
    c_rms[2, idx] = np.around(len(rms[rms > 0.1]) / len(total_rms[total_rms > 0.1]) * 100, decimals=2)

    counts[:, idx], bins = np.histogram(mags, range=[8, 22], bins=14)

    n_pass_other = len(stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                                    (stetson_daily[str(dy) + '_l'] > lstet_cut) &
                                    (stetson_daily['object_type'] != 'LSST')])

    per_pass_other[idx]  = np.around((n_pass_other / len(stetson_daily[stetson_daily['object_type'] != 'LSST'])) * 100, decimals=2)

plt.figure(figsize=(9,6))

plt.boxplot([c_rms[0, :], c_rms[1, :], c_rms[2, :]],
            labels=[r'$\Delta_{90} < 0.01$', r'$0.01 < \Delta_{90} < 0.1$', r'$\Delta_{90} > 0.1$'])
plt.ylabel('Percentage of Variable Sources from Rubin-LSST', fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Amplitude", fontsize=15)
plt.savefig("rms_comparison.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()

count_avg = np.mean(counts, axis=1)

plt.figure(figsize=(9,6))
plt.stairs(count_avg, bins, edgecolor='k', linewidth=2)
plt.plot([16, 16], [0,6500], c='r', linewidth=2)
plt.text(12, 6000, "Likely Blends", fontsize=12, color="k")
plt.ylabel('Average Count', fontsize=15)
plt.ylim([0, 6500])
plt.yticks(fontsize=12)
plt.xlabel(r'V$_T$', fontsize=15)
plt.xlim([8,22])
plt.xticks(fontsize=12)
plt.savefig("mag_recovery_lsst_j-l-stetson.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(9,6))

plt.boxplot([per_pass_lsst, per_pass_other], labels=['Rubin-LSST', 'All Others'])
plt.ylabel('% of Stars Passing Stetson J/L Cuts', fontsize=15)
plt.yticks(fontsize=12)
plt.ylim([0, 20])
plt.xticks(fontsize=12)
plt.xlabel("Object Source", fontsize=15)
plt.savefig("stetson_daily_comparison.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()
