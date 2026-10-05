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
data_dir = "/Volumes/OUMUAMUA/toros/commissioning/varstats/FIELD_0e.001/"

# data_dir = Configuration.LIGHTCURVE_FIELD_DIRECTORY
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)

lsststats = varstats[varstats.source_id == varstats.lsst_id].copy().reset_index(drop=True)
lsststats['source_id'] = lsststats['source_id'].astype(int)

# get the known LSST variables
known = pd.read_csv(Configuration.DATA_DIRECTORY + "/lsst/lsst_data_47tuc_variables.csv",
                    delimiter=',',
                    header=0,
                    low_memory=False,
                    index_col=0)
known_g = known[known.band == 'g'].groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean', 'psfFlux': 'mean'}).reset_index()
known_r = known[known.band == 'r'].groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean', 'psfFlux': 'mean'}).reset_index()
known_i = known[known.band == 'i'].groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean', 'psfFlux': 'mean'}).reset_index()
known_y = known[known.band == 'y'].groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean', 'psfFlux': 'mean'}).reset_index()

known_g['g'] = -2.5 * np.log10(known_g.psfFlux) + 31.4
known_r['r'] = -2.5 * np.log10(known_r.psfFlux) + 31.4
known_i['i'] = -2.5 * np.log10(known_i.psfFlux) + 31.4
known_y['y'] = -2.5 * np.log10(known_y.psfFlux) + 31.4

# get the full cutoffs
jstet_results = Varstats.stetson_j_peak_and_cutoff(varstats[(varstats.source_id != varstats.lsst_id)].jstet,
                                                   sigma_method="mirror_std", n_sigma=3.0)
jstet_cut = jstet_results['cutoff']

lstet_results = Varstats.stetson_j_peak_and_cutoff(varstats[(varstats.source_id != varstats.lsst_id)].lstet,
                                                   sigma_method="mirror_std", n_sigma=3.0)
lstet_cut = lstet_results['cutoff']

n_var_pass_lsst_total = len(lsststats[(lsststats.jstet > jstet_cut) & (lsststats.lstet > lstet_cut)])

Utils.log("The number of LSST stars that pass the cuts with the full lc are: " + str(n_var_pass_lsst_total),
          "info")
Utils.log("As a percentage that is: " + str(np.around(n_var_pass_lsst_total / len(lsststats) * 100, decimals=2)) + "%",
          "info")

dys = np.array([2460584, 2460586, 2460599, 2460600, 2460601, 2460614, 2460617,
                2460619, 2460620, 2460621, 2460622, 2460623, 2460626, 2460635])

# daily analysis
stetson_daily = pd.read_csv(data_dir + "stetson_metrics_daily.txt", sep=' ', low_memory=False)

counts = np.zeros((14, len(dys)))
c_rms = np.zeros((3, len(dys)))
per_pass_lsst = np.zeros(len(dys))
per_pass_other = np.zeros(len(dys))

mag_pass = []
amp_pass = []
id_pass = []

for idx, dy in enumerate(dys):

    try:
        jstet_results = Varstats.stetson_j_peak_and_cutoff(stetson_daily[stetson_daily.cat_source =='toros'][str(dy) + '_j'],
                                                           sigma_method="mirror_std", n_sigma=3.0)
        jstet_cut = jstet_results['cutoff']

        lstet_results = Varstats.stetson_j_peak_and_cutoff(stetson_daily[stetson_daily.cat_source =='toros'][str(dy) + '_l'],
                                                           sigma_method="mirror_std", n_sigma=3.0)
        lstet_cut = lstet_results['cutoff']

        n_pass_lsst = len(stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                                        (stetson_daily[str(dy) + '_l'] > lstet_cut) &
                                        (stetson_daily['object_type'] == 'LSST')])


        mags = stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                             (stetson_daily[str(dy) + '_l'] > lstet_cut) &
                             (stetson_daily['object_type'] == 'LSST')].mag.to_numpy() - 5.4

        mag_pass.extend(mags)

        amps = stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                            (stetson_daily[str(dy) + '_l'] > lstet_cut) &
                            (stetson_daily['object_type'] == 'LSST')].d90.to_numpy()

        amp_pass.extend(amps)

        ids = stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                            (stetson_daily[str(dy) + '_l'] > lstet_cut) &
                            (stetson_daily['object_type'] == 'LSST')].name.to_numpy()

        id_pass.extend(ids)
    except:
        continue
mag_pass = np.array(mag_pass)
g_mags_pass = known_g[known_g.diaObjectId.isin(np.array(id_pass).astype(int))].g.to_numpy()
r_mags_pass = known_r[known_r.diaObjectId.isin(np.array(id_pass).astype(int))].r.to_numpy()
i_mags_pass = known_i[known_i.diaObjectId.isin(np.array(id_pass).astype(int))].i.to_numpy()
y_mags_pass = known_y[known_y.diaObjectId.isin(np.array(id_pass).astype(int))].y.to_numpy()


plt.figure(figsize=(9,6))

plt.scatter(mag_pass, amp_pass, marker='.', alpha=0.3, c='k')
plt.ylabel(r'Variable Star Amplitude ($\Delta_{90}$)', fontsize=15)
plt.yticks(fontsize=12)
plt.yscale('log')
plt.xticks(fontsize=12)
plt.xlabel(r"$T_V$", fontsize=15)
plt.savefig("amplitude_lsst_vars.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()


plt.figure(figsize=(9,6))

plt.boxplot([g_mags_pass[~np.isnan(g_mags_pass)], r_mags_pass[~np.isnan(r_mags_pass)],
             i_mags_pass[~np.isnan(i_mags_pass)], y_mags_pass[~np.isnan(y_mags_pass)], mag_pass],
            labels=['g', 'r', 'i', 'y', r'$T_V$'])
plt.ylabel('Magnitude Range', fontsize=15)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Filter", fontsize=15)
plt.savefig("mag_comparison.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()

Utils.log('The smallest amplitude recovery was ' + str(np.nanmin(amp_pass)) + '.', 'info')
Utils.log('The faintest magnitude recovery was ' + str(np.nanmax(mag_pass)) + '.', "info")
