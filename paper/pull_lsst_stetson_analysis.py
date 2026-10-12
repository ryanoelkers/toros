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
data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"

# data_dir = Configuration.LIGHTCURVE_FIELD_DIRECTORY
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
varstats['v'] = varstats['mag'] - 5.4

lsststats = varstats[varstats.cat_source == 'lsst'].copy().reset_index(drop=True)


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

n_var_pass_lsst_total = len(lsststats[lsststats.var_pass == 1])

Utils.log("The number of LSST stars that pass the cuts with the full lc are: " + str(n_var_pass_lsst_total),
          "info")
Utils.log("As a percentage that is: " + str(np.around(n_var_pass_lsst_total / len(lsststats) * 100, decimals=2)) + "%",
          "info")

dys = np.array([2460584, 2460586, 2460599, 2460600, 2460601, 2460614, 2460617,
                2460619, 2460620, 2460621, 2460622, 2460623, 2460626, 2460635])
# cumulative analysis
stetson_cuml = pd.read_csv(data_dir + "stetson_metrics_cumulative.txt", sep=' ', low_memory=False)
n_pass_lsst = np.zeros(len(dys))
stetson_cuml['day_pass'] = 15
stetson_cuml['grps'] = varstats['grps'].to_numpy()
stetson_cuml['edge'] = varstats['edge'].to_numpy()
stetson_cuml['pnts'] = varstats['pnts'].to_numpy()

for idx, dy in enumerate(dys):

    jstet_results = Varstats.stetson_j_peak_and_cutoff(stetson_cuml[stetson_cuml.cat_source =='toros'][str(dy) + '_j'],
                                                       sigma_method="mirror_std", n_sigma=3.0)
    jstet_cut = jstet_results['cutoff']

    lstet_results = Varstats.stetson_j_peak_and_cutoff(stetson_cuml[stetson_cuml.cat_source =='toros'][str(dy) + '_l'],
                                                       sigma_method="mirror_std", n_sigma=3.0)
    lstet_cut = lstet_results['cutoff']

    n_pass_lsst[idx] = len(stetson_cuml[(stetson_cuml[str(dy) + '_j'] > jstet_cut) &
                                        (stetson_cuml[str(dy) + '_l'] > lstet_cut) &
                                        (stetson_cuml['cat_source'] == 'lsst') &
                                        (stetson_cuml.grps == 0) & (stetson_cuml.pnts == 0) & (stetson_cuml.edge ==0)])

    stetson_cuml.loc[(stetson_cuml[str(dy) + '_j'] > jstet_cut) &
                     (stetson_cuml[str(dy) + '_l'] > lstet_cut) &
                     (stetson_cuml['cat_source'] == 'lsst') &
                     (stetson_cuml.day_pass == 15) &
                     (stetson_cuml.grps == 0) & (stetson_cuml.pnts == 0) & (stetson_cuml.edge ==0), 'day_pass'] = idx + 1

dy_pass_cum = np.zeros(14)
dys_obs = np.arange(14) + 1

mn_pass, mdn_pass, sg_pass = scs(n_pass_lsst, sigma=2.5)
for idx in range(1, 15):
    dy_pass_cum[idx-1] = np.around(len(stetson_cuml[stetson_cuml.day_pass <= idx]) / len(stetson_cuml[stetson_cuml.day_pass <= 14]), decimals=2)

Utils.log("The avar number of stars recovered cumulatively is: " + str(np.sum(n_pass_lsst)), "info")
Utils.log("The unique number of stars recovered cumulativel is: " + str(len(stetson_cuml[stetson_cuml.day_pass <= 14])), "info")
plt.figure(figsize=(9,6))
plt.plot(dys_obs, dy_pass_cum, c='k', linewidth=2)
plt.xlabel('Number of Days of Observation', fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('Cumulative Fraction of Recovered Variables', fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("toros_daily_cumulative.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()
# daily analysis
stetson_daily = pd.read_csv(data_dir + "stetson_metrics_daily.txt", sep=' ', low_memory=False)
n_pass_lsst = np.zeros(len(dys))
id_pass = []

for idx, dy in enumerate(dys):

    jstet_results = Varstats.stetson_j_peak_and_cutoff(stetson_daily[stetson_daily.cat_source =='toros'][str(dy) + '_j'],
                                                       sigma_method="mirror_std", n_sigma=3.0)
    jstet_cut = jstet_results['cutoff']

    lstet_results = Varstats.stetson_j_peak_and_cutoff(stetson_daily[stetson_daily.cat_source =='toros'][str(dy) + '_l'],
                                                       sigma_method="mirror_std", n_sigma=3.0)
    lstet_cut = lstet_results['cutoff']

    n_pass_lsst[idx] = len(stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                                    (stetson_daily[str(dy) + '_l'] > lstet_cut) &
                                    (stetson_daily['cat_source'] == 'lsst')])

    ids = stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                        (stetson_daily[str(dy) + '_l'] > lstet_cut) &
                        (stetson_daily['cat_source'] == 'lsst')].name.to_numpy()

    id_pass.extend(ids)

id_pass_uni = np.unique(id_pass)

lsst = varstats[varstats.source_id.isin(np.array(id_pass_uni)) &
                         (varstats.pnts == 0) & (varstats.edge == 0) & (varstats.grps == 0)].copy()

Utils.log("The number of unique stars recovered is: " + str(len(lsst)), "info")
Utils.log("On average the number of stars recovered each night is " +
          str(np.around(len(id_pass) / len(dys), decimals=0)), "info")
lsst["source_id"] = lsst["source_id"].astype("int64")
lsststats = lsst.merge(known_r, left_on='source_id', right_on='diaObjectId', how='inner')
lsststats = lsststats[['star_id', 'v', 'rms', 'd90', 'r']].copy().reset_index(drop=True)
lsststats.to_csv("jstet_daily_r.txt", sep=',', index=False)
