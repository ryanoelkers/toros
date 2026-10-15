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

other_stats = varstats[(varstats.var_pass == 1) | (varstats.per_pass == 1) | (varstats.day_pass == 1)].copy().reset_index(drop=True)
other_stats.to_csv("filtergraph.csv", index=False, sep=',')

lsststats = varstats[varstats.cat_source == 'lsst'].copy().reset_index(drop=True)

# get the known LSST variables
lsst_dir = "/Users/yuw816/Data/toros/commissioning/lsst/"
known = pd.read_csv(lsst_dir + "lsst_data_47tuc_variables.csv",
                    delimiter=',',
                    header=0,
                    low_memory=False,
                    index_col=0)
tot_known = known.groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean'}).reset_index()
known['psfMag'] = -2.5 * np.log10(known.psfFlux) + 31.4

Utils.log('We provide photometry for ' + str(len(lsststats) / len(tot_known) * 100) + " lsst stars.", "info")
known_g = known[known.band == 'g'].groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean', 'psfFlux': 'mean', 'psfMag': 'std'}).reset_index()
known_r = known[known.band == 'r'].groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean', 'psfFlux': 'mean', 'psfMag': 'std'}).reset_index()
known_i = known[known.band == 'i'].groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean', 'psfFlux': 'mean', 'psfMag': 'std'}).reset_index()
known_y = known[known.band == 'y'].groupby('diaObjectId').agg({'coord_ra': 'mean', 'coord_dec': 'mean', 'psfFlux': 'mean', 'psfMag': 'std'}).reset_index()

known_g['g'] = -2.5 * np.log10(known_g.psfFlux) + 31.4
known_r['r'] = -2.5 * np.log10(known_r.psfFlux) + 31.4
known_i['i'] = -2.5 * np.log10(known_i.psfFlux) + 31.4
known_y['y'] = -2.5 * np.log10(known_y.psfFlux) + 31.4
lsststats["source_id"] = lsststats["source_id"].astype("int64")
lsst_g = lsststats.merge(known_g, left_on='source_id', right_on='diaObjectId', how='inner')
lsst_r = lsststats.merge(known_r, left_on='source_id', right_on='diaObjectId', how='inner')
lsst_i = lsststats.merge(known_i, left_on='source_id', right_on='diaObjectId', how='inner')


mean_g = lsst_g[(lsst_g.cntm == 0) & (lsst_g.v > 19)].g.mean()
rms_g = lsst_g[(lsst_g.cntm == 0) & (lsst_g.v > 19)].rms.quantile([.25, .75]).to_numpy()
mean_r = lsst_r[(lsst_r.cntm == 0) & (lsst_r.v > 19)].r.mean()
rms_r = lsst_r[(lsst_r.cntm == 0) & (lsst_r.v > 19)].rms.quantile([.25, .75]).to_numpy()
mean_i = lsst_i[(lsst_i.cntm == 0) & (lsst_i.v > 19)].i.mean()
rms_i = lsst_i[(lsst_i.cntm == 0) & (lsst_i.v > 19)].rms.quantile([.25, .75]).to_numpy()
Utils.log("The mean g: " + str(np.around(mean_g, decimals=1)) +
          " mean r: " + str(np.around(mean_r, decimals=1)) +
          " mean i: " + str(np.around(mean_i, decimals=1)), "info")
Utils.log("The typical SNR is " + str(1/np.mean([rms_g, rms_r, rms_i],axis=0)[1]) +
          " to " +str(1/np.mean([rms_g, rms_r, rms_i],axis=0)[0]), "info")

tot_cov_g = len(known_g[known_g.g < mean_g])
tot_cov_g_id = known_g[known_g.g < mean_g].diaObjectId.tolist()
tot_cov_r = len(known_r[known_r.r < mean_r])
tot_cov_r_id = known_r[known_r.r < mean_r].diaObjectId.tolist()
tot_cov_i = len(known_i[known_i.i < mean_i])
tot_cov_i_id = known_i[known_i.i < mean_i].diaObjectId.tolist()
fin_id = tot_cov_g_id + tot_cov_r_id + tot_cov_i_id
unique_list = list(dict.fromkeys(fin_id))

Utils.log("TOROS can provide coverage of approximately " +
          str(np.around(len(unique_list) / len(tot_known) * 100, decimals=0)) + "% in the Forced Sourced list.",
          "info")
n_var_pass_lsst_total = len(lsststats[(lsststats.var_pass == 1) | (lsststats.per_pass == 1) | (lsststats.day_pass ==1)])

Utils.log("The number of LSST stars that pass the cuts with the full lc are: " + str(n_var_pass_lsst_total),
          "info")
Utils.log("As a percentage that is: " + str(np.around(n_var_pass_lsst_total / len(lsststats) * 100, decimals=2)) + "%",
          "info")

dys = np.array([2460584, 2460586, 2460599, 2460600, 2460601, 2460614, 2460617,
                2460619, 2460620, 2460621, 2460622, 2460623, 2460626, 2460635])


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

    ids = stetson_daily[(stetson_daily[str(dy) + '_j'] > jstet_cut) &
                        (stetson_daily[str(dy) + '_l'] > lstet_cut) &
                        (stetson_daily['cat_source'] == 'lsst')].name.to_numpy()

    lsst_pass = varstats[varstats.source_id.isin(np.array(ids)) &
                         (varstats.pnts == 0) & (varstats.edge == 0) & (varstats.grps == 0)].copy()

    n_pass_lsst[idx] = len(lsst_pass)

    id_pass.extend(ids)

id_pass_uni = np.unique(id_pass)

lsst = varstats[~varstats.source_id.isin(np.array(id_pass_uni)) &
                         (varstats.pnts == 0) & (varstats.edge == 0) & (varstats.grps == 0)].copy()

mn_d90, mdn_d90, sg_d90 = scs(lsst.d90.to_numpy(), sigma=3)

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.bar(dys - 2460580, n_pass_lsst, color='k', edgecolor='black')
plt.xlabel('JD - 2460580', fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('Number of Flagged Rubin-LSST Variables', fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig("toros_daily_per_day.png", dpi=200, bbox_inches='tight')
# #plt.show()
# plt.close()

mn_pass, mdn_pass, sg_pass = scs(n_pass_lsst, sigma=3)

Utils.log("The number of unique stars recovered on a daily basis is: " + str(len(lsst)) , "info")
Utils.log("As a percentage that is: " + str(np.around(len(lsst) / len(lsststats) * 100, decimals=2)) + "%",
          "info")
Utils.log("On average the number of stars recovered each night is " +
          str(np.around(mn_pass, decimals=0)) + "+/-" + str(np.around(sg_pass, decimals=0)), "info")


lsst["source_id"] = lsst["source_id"].astype("int64")
lsst_plots = lsst.merge(known_r, left_on='source_id', right_on='diaObjectId', how='inner')

# plt.figure(figsize=(9,6))
# plt.scatter(np.flip(lsst_plots.r.to_numpy()), np.flip(lsst_plots.d90.to_numpy()),
#             c=np.flip(lsst_plots.v.to_numpy()), cmap='inferno_r')
# plt.yscale('log')
# plt.colorbar().set_label(r'$T_V$', fontsize=12)
# plt.xlabel('Rubin-LSST r', fontsize=15)
# plt.xticks(fontsize=15)
# plt.ylabel(r'Variable Amplitude [$\Delta_{90}$]', fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig("toros_daily_r_v_d90.png", dpi=200, bbox_inches='tight')
# plt.show()
# plt.close()

# plt.figure(figsize=(9,6))
# plt.scatter(np.flip(lsst_plots.r.to_numpy()), np.flip(lsst_plots.rms.to_numpy()),
#             c=np.flip(lsst_plots.v.to_numpy()), cmap='inferno_r')
# plt.yscale('log')
# plt.colorbar().set_label(r'$T_V$', fontsize=12)
# plt.xlabel('Rubin-LSST r', fontsize=15)
# plt.xticks(fontsize=15)
# plt.ylabel('rms', fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig("toros_daily_r_v_rms.png", dpi=200, bbox_inches='tight')
# plt.show()
# plt.close()

# cumulative analysis
stetson_cuml = pd.read_csv(data_dir + "stetson_metrics_cumulative.txt", sep=' ', low_memory=False)
n_pass_lsst = np.zeros(len(dys))
stetson_cuml['day_pass'] = 15
stetson_cuml['grps'] = varstats['grps'].to_numpy()
stetson_cuml['edge'] = varstats['edge'].to_numpy()
stetson_cuml['pnts'] = varstats['pnts'].to_numpy()

day_sep = np.zeros(14)

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
                     (stetson_cuml.grps == 0) & (stetson_cuml.pnts == 0) & (stetson_cuml.edge == 0), 'day_pass'] = idx + 1
    day_sep = dy - dys[0]

dy_pass_cum = np.zeros(14)
dys_obs = np.arange(14) + 1

mn_pass, mdn_pass, sg_pass = scs(n_pass_lsst, sigma=3)
for idx in range(1, 15):
    dy_pass_cum[idx-1] = np.around(len(stetson_cuml[stetson_cuml.day_pass <= idx]) / len(stetson_cuml[stetson_cuml.day_pass <= 14]), decimals=2)

Utils.log("The number of stars recovered after 2 days is: " + str(np.around(dy_pass_cum[1], decimals=2)) +
                                                                  " and after 4 is " +
                                                                  str(np.around(dy_pass_cum[3], decimals=2)), "info")
Utils.log("The average number of stars recovered cumulatively is: " + str(int(mn_pass)) + "+/-" + str(int(sg_pass)), "info")
Utils.log("The unique number of stars recovered cumulatively is: " + str(len(stetson_cuml[stetson_cuml.day_pass <= 14])), "info")
plt.subplot(1, 2, 2)
plt.plot(dys_obs, dy_pass_cum, c='k', linewidth=2)
plt.xlabel('Cumulative Number of Days of Observation', fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel('Cumulative Fraction of Recovered Variables', fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("toros_jstet_analysis.png", dpi=200, bbox_inches='tight')
plt.show()
plt.close()

