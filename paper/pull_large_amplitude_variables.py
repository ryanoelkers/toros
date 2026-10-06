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
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from libraries.varstats import Varstats

data_dir = "/Volumes/nomad_vandy/toros/lc_stats/"
lc_dir = "/Volumes/nomad_vandy/toros/fin/"

# read in the varstats file, and exclude the LSST variables for now
fullstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)

varstats = fullstats[fullstats.cat_source == 'toros'].copy().reset_index(drop=True)

jstet_results = Varstats.stetson_j_peak_and_cutoff(varstats.jstet, sigma_method="mirror_std", n_sigma=3.0)
jstet_cut = jstet_results['cutoff']
lstet_results = Varstats.stetson_j_peak_and_cutoff(varstats.lstet, sigma_method="mirror_std", n_sigma=3.0)
lstet_cut = lstet_results['cutoff']

# plt.figure(figsize=(15,6))
#
# plt.subplot(1, 2, 1)
# plt.hist(varstats['jstet'], bins=30, range=[0, 30], histtype='step', color='k')
# plt.plot([jstet_cut, jstet_cut], [0, 62000], c='r', linewidth=3)
# plt.text(jstet_cut + .5, 50000, "J > " + str(np.around(jstet_cut, decimals=2)), fontsize=20, color="k")
# plt.xlabel('J', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylim([0, 60000])
# plt.ylabel('Count', fontsize=20)
# plt.yticks(fontsize=15)
#
# plt.subplot(1, 2, 2)
# plt.hist(varstats['lstet'], bins=30, range=[0, 30], histtype='step', color='k')
# plt.plot([lstet_cut, lstet_cut], [0, 82000], c='r', linewidth=3)
# plt.text(lstet_cut + .5, 65000, "L > " + str(np.around(lstet_cut, decimals=2)), fontsize=20, color="k")
# plt.xlabel('L', fontsize=20)
# plt.xticks(fontsize=15)
# plt.ylim([0, 80000])
# plt.ylabel('Count', fontsize=20)
# plt.yticks(fontsize=15)
#
# plt.savefig("toros_jl_cutoffs.png", dpi=200, bbox_inches='tight')
# plt.show()
# plt.close()# plt.figure(figsize=(15,6))
# #
# # plt.subplot(1, 2, 1)
# # plt.hist(varstats['jstet'], bins=30, range=[0, 30], histtype='step', color='k')
# # plt.plot([jstet_cut, jstet_cut], [0, 62000], c='r', linewidth=3)
# # plt.text(jstet_cut + .5, 50000, "J > " + str(np.around(jstet_cut, decimals=2)), fontsize=20, color="k")
# # plt.xlabel('J', fontsize=20)
# # plt.xticks(fontsize=15)
# # plt.ylim([0, 60000])
# # plt.ylabel('Count', fontsize=20)
# # plt.yticks(fontsize=15)
# #
# # plt.subplot(1, 2, 2)
# # plt.hist(varstats['lstet'], bins=30, range=[0, 30], histtype='step', color='k')
# # plt.plot([lstet_cut, lstet_cut], [0, 82000], c='r', linewidth=3)
# # plt.text(lstet_cut + .5, 65000, "L > " + str(np.around(lstet_cut, decimals=2)), fontsize=20, color="k")
# # plt.xlabel('L', fontsize=20)
# # plt.xticks(fontsize=15)
# # plt.ylim([0, 80000])
# # plt.ylabel('Count', fontsize=20)
# # plt.yticks(fontsize=15)
# #
# # plt.savefig("toros_jl_cutoffs.png", dpi=200, bbox_inches='tight')
# # plt.show()
# # plt.close()

# Do the daily metric cutoffs
fullstats.loc[(fullstats.jstet > jstet_cut) &
              (fullstats.lstet > lstet_cut) &
              (fullstats.pnts == 0) &
              (fullstats.edge == 0) &
              (fullstats.grps == 0), 'var_pass'] = 1

pass_vars = fullstats[(fullstats.var_pass == 1) & (fullstats.cat_source == 'toros')].copy().reset_index(drop=True)
pass_lsst = fullstats[(fullstats.var_pass == 1) & (fullstats.cat_source == 'lsst')].copy().reset_index(drop=True)

fullstats.to_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + Configuration.FIELD + "_varstats.txt",
                 sep=' ', header=True, index=False)

Utils.log("The number of TOROS stars with large amplitude variability is " + str(len(pass_vars)), "info")
Utils.log("The number of previous variables with large amplitude variability is: " + str(len(pass_vars[pass_vars.pvar == 1])), "info")
Utils.log("The number of previous xray with large amplitude variability is: " + str(len(pass_vars[pass_vars.xray == 1])), "info")
Utils.log("The number of lsst sources with large amplitude variability is " + str(len(pass_lsst)), "info")

plt.figure(figsize=(9,6))
plt.scatter(pass_vars.xcen, pass_vars.ycen,
            c='k', marker='.', label='TOROS Objects with Daily Variability', alpha=0.7)
plt.xlabel('X Pixel', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Y Pixel', fontsize=20)
plt.yticks(fontsize=15)
plt.savefig("toros_xy_daily.png", dpi=200, bbox_inches='tight')
# plt.show()
plt.close()

f = open("large_variable_table.txt", "w")
f.write("name ra dec rms jstet lstet cntm G47T N121 pvar xray lsst\n")
g = open("large_variable_table.mrt", "w")
for idx, row in pass_vars.iterrows():
    c = SkyCoord(ra=row.ra * u.deg, dec=row.dec * u.deg)
    ra_hms = c.ra.hms
    dec_dms = c.dec.dms
    ra_str = c.ra.to_string(unit=u.hourangle, sep=":", precision=2, pad=True)
    dec_str = c.dec.to_string(sep=":", precision=2, alwayssign=True, pad=True)

    name = str(row.source_id)
    ra = ra_str
    dec = dec_str
    v = str(np.around(row.v, decimals=3))
    rms = str(np.around(row.rms, decimals=3))
    jstet = str(np.around(row.jstet, decimals=2))
    lstet = str(np.around(row.lstet, decimals=2))
    cntm = str(int(row.cntm))
    G47T = str(int(row.G47T))
    N121 = str(int(row.N121))
    previous_var = str(int(row.pvar))
    previous_xray = str(int(row.xray))
    in_lsst = str(int(row.lsst))
    line = (name + " " + ra + " " + dec + " " + v + " " + rms + ' ' +
            jstet + " " + lstet + " " +
            cntm + " " +
            G47T + " " + N121 + " " + previous_var + " " + previous_xray + " " + in_lsst + "\n")
    g.write(line)

    if idx % 500 == 0:
        line = (name + " & " + ra + " & " + dec + " & " + v + " & " + rms + ' & ' +
                jstet + " & " + lstet + " & " +
                cntm + " & " +
                G47T + " & " + N121 + " & " + previous_var + " & " + previous_xray + " & " + in_lsst +  "\n")
        f.write(line)

f.close()
g.close()

# for idx, row in pass_vars.iterrows():
#
#         if row.G47T == 0:
#             if row.chip < 10:
#                     lc = pd.read_csv(lc_dir +
#                                      "0" + str(row.chip) + "/" +
#                                      Configuration.FIELD + "_" + str(row.source_id) + ".lc",
#                                      sep=" ")
#             else:
#                     lc = pd.read_csv(lc_dir +
#                                      str(row.chip) + "/" +
#                                      Configuration.FIELD + "_" + str(row.source_id) + ".lc",
#                                      sep=" ")
#
#             jd = lc[lc.mag > 0].jd.to_numpy()
#             mag = lc[lc.mag > 0].mag.to_numpy()
#             err = lc[lc.mag > 0].err.to_numpy()
#
#             plt.title(str(row.source_id) + " " + str(row.out_mag_std))
#             plt.errorbar(jd, mag, yerr=err, fmt='none', c='k')
#             plt.scatter(jd, mag, marker='.', c='k')
#             plt.gca().invert_yaxis()
#             plt.show()
