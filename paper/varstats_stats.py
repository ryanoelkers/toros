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

data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"
tv_zpt = 5.4
# read in the varstats file, and exclude the LSST variables for now
fullstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
fullstats['v'] = fullstats.mag - tv_zpt

one_metric = len(fullstats[((fullstats.day_pass == 1) |
                           (fullstats.per_pass == 1) |
                           (fullstats.var_pass == 1)) &
                           (fullstats.cat_source == 'toros')])
all_metric = len(fullstats[(fullstats.day_pass == 1) &
                           (fullstats.per_pass == 1) &
                           (fullstats.var_pass == 1) &
                           (fullstats.cat_source == 'toros')])
pass_var = len(fullstats[(fullstats.var_pass == 1) & (fullstats.cat_source == 'toros')])
pass_per = len(fullstats[(fullstats.per_pass == 1) & (fullstats.cat_source == 'toros')])
pass_day = len(fullstats[(fullstats.day_pass == 1) & (fullstats.cat_source == 'toros')])
Utils.log("All metrics " + str(all_metric), "info")
Utils.log("One metric " + str(one_metric), "info")
Utils.log("Var metrics " + str(pass_var), "info")
Utils.log("Per metrics " + str(pass_per), "info")
Utils.log("Day metrics " + str(pass_day), "info")

prv_var = len(fullstats[(fullstats.var_pass == 1) &
                        (fullstats.cat_source == 'toros') &
                        (fullstats.pvar == 1)])
prv_day = len(fullstats[(fullstats.day_pass == 1) &
                        (fullstats.cat_source == 'toros') &
                        (fullstats.pvar == 1)])
prv_per = len(fullstats[(fullstats.per_pass == 1) &
                        (fullstats.cat_source == 'toros') &
                        (fullstats.pvar == 1)])
tot_vars = len(fullstats[(fullstats.cat_source == 'toros') &
                        (fullstats.pvar == 1)])
xray_var = len(fullstats[(fullstats.var_pass == 1) &
                        (fullstats.cat_source == 'toros') &
                        (fullstats.xray == 1)])
xray_per = len(fullstats[(fullstats.per_pass == 1) &
                        (fullstats.cat_source == 'toros') &
                        (fullstats.xray == 1)])
xray_day = len(fullstats[(fullstats.day_pass == 1) &
                        (fullstats.cat_source == 'toros') &
                        (fullstats.xray == 1)])
tot_xray = len(fullstats[(fullstats.cat_source == 'toros') &
                        (fullstats.xray == 1)])

Utils.log("Total variables " + str(tot_vars), "info")
Utils.log("Previous variables " + str(prv_var), "info")
Utils.log("Previous periodics " + str(prv_per), "info")
Utils.log("Previous daily " + str(prv_day), "info")
Utils.log("Total xray " + str(tot_xray), "info")
Utils.log("Xray variables " + str(xray_var), "info")
Utils.log("Xray periodics " + str(xray_per), "info")
Utils.log("Xray daily " + str(xray_day), "info")

one_metric = len(fullstats[((fullstats.day_pass == 1) |
                           (fullstats.per_pass == 1) |
                           (fullstats.var_pass == 1)) &
                           (fullstats.cat_source == 'lsst')])
all_metric = len(fullstats[(fullstats.day_pass == 1) &
                           (fullstats.per_pass == 1) &
                           (fullstats.var_pass == 1) &
                           (fullstats.cat_source == 'lsst')])
pass_var = len(fullstats[(fullstats.var_pass == 1) & (fullstats.cat_source == 'lsst')])
pass_per = len(fullstats[(fullstats.per_pass == 1) & (fullstats.cat_source == 'lsst')])
pass_day = len(fullstats[(fullstats.day_pass == 1) & (fullstats.cat_source == 'lsst')])
tot_lsst = len(fullstats[fullstats.cat_source == 'lsst'])

Utils.log("Total LSST " + str(tot_lsst), "info")
Utils.log("All metrics LSST " + str(all_metric), "info")
Utils.log("One metric LSST " + str(one_metric), "info")
Utils.log("Var metrics LSST " + str(pass_var), "info")
Utils.log("Per metrics LSST " + str(pass_per), "info")
Utils.log("Day metrics LSST " + str(pass_day), "info")

limit_lsst = fullstats[((fullstats.day_pass == 1) | (fullstats.per_pass == 1) | (fullstats.var_pass == 1)) &
                       (fullstats.cat_source == 'lsst') &
                       (fullstats.v > 19)].copy().reset_index(drop=True)
lsst = fullstats[((fullstats.day_pass == 1) | (fullstats.per_pass == 1) | (fullstats.var_pass == 1)) &
                       (fullstats.cat_source == 'lsst')].copy().reset_index(drop=True)
v_max_lsst = limit_lsst.v.max()
v_min_lsst = lsst.v.min()
snr_range = np.array([1/ limit_lsst.rms.quantile(.95), 1/ limit_lsst.rms.quantile(.05)])
v_max_toros = fullstats[fullstats.cat_source == 'toros'].v.max()

Utils.log("Min magnitude for LSST is " + str(v_max_lsst), "info")
Utils.log("Max magnitude for LSST is " + str(v_min_lsst), "info")
Utils.log("Min magnitude for TOROS is " + str(v_max_toros), "info")
Utils.log("SNR range for LSST is " + str(int(snr_range[0])) + "-" + str(int(snr_range[1])), "info")

# write out the rms table

# f = open("rms_table.txt", "w")
# f.write("name ra dec rms d90 minrms fullrms\n")
# g = open("rms_table.mrt", "w")
#
# varstats = fullstats[fullstats.cat_source == 'toros'].copy().reset_index(drop=True)
#
# for idx, row in varstats.iterrows():
#     c = SkyCoord(ra=row.ra * u.deg, dec=row.dec * u.deg)
#     ra_hms = c.ra.hms
#     dec_dms = c.dec.dms
#     ra_str = c.ra.to_string(unit=u.hourangle, sep=":", precision=2, pad=True)
#     dec_str = c.dec.to_string(sep=":", precision=2, alwayssign=True, pad=True)
#
#     name = str(row.source_id)
#     ra = ra_str
#     dec = dec_str
#     v = str(np.around(row.v, decimals=3))
#     rms = str(np.around(row.rms, decimals=3))
#     d90 = str(np.around(row.d90, decimals=3))
#     minrms = str(np.around(row.min_rms, decimals=3))
#     fullrms = str(np.around(row.full_rms, decimals=3))
#
#     line = (name + " " + ra + " " + dec + " " + v + " " + rms + ' ' +
#             d90 + " " + minrms + " " + fullrms + "\n")
#     g.write(line)
#
#     if idx % 10000 == 0:
#         line = (name + " & " + ra + " & " + dec + " & " + v + " & " + rms + " & " +
#                 d90 + " & " + minrms + " & " + fullrms + "\n")
#         f.write(line)
#
# f.close()
# g.close()
