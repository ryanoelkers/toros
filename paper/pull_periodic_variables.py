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
from astropy.stats import sigma_clipped_stats as scs
from astropy.coordinates import SkyCoord
from astropy import units as u

data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"

# read in the varstats file, and exclude the LSST variables for now
fullstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
fullstats.per_pass = 0
fullstats.loc[(fullstats.prnk == 0) &
              (fullstats.pnts == 0) &
              (fullstats.fap < 0.001) &
              (fullstats.grps == 0) &
              (fullstats.edge == 0) &
              (fullstats.prd > 0) &
              (fullstats.otlr < 10), 'per_pass'] = 1

fullstats.to_csv(data_dir + Configuration.FIELD + "_varstats.txt",
                 sep=' ', header=True, index=False)
pass_pers = fullstats[(fullstats.per_pass == 1) & (fullstats.cat_source == 'toros')].copy().reset_index(drop=True)
pass_lsst = fullstats[(fullstats.per_pass == 1) & (fullstats.cat_source == 'lsst')].copy().reset_index(drop=True)

# output the table for the paper
f = open("periodic_variable_table.txt", "w")
f.write("name ra dec rms period power fap simp cntm otlr G47T N121 pvar xray lsst\n")
g = open("periodic_variable_table.mrt", "w")
for idx, row in pass_pers.iterrows():
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
    period = str(np.around(row.prd, decimals=6))
    power = str(np.around(row.pwr, decimals=3))
    fap = str(np.around(row.fap, decimals=3))
    simp = str(int(row.simp))
    cntm = str(int(row.cntm))
    otlr = str(int(row.otlr))
    G47T = str(int(row.G47T))
    N121 = str(int(row.N121))
    previous_var = str(int(row.pvar))
    previous_xray = str(int(row.xray))
    in_lsst = str(int(row.lsst))
    line = (name + " " + ra + " " + dec + " " + v + " " + rms + ' ' +
            period + " " + power + " " + fap + " " +
            simp + " " + cntm + " " + otlr + " " +
            G47T + " " + N121 + " " + previous_var + " " + previous_xray + " " + in_lsst + "\n")
    g.write(line)

    if idx % 250 == 0:
        line = (name + " & " + ra + " & " + dec + " & " + v + " & " + rms + " & " +
                period + " & " + power + " & " + fap + " & " +
                simp + " & " + cntm + " & " + otlr + " & " +
                G47T + " & " + N121 + " & " + previous_var + " & " + previous_xray + " & " + in_lsst +  "\n")
        f.write(line)

f.close()
g.close()

Utils.log("The number of TOROS stars with a period is: " + str(len(pass_pers)), "info")
Utils.log("The number of previous variables with a period is: " + str(len(pass_pers[pass_pers.pvar == 1])), "info")
Utils.log("The number of previous xray with a period is: " + str(len(pass_pers[pass_pers.xray == 1])), "info")
Utils.log("The number of lsst sources with a period is: " + str(len(pass_lsst)), "info")

plt.figure(figsize=(9,6))
plt.scatter(pass_pers.xcen, pass_pers.ycen,
            c='k', marker='.', label='TOROS Objects with Periodic Variability', alpha=0.7)
plt.xlabel('X Pixel', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Y Pixel', fontsize=20)
plt.yticks(fontsize=15)
plt.savefig("toros_xy_periodic.png", dpi=200, bbox_inches='tight')
# plt.show()
plt.close()
