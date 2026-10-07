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

# read in the varstats file, and exclude the LSST variables for now
fullstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)

varstats = fullstats[fullstats.cat_source == 'toros'].copy().reset_index(drop=True)
lsststats = fullstats[fullstats.cat_source == 'lsst'].copy().reset_index(drop=True)

# Do the daily metric cutoffs
fullstats.loc[(fullstats.out_mag_std > 0) &
              (fullstats.pnts == 0) &
              (fullstats.edge == 0) &
              (fullstats.grps == 0), 'day_pass'] = 1

pass_days = fullstats[(fullstats.day_pass == 1) & (fullstats.cat_source == 'toros')].copy().reset_index(drop=True)
pass_lsst = fullstats[(fullstats.day_pass == 1) & (fullstats.cat_source == 'lsst')].copy().reset_index(drop=True)

fullstats.to_csv(data_dir + Configuration.FIELD + "_varstats.txt",
                 sep=' ', header=True, index=False)

Utils.log("The number of TOROS stars with some kind of outburst between nights is " + str(len(pass_days)), "info")
Utils.log("The number of previous variables with an outburst is: " + str(len(pass_days[pass_days.pvar == 1])), "info")
Utils.log("The number of previous xray with an outburst is: " + str(len(pass_days[pass_days.xray == 1])), "info")
Utils.log("The number of lsst sources with an outburst is: " + str(len(pass_lsst)), "info")

plt.figure(figsize=(9,6))
plt.scatter(pass_days.xcen, pass_days.ycen,
            c='k', marker='.', label='TOROS Objects with Daily Variability', alpha=0.7)
plt.xlabel('X Pixel', fontsize=20)
plt.xticks(fontsize=15)
plt.ylabel('Y Pixel', fontsize=20)
plt.yticks(fontsize=15)
plt.savefig("toros_xy_daily.png", dpi=200, bbox_inches='tight')
# plt.show()
plt.close()

f = open("daily_variable_table.txt", "w")
f.write("name ra dec rms d90 ndays cntm G47T N121 pvar xray lsst\n")
g = open("daily_variable_table.mrt", "w")
for idx, row in pass_days.iterrows():
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
    d90 = str(np.around(row.d90, decimals=3))
    days = str(int(row.out_mag_std))
    cntm = str(int(row.cntm))
    G47T = str(int(row.G47T))
    N121 = str(int(row.N121))
    previous_var = str(int(row.pvar))
    previous_xray = str(int(row.xray))
    in_lsst = str(int(row.lsst))
    line = (name + " " + ra + " " + dec + " " + v + " " + rms + ' ' + d90 + " " + days + " " +
            cntm + " " +
            G47T + " " + N121 + " " + previous_var + " " + previous_xray + " " + in_lsst + "\n")
    g.write(line)

    if idx % 500 == 0:
        line = (name + " & " + ra + " & " + dec + " & " + v + " & " + rms + " & " + d90 + " & " + days + " & " +
                cntm + " & " +
                G47T + " & " + N121 + " & " + previous_var + " & " + previous_xray + " & " + in_lsst +  "\n")
        f.write(line)

f.close()
g.close()