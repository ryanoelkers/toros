import pandas as pd
import numpy as np
import matplotlib
import logging
import random
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from config import Configuration
from libraries.varstats import Varstats
from scipy.stats import median_abs_deviation as mad

# read in the varstats file
varstats = pd.read_csv(Configuration.LIGHTCURVE_FIELD_DIRECTORY + Configuration.FIELD + "_varstats.txt",
                       sep=' ', low_memory=False)

# read in the first light curve just to get the required dates
if varstats.iloc[0].chip < 10:
    lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RESCALE_DIRECTORY + '/0' + str(varstats.iloc[0].chip) + '/' +
                     Configuration.FIELD + '_' + str(varstats.iloc[0].source_id) + '.lc',
                     sep=" ")
else:
    lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RESCALE_DIRECTORY + '/' + str(varstats.iloc[0].chip) + '/' +
                     Configuration.FIELD + '_' + str(varstats.iloc[0].source_id) + '.lc',
                     sep=" ")
jd = lc.jd.to_numpy()
del lc

# get the dates for the output
dys = np.unique(jd.astype(int))  # the unique julian dates
dys = np.delete(dys, -2)  # get rid of a day with a single observation
dys_int = jd.astype(int)  # the integer date for every observation
dys_int = np.delete(dys_int, 242)  # get rid of a day with a single observation

# open the output files
f_dy = open("stetson_metrics_daily.txt", "w")
f_dys = open("stetson_metrics_cumulative.txt", "w")

# set up the header file
header = "name mag d90 object_type"
for dy in dys:
    header = header + " " + str(int(dy)) + "_j"
    header = header + " " + str(int(dy)) + "_l"
header = header + "\n"

# write the headers to the file
f_dy.write(header)
f_dys.write(header)

# now loop through each light curve getting the daily j/l, and the cumulative j/l
for idx, row in varstats.iterrows():

    # read in the light curve
    if row.chip < 10:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RESCALE_DIRECTORY + '/0' + str(row.chip) + '/' +
                         Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                         sep=" ")
    else:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RESCALE_DIRECTORY + '/' + str(row.chip) + '/' +
                         Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                         sep=" ")

    # set up the line for the star
    line_dy = str(row.source_id) + " " + str(row.mag) + " " + str(row.d90) + " " + str(row.object_type)
    line_dys = str(row.source_id) + " " + str(row.mag) + " " + str(row.d90) + " " + str(row.object_type)

    # calculate the j/l per day and cumulative
    for dy in dys:

        # get the indecies to calculate j/l
        times_dy = np.argwhere(dys_int == dy)
        times_dys = np.argwhere(dys_int <= dy)

        # pull out the light curve data
        mag_dy = lc.mag.to_numpy()[times_dy].flatten()
        err_dy = lc.err.to_numpy()[times_dy].flatten()
        mag_dys = lc.mag.to_numpy()[times_dys].flatten()
        err_dys = lc.err.to_numpy()[times_dys].flatten()

        # calculate the metrics
        # for daily obs
        j_dy, _, l_dy = Varstats.stetson_metrics(mag_dy[(mag_dy > 0) & (err_dy > 0)],
                                                 err_dy[(mag_dy > 0) & (err_dy > 0)])
        line_dy = line_dy + " " + str(np.around(j_dy, decimals=3))
        line_dy = line_dy + " " + str(np.around(l_dy, decimals=3))

        # for cumulative obs
        j_dys, _, l_dys = Varstats.stetson_metrics(mag_dys[(mag_dys > 0) & (err_dys > 0)],
                                                   err_dys[(mag_dys > 0) & (err_dys > 0)])
        line_dys = line_dys + " " + str(np.around(j_dys, decimals=3))
        line_dys = line_dys + " " + str(np.around(l_dys, decimals=3))

    # end the line
    line_dy = line_dy + "\n"
    line_dys = line_dys + "\n"

    # write the line
    f_dy.write(line_dy)
    f_dys.write(line_dys)

    if idx % 1000 == 0:
        Utils.log("Working on the next 1000. " + str(len(varstats) - idx - 1) + " stars remain.", "info")

f_dy.close()
f_dys.close()

Utils.log("All finished. See ya later alligator.", "info")
