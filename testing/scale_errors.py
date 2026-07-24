import matplotlib
import logging
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
import matplotlib.pyplot as plt
from config import Configuration
from libraries.utils import Utils
import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats as scs

# remove stars near 47 Tuc and the smallcluster
star_list = pd.read_csv("/Users/yuw816/Data/toros/commissioning/master/" + Configuration.FIELD + "/"
                        + Configuration.FIELD + "_star_list.txt", sep=' ', low_memory=False, index_col=0)

# redo the uncertainties?
reydo = 'Y'

if reydo == 'Y':
    f = open("/Volumes/OUMUAMUA/toros/commissioning/lc/" + Configuration.FIELD + "_errors.txt", 'w')
    f.write('file mag rms erms\n')
    for idx, row in star_list.iterrows():

        if row.chip < 10:
            lc = pd.read_csv('/Volumes/OUMUAMUA/toros/commissioning/lc/' + Configuration.FIELD + '/0' +
                             str(row.chip) + '/' + Configuration.FIELD + '_' + row.source_id + '.lc',
                             sep=' ')
        else:
            lc = pd.read_csv('/Volumes/OUMUAMUA/toros/commissioning/lc/' + Configuration.FIELD + '/' +
                             str(row.chip) + '/' + Configuration.FIELD + '_' + row.source_id + '.lc',
                             sep=' ')

        mag, _, full_rms = scs(lc[(lc.mag > 0) & (lc.err > 0)].mag, sigma=2.5)
        lc['dys'] = lc.jd.to_numpy().astype('int')

        rms_vals = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'std'}).to_numpy().flatten()
        num_obs = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'count'}).to_numpy().flatten()

        erms = lc[(lc.mag > 0) & (lc.err > 0)].err.mean()
        try:
            rms = np.min(rms_vals[num_obs >= 6])
        except:
            rms = full_rms

        f.write(Configuration.FIELD + '_' + row.source_id + '.lc ' +
                str(np.around(mag, decimals=4)) + ' ' +
                str(np.around(rms, decimals=4)) + ' ' +
                str(np.around(erms, decimals=4)) + '\n')
        if idx % 1000 == 0:
            Utils.log(str(len(star_list) - idx - 1) + ' stars remaining for error calculations.', "info")
    f.close()