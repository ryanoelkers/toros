import pandas as pd
from config import Configuration
import numpy as np
from libraries.utils import Utils
import matplotlib
import logging
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
import matplotlib.pyplot as plt

data_dir = "/Volumes/OUMUAMUA/toros/commissioning/varstats/FIELD_0e.001/"

varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt", sep=' ', low_memory=False)
varstats = varstats[(varstats.source_id != varstats.lsst_id)].copy().reset_index(drop=True)

sim_pers = np.zeros(len(varstats))
pwr_rnk = np.zeros(len(varstats))
for idx, row in varstats.iterrows():

    sim_pers[idx] = len(varstats[varstats.prd == row.prd])
    pwr_rnk[idx] = len(varstats[(varstats.pwr > row.pwr) & (varstats.prd == row.prd)])

    if idx % 1000 == 0:
        Utils.log(str(len(varstats) - idx - 1), "info")

print('hold')