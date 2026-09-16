import pandas as pd
from libraries.utils import Utils
comp_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/raw/"
str_dir = "/Volumes/OUMUAMUA/toros/commissioning/lc/FIELD_0e.001/raw/"

star_list = pd.read_csv("/Users/yuw816/Data/toros/commissioning/master/FIELD_0e.001/FIELD_0e.001_star_list.txt",
                        delimiter=' ',
                        header=0,
                        low_memory=False)

og_stars = star_list[star_list.source_id != star_list.lsst_id].copy().reset_index(drop=True)

for idx, row in og_stars.iterrows():
    if row.chip < 10:
        lc = pd.read_csv(comp_dir + '0' + str(row.chip) + "/FIELD_0e.001_" + str(row.source_id) + ".lc", sep=" ")
    else:
        lc = pd.read_csv(comp_dir + '' + str(row.chip) + "/FIELD_0e.001_" + str(row.source_id) + ".lc", sep=" ")
    if row.chip < 10:
        lc.to_csv(str_dir + '0' + str(row.chip) + "/FIELD_0e.001_" + str(row.source_id) + ".lc", sep=" ")
    else:
        lc.to_csv(str_dir + str(row.chip) + "/FIELD_0e.001_" + str(row.source_id) + ".lc", sep=" ")

    if idx % 1000 == 0:
        Utils.log(str(len(og_stars) - 1 - idx) + ' stars remain.', "info")
