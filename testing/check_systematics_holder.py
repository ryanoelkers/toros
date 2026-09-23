import pandas as pd
import matplotlib
import logging
from libraries.utils import Utils
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
from config import Configuration
from astropy.stats import sigma_clipped_stats as scs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

std_id = 25921 # 25921
# read in the star list
star_list = pd.read_csv("/Volumes/OUMUAMUA/toros/commissioning/master/FIELD_0e.001/"
                        + Configuration.FIELD + '_star_list.txt',
                        delimiter=' ',
                        header=0,
                        low_memory=False)
# star_list['gc_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
#                                 (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)

#gc_star = star_list.loc[std_id, 'gc_star']
xcen = star_list.loc[std_id, 'xcen']
ycen = star_list.loc[std_id, 'ycen']
mean_mag = star_list.loc[std_id, 'master_mag']

# the directories where the light curves reside
dir = "/Volumes/OUMUAMUA/toros/commissioning/lc/FIELD_0e.001/raw/"
rs_dir = "/Volumes/OUMUAMUA/toros/commissioning/lc/FIELD_0e.001/rescale/"
dirold = "/Volumes/OUMUAMUA/toros/commissioning/lc/FIELD_0e.001_hold/"

ntr_stars = 500

nme=[]

# get the unique days
if star_list.loc[std_id, 'chip'] < 10:
    lc_cmp = pd.read_csv(rs_dir + "0" + str(star_list.loc[std_id, 'chip']) + "/FIELD_0e.001_" +
                         str(star_list.loc[std_id, 'source_id']) + ".lc",
                         sep=" ")
    lc_tst = pd.read_csv(dir + "0" + str(star_list.loc[std_id, 'chip']) + "/FIELD_0e.001_" +
                         str(star_list.loc[std_id, 'source_id']) + ".lc",
                         sep=" ")
else:
    lc_cmp = pd.read_csv(rs_dir + str(star_list.loc[std_id, 'chip']) + "/FIELD_0e.001_" +
                     str(star_list.loc[std_id, 'source_id'])+".lc",
                     sep=" ")
    lc_tst = pd.read_csv(dir + str(star_list.loc[std_id, 'chip']) + "/FIELD_0e.001_" +
                     str(star_list.loc[std_id, 'source_id'])+".lc",
                     sep=" ")
lc_tst['dys'] = lc_tst.jd.astype(int)
lc_cmp['dys'] = lc_cmp.jd.astype(int)
d_dys = lc_tst.jd.to_numpy()
s_dys = lc_tst.dys.astype(str).to_numpy()
u_dys = lc_tst.dys.astype(str).unique()
n_dys = lc_tst.groupby('dys')['mag'].count().values
lc_sze = len(lc_tst)

# the holders for each day
lc_hold = np.zeros([lc_sze, ntr_stars])
x_hold = np.zeros([ntr_stars])
y_hold = np.zeros([ntr_stars])

indx = 0
for idx, row in star_list[std_id - 500:std_id + 500].iterrows():

    # update the name vector
    nme.append(str(row.star_id))

    # read in the light curves
    try:
        if row.chip < 10:
            lc = pd.read_csv(dir + '/0' + str(row.chip) + '/' +
                             Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                             sep=" ")
        else:
            lc = pd.read_csv(dir + '/' + str(row.chip) + '/' +
                             Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                             sep=" ")
        if idx % 1000 == 0:
            Utils.log("Finished with 1000.", "info")

        # update the matrix values
        lc_hold[:, indx] = lc.mag.to_numpy() - lc[lc.mag > 0].mag.median()
        x_hold[indx] = row.xcen
        y_hold[indx] = row.ycen

    except:
        continue
        # lc_hold[:, idx] = np.zeros(lc_sze) - 9.999
    indx = indx + 1
df = pd.DataFrame(lc_hold, columns=nme)
cc = df.corrwith(lc_tst.mag - lc_tst[lc_tst.mag > 0].mag.median())

dist = np.sqrt((x_hold - xcen) ** 2 + (y_hold - ycen) ** 2)
pss_str = np.argwhere((cc > .8) & (dist > 16)).flatten()
wgts = cc[pss_str]

if len(wgts) > 0:
    lc_avg = np.average(df.loc[:, np.array(nme)[pss_str]], axis=1, weights=wgts)

    # data frame generation from a matrix
    cln_lc = np.zeros(lc_sze)
    from scipy.stats import linregress
    for jj, idy in enumerate(u_dys):

        if n_dys[jj] > 1:
            pss_dys = np.argwhere(s_dys == idy).flatten()

            avg_dy = lc_avg[pss_dys]
            zpt = np.median(avg_dy)

            # 4. Make predictions
            res = linregress(avg_dy, lc_tst[lc_tst.dys == int(idy)].mag.to_numpy())
            sys_model = res.slope * avg_dy  #+ res.intercept

            # plt.scatter(np.arange(n_dys[jj]), lc_cmp[lc_cmp.dys == int(idy)].mag, c='k')
            # plt.scatter(np.arange(n_dys[jj]), lc_tst[lc_tst.dys == int(idy)].mag, c='b')
            # plt.scatter(np.arange(n_dys[jj]), sys_model, c='r')
            # plt.show()

            cln_lc[lc_tst.dys == int(idy)] = lc_tst[lc_tst.dys == int(idy)].mag - sys_model
            _, _, og_std = scs(lc_cmp[lc_cmp.dys == int(idy)].mag)
            _, _, nw_std = scs(cln_lc[lc_tst.dys == int(idy)])
            print(og_std, nw_std, 'scale')

        else:
            pss_dys = np.argwhere(s_dys == idy).flatten()
            avg_dy = lc_avg[pss_dys]
            zpt = np.median(avg_dy)
            cln_lc[lc_tst.dys == int(idy)] = lc_tst[lc_tst.mag > 0].mag.median()
else:
    cln_lc = np.zeros(lc_sze)
    tr_stars = 1000
    lc_hold = np.zeros([lc_sze, tr_stars])
    star_list['dmag'] = np.abs(mean_mag - star_list['master_mag'])
    star_list['dist'] = np.sqrt((star_list.y - ycen) ** 2 + (star_list.x - xcen) ** 2)

    if gc_star == 0:
        trend_list = star_list[(star_list.dist > Configuration.APER_SIZE) &
                               (star_list.dmag < .5) &
                               (star_list.object_type == 'Star') &
                               (star_list.gc_star == 0)].copy().sort_values(by=['dmag'])[0:tr_stars].reset_index(
            drop=True)
    else:
        trend_list = star_list[(star_list.dist > Configuration.APER_SIZE) &
                               (star_list.dmag < .5) &
                               (star_list.object_type == 'Star') &
                               (star_list.gc_star == 1)].copy().sort_values(by=['dmag'])[0:tr_stars].reset_index(
            drop=True)

    for jj, rr in trend_list.iterrows():
        if rr.chip < 10:
            tr = pd.read_csv(dir + '/0' + str(rr.chip) + '/' +
                             Configuration.FIELD + "_" + str(rr.source_id) + ".lc",
                             sep=" ")
        else:
            tr = pd.read_csv(dir + '/' + str(rr.chip) + '/' +
                             Configuration.FIELD + "_" + str(rr.source_id) + ".lc",
                             sep=" ")

        lc_hold[:, jj] = tr.mag.to_numpy() - tr[tr.mag > 0].mag.median()
    lc_hold[lc_hold == 0] = -10
    lc_avg = np.zeros(lc_sze)
    for iii in np.arange(lc_sze):
        valys = lc_hold[iii,:]
        _, off, _ = scs(valys[valys>-10], sigma=2)
        lc_avg[iii] = off
    # lc_avg = np.average(lc_hold, axis=1)

    for zz, idy in enumerate(u_dys):

        if n_dys[zz] > 1:
            sys_model = lc_avg[lc_tst.dys == int(idy)]
            #plt.scatter(np.arange(n_dys[zz]), lc_cmp[lc_cmp.dys == int(idy)].mag, c='k')
            #plt.scatter(np.arange(n_dys[zz]), lc_tst[lc_tst.dys == int(idy)].mag - sys_model, c='orange')
            # plt.scatter(np.arange(n_dys[zz]), sys_model + lc_tst[lc_tst.dys == int(idy)].mag.median(), c='r')
            #plt.show()

            cln_lc[lc_tst.dys == int(idy)] = lc_tst[lc_tst.dys == int(idy)].mag - sys_model
            _, _, og_std = scs(lc_cmp[lc_cmp.dys == int(idy)].mag)
            _, _, nw_std = scs(cln_lc[lc_tst.dys == int(idy)])
            print(og_std, nw_std, 'median')
        else:
            cln_lc[lc_tst.dys == int(idy)] = lc_tst[lc_tst.dys == int(idy)].mag

plt.scatter(lc_tst.jd, cln_lc)
plt.scatter(lc_cmp.jd, lc_cmp.mag)
plt.show()