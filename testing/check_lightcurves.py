import pandas as pd
import matplotlib
import logging
matplotlib.set_loglevel(level = 'warning')
matplotlib.use("TkAgg")
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
import matplotlib.pyplot as plt
from config import Configuration
import numpy as np
from libraries.utils import Utils
from astropy.stats import sigma_clipped_stats as scs
from astropy.stats import sigma_clip
from twirl.geometry import sparsify

# pull in the star list for the photometry
# star_list = pd.read_csv(Configuration.MASTER_DIRECTORY + Configuration.FIELD + '_star_list.txt',
#                            delimiter=' ',
#                            header=0)
# star_list['chip'] = 1
# # add "chip" to the star_list
# kk = 1
# for idx in range(0, 10560, 1320):
#        for idy in range(0, 10560, 5280):
#             star_list['chip'] = np.where((star_list.xcen > idx) & (star_list.xcen < idx + 1320) &
#                                         (star_list.ycen > idy) & (star_list.ycen < idy + 5280),
#                                          kk, star_list.chip)
#             kk = kk + 1
# star_list['bd_star'] = np.where((star_list['xcen'] > 4300) & (star_list['xcen'] < 9300) &
#                                 (star_list['ycen'] > 3600) & (star_list['ycen'] < 8200), 1, 0)
# star_list = star_list[star_list.bd_star == 0].copy()
#
# idxs = star_list.index.values.tolist()
# star_list.to_csv('star_list.csv', sep=' ')
#
# # get the flux files to read in
# files, dates = Utils.get_all_files_per_field(Configuration.FLUX_DIRECTORY,
#                                              Configuration.FIELD,
#                                              'flux',
#                                              '.flux')
# nfiles = len(files)
# num_rrows = len(star_list)
# # make the holders for the light curves
# jd = np.zeros(nfiles)
# mag = np.zeros((num_rrows, nfiles))
# err = np.zeros((num_rrows, nfiles))
# zpt = np.zeros((num_rrows, nfiles))
# bkg = np.zeros((num_rrows, nfiles))
# sky = np.zeros((num_rrows, nfiles))
# for idy, file in enumerate(files):
#     # read in the data frame with the flux information
#     img_flux = pd.read_csv(file, header=0, low_memory=False)
#     jd[idy] = img_flux.loc[0, 'jd']
#     img_flux = img_flux.loc[idxs]
#
#     # set the data to the numpy array
#     mag[:, idy] = img_flux['mag'].to_numpy()
#     err[:, idy] = img_flux['mag_er'].to_numpy()
#     zpt[:, idy] = img_flux['zpt'].to_numpy()
#     bkg[:, idy] = img_flux['bkg'].to_numpy()
#     sky[:, idy] = img_flux['sky'].to_numpy()
#
#     if (idy % 100 == 0) & (idy > 0):
#          Utils.log("100 flux files read. " + str(nfiles - idy - 1) + ' files remain.', "info")
#
# mag = pd.DataFrame(data=mag).to_csv('mag1.csv')
# err = pd.DataFrame(data=err).to_csv('err1.csv')
# zpt = pd.DataFrame(data=zpt).to_csv('zpt1.csv')
# bkg = pd.DataFrame(data=bkg).to_csv('bkg1.csv')
# sky = pd.DataFrame(data=sky).to_csv('sky1.csv')

star_list = pd.read_csv('star_list.csv', sep=' ')

mag1 = pd.read_csv('mag1.csv', index_col=0)
err1 = pd.read_csv('err1.csv', index_col=0)
zpt1 = pd.read_csv('zpt1.csv', index_col=0)
bkg1 = pd.read_csv('bkg1.csv', index_col=0)
sky1 = pd.read_csv('sky1.csv', index_col=0)

jd = pd.read_csv('time.csv', index_col=0).rename(columns={'0': 'jd'}).reset_index().rename(columns={'index': 'idx'})
jd['bd'] = 0
jd = jd.sort_values(by='jd').reset_index(drop=True)
img_list = pd.read_csv('/Users/yuw816/Data/toros/commissioning/clean/image_list.csv', sep=',')
jd['bd'] = np.where(img_list['Bad'] == 1, 1, jd['bd'])
jd = jd.sort_values(by='idx')
jd['bd'] = np.where(mag1.median(axis=0).to_numpy() < 0, 1, jd['bd'])
bd_img = jd[jd.bd == 1]['idx'].to_numpy().astype(str)

mag1 = mag1.drop(bd_img, axis = 1)
zpt1 = zpt1.drop(bd_img, axis = 1)
err1 = err1.drop(bd_img, axis = 1)

jd = jd[jd.bd == 0].reset_index(drop=True)
jd = jd['jd'].to_numpy()

snr0 = np.zeros(len(star_list))
snr1 = np.zeros(len(star_list))
zz = 0

for idy, row in star_list.loc[4:].iterrows():
    raw1 = mag1.loc[idy].to_numpy()

    zp = zpt1.loc[idy].to_numpy()

    # get the difference in magnitude
    dmag = np.abs(row.master_mag - star_list.master_mag.to_numpy())
    dd = np.sqrt((row.xcen - star_list.xcen.to_numpy())**2 +
                 (row.ycen - star_list.ycen.to_numpy())**2)
    chp = star_list.chip.to_numpy()
    chp_st = row.chip

    # only get nearby stars of similar magnitude
    vv = np.argwhere((dmag < 2) & (dd > 0) & (chp == chp_st)).reshape(-1)
    trnd1 = np.zeros(len(raw1))

    for idz, cl in enumerate(mag1.columns.tolist()):
        _, trnd1[idz], _ = scs(mag1.loc[vv, cl].to_numpy() - star_list.loc[vv].master_mag.to_numpy(), sigma = 3)

    ph = (jd - np.min(jd)) / 0.37143 % 1
    plt.scatter(ph[(raw1 > 0)], raw1[(raw1 > 0)] - trnd1[(raw1 > 0)], c='k', marker='.')

    plt.gca().invert_yaxis()
    plt.title('P=0.59483867')
    plt.show()

    if (zz % 1500 == 0) & (zz > 0):
        plt.scatter(star_list[snr1 > 0].master_mag, snr1[snr1 > 0], marker='.', c='r', alpha=0.5)
        plt.scatter(star_list[snr0 > 0].master_mag, snr0[snr0 > 0], marker='.', c='k', alpha=0.5)
        plt.yscale('log')
        plt.show()
    zz = zz + 1