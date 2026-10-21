import logging
from libraries.utils import Utils
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import pandas as pd
from config import Configuration
from astropy.stats import sigma_clip
import numpy as np

tv_zpt = 5.4


# get the varstats file
data_dir = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/lc_stats/"
varstats = pd.read_csv(data_dir + Configuration.FIELD + "_varstats.txt",
                       delimiter=' ', low_memory=False)
varstats['v'] = varstats.apply(lambda x: x.mag - tv_zpt, axis=1)
varstats = varstats[varstats.cat_source == 'toros'].copy().reset_index(drop=True)

# location of the light curves
in_lc = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/star_list/fin/"
out_lc = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/star_list/release/"
out_fig = "/Users/yuw816/Data/toros/commissioning/lc/FIELD_0e.001/star_list/figures/"

for idx, row in varstats.iterrows():
    if row.chip == 1:
        if row.star_id < 10:
            nme = '00000' + str(row.star_id)
        elif (row.star_id >= 10) & (row.star_id < 100):
            nme = '0000' + str(row.star_id)
        elif (row.star_id >= 100) & (row.star_id < 1000):
            nme = '000' + str(row.star_id)
        elif (row.star_id >= 1000) & (row.star_id < 10000):
            nme = '00' + str(row.star_id)
        elif (row.star_id >= 10000) & (row.star_id < 100000):
            nme = '0' + str(row.star_id)
        else:
            nme = str(row.star_id)

        if row.chip < 10:
            lc = pd.read_csv(in_lc +
                             '0' + str(row.chip) + "/" +
                             Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                             sep=" ")
        else:
            lc = pd.read_csv(in_lc +
                             str(row.chip) + "/" +
                             Configuration.FIELD + "_" + str(row.source_id) + ".lc",
                             sep=" ")

        lc['mag'] = lc['mag'] - tv_zpt
        lc['ph'] = (lc.jd - lc.jd.min()) / row.prd % 1

        plt_lc = lc[(lc.mag > 0) & (lc.err > 0)].copy().reset_index(drop=True)
        clipped_data = sigma_clip(plt_lc.mag.to_numpy())

        plt.figure(figsize=(9, 12))
        plt.subplot(2, 1, 1)
        plt.errorbar(plt_lc[~clipped_data.mask].jd - 2460580, plt_lc[~clipped_data.mask].mag,
                     yerr = plt_lc[~clipped_data.mask].err, c='k', fmt='none')
        plt.scatter(plt_lc[~clipped_data.mask].jd - 2460580, plt_lc[~clipped_data.mask].mag, c='k')
        plt.gca().invert_yaxis()
        plt.xlabel('JD - 2460580 [d]')
        plt.ylabel(r'$T_V$')
        plt.title('FIELD 0e.001 ' + nme)
        plt.subplot(2, 1, 2)

        plt.errorbar(plt_lc[~clipped_data.mask].ph, plt_lc[~clipped_data.mask].mag,
                     yerr = plt_lc[~clipped_data.mask].err, c='k', fmt='none')
        plt.scatter(plt_lc[~clipped_data.mask].ph, plt_lc[~clipped_data.mask].mag, c='k')
        plt.errorbar(plt_lc[~clipped_data.mask].ph + 1, plt_lc[~clipped_data.mask].mag,
                     yerr = plt_lc[~clipped_data.mask].err, c='k', fmt='none')
        plt.scatter(plt_lc[~clipped_data.mask].ph + 1, plt_lc[~clipped_data.mask].mag, c='k')
        plt.gca().invert_yaxis()
        plt.xlabel('Phase')
        plt.ylabel(r'$T_V$')
        plt.title('P=' + str(np.around(row.prd, decimals=6)) + 'd')
        # plt.show()

        lc = lc[['jd', 'mag', 'err', 'raw', 'trd']].copy().reset_index(drop=True)
        lc['jd'] = np.around(lc['jd'].to_numpy(), decimals=6)
        lc['mag'] = np.around(lc['mag'].to_numpy(), decimals=4)
        lc['err'] = np.around(lc['err'].to_numpy(), decimals=4)
        lc['raw'] = np.around(lc['raw'].to_numpy(), decimals=4)
        lc['trd'] = np.around(lc['trd'].to_numpy(), decimals=4)

        if row.chip < 10:
            lc.to_csv(out_lc + '0' + str(row.chip) + "/" + Configuration.FIELD + "_" + nme + ".lc",
                      float_format='%.4f', sep=',', index=False)
            plt.savefig(out_fig + '0' + str(row.chip) + "/" + Configuration.FIELD + "_" + nme + ".png",
                        dpi=200, bbox_inches='tight')
            plt.close()
        else:
            lc.to_csv(out_lc + str(row.chip) + "/" + Configuration.FIELD + "_" + nme + ".lc",
                      float_format='%.4f', sep=',', index=False)
            plt.savefig(out_fig + str(row.chip) + "/" + Configuration.FIELD + "_" + nme + ".png",
                        dpi=200, bbox_inches='tight')
            plt.close()

    if idx % 1000 == 0:
        Utils.log('Finished with 1000, moving onto the next. ' + str(len(varstats) - idx - 1) + '.', "info")
