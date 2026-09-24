# first get the zeropoint determination from the flux measurements
flux_files, flux_dates = Utils.get_all_files_per_field(Configuration.FLUX_DIRECTORY,
                                                       Configuration.FIELD,
                                                       'flux',
                                                       '.flux')

# set up the zpt holder
zpt_offset = np.zeros(len(flux_files))
jd_offset = np.zeros(len(flux_files))
Utils.log("Pulling zeropoint offset from flux files...", "info")
for fidx, ffile in enumerate(flux_files):
    flux_df = pd.read_csv(ffile, nrows=1, header=0)
    zpt_offset[fidx] = flux_df['zpt'].values
    jd_offset[fidx] = flux_df['jd'].values

    if fidx % 50 == 0:
        Utils.log("..." + str(int(np.around(fidx / len(flux_files) * 100, decimals=0))) + "% complete.",
                  "info")

# sort to the correct time
time_srt = np.argsort(jd_offset)
zpt_offset = zpt_offset[time_srt]

# open the error file for writing
f = open(Configuration.LIGHTCURVE_FIELD_DIRECTORY + Configuration.FIELD + "_errors.txt", "w")
header = 'name mag rms erms orms x y chip object_type\n'
f.write(header)

for idx, row in star_list.tail(6000).iterrows():
    if idx % 1000 == 0:
        Utils.log("Working to detrend the next 1000 light curves. " +
                  str(len(star_list) - idx - 1) + " light curves remain.", "info")

    # read in the light curve
    if row.chip < 10:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                         '0' + str(row.chip) + '/' + Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                         sep=' ')
    else:
        lc = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                         str(row.chip) + '/' + Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                         sep=' ')

    # determine the offset in magnitude and distance from the target star
    star_list['dmag'] = np.abs(row.master_mag - star_list['master_mag'])
    star_list['dist'] = np.sqrt((star_list.y - row.y) ** 2 + (star_list.x - row.x) ** 2)

    # grab the list of trends stars based on whether or not the star is in 47-Tuc
    if row.gc_star == 0:
        trend_list = star_list[(star_list.gc_star == 0) &
                               (star_list.dist > Configuration.APER_SIZE) &
                               (star_list.object_type == 'Star')].copy().sort_values(by='dmag')[
                     0:trend_stars].reset_index(drop=True)
    else:
        trend_list = star_list[(star_list.gc_star == 1) &
                               (star_list.dist > Configuration.APER_SIZE) &
                               (star_list.object_type == 'Star')].copy().sort_values(by='dmag')[
                     0:trend_stars].reset_index(drop=True)

    # set up the empty collection vectors
    cols = {}
    col_nme = []
    mgs = np.zeros(len(trend_list))

    kk = 0  # initialize the star names
    for idy, rw in trend_list.iterrows():
        # read in the trend light curves
        if rw.chip < 10:
            tr = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                             '0' + str(rw.chip) + '/' + Configuration.FIELD + '_' + str(rw.source_id) + '.lc',
                             sep=' ')
        else:
            tr = pd.read_csv(Configuration.LIGHTCURVE_FIELD_RAW_DIRECTORY +
                             str(rw.chip) + '/' + Configuration.FIELD + '_' + str(rw.source_id) + '.lc',
                             sep=' ')

        # only accept the star for detrending if it has non-zero values
        if len(tr[tr.mag > 0]) > 0:
            # subtract the median value from the light curve
            cols['mag_' + str(kk)] = tr.mag.to_numpy() - tr[tr.mag > 0].mag.median()
            # grab the value you subtracted for safe keeping
            mgs[idy] = tr[tr.mag > 0].mag.median()
            # append the column name list to make the data frame
            col_nme.append('mag_' + str(kk))
            # update kk
            kk = kk + 1
        del tr
    del trend_list

    # make the trend_df
    trend_df = pd.DataFrame(cols, columns=col_nme)

    # set up the trend vector
    lc['trd'] = np.zeros(len(lc))
    lc['zpt'] = zpt_offset

    # loop through each day finding the appropriate offset
    for ii in range(len(lc)):
        # get the offset for the specific day
        offsets = trend_df.loc[ii].to_numpy()

        # initialize the holding lists
        mg = []
        off = []

        # get the offset list in X.X mag chunks to interpolate around outliers
        for jj in np.arange(np.min(mgs), np.max(mgs), 0.1):
            # ignore any empty space
            if len(offsets[(mgs >= jj) & (mgs < jj + 0.1) & (offsets > -20)]) > 0:
                # get the current magnitude bin
                mg.append(jj + 0.05)
                # get the median offset with 2.5 sigma clipping
                _, mg_mdn, _ = scs(offsets[(mgs >= jj) & (mgs < jj + 0.1) & (offsets > -20)], sigma=2.5)
                # the value shouldn't be nan, but if it is, then just use the median of the whole day
                if np.isnan(mg_mdn):
                    off.append(np.median(offsets[(mgs >= jj) & (mgs < jj + 0.1) & (offsets > -20)]))
                else:
                    off.append(mg_mdn)
        try:
            # get the trend value for this observations
            trd = np.interp(lc.mag[ii], mg, off)

            # if it is nan, then just use the median for the full day
            if np.isnan(trd):
                lc.loc[ii, 'trd'] = np.nanmedian(off)
            else:
                lc.loc[ii, 'trd'] = trd
        except:
            lc.loc[ii, 'trd'] = -9.9999
    del trend_df

    lc = lc.rename(columns={'mag': 'raw'})
    lc['mag'] = lc['raw'] - lc['trd']

    # rearrange the light curve information for better outputs
    lc = lc[['jd', 'mag', 'err', 'raw', 'trd', 'zpt', 'sky', 'bkg', 'x', 'y', 'nstars', 'airmass']]

    # update bad data
    lc.raw = np.where(lc.raw < 0, -9.9999, lc.raw)
    lc.mag = np.where(lc.raw < 0, -9.9999, lc.mag)

    # calculate statistics for the error analysis
    mag, _, full_rms = scs(lc[(lc.mag > 0) & (lc.err > 0)].mag, sigma=2.5)
    lc['dys'] = lc.jd.to_numpy().astype('int')

    rms_vals = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'std'}).to_numpy().flatten()
    num_obs = lc[(lc.mag > 0) & (lc.err > 0)].groupby('dys').agg({'mag': 'count'}).to_numpy().flatten()

    erms = lc[(lc.mag > 0) & (lc.err > 0)].err.mean()
    try:
        rms = np.median(rms_vals[num_obs >= 6])
    except:
        rms = full_rms

    # output the statistics
    line = (Configuration.FIELD + "_" + str(row.source_id) + ".lc" + " " +
            str(np.around(row.master_mag, decimals=4)) + " " +
            str(np.around(rms, decimals=4)) + " " +
            str(np.around(erms, decimals=4)) + " " +
            str(np.around(full_rms, decimals=4)) + " " +
            str(np.around(row.xcen, decimals=2)) + " " +
            str(np.around(row.ycen, decimals=2)) + " " +
            str(int(row.chip)) + " " +
            str(row.object_type) + "\n")
    f.write(line)

    lc = lc.drop(columns=['dys'])

    # update print formats
    lc.mag = lc.mag.map(lambda x: '%0.4f' % x)
    lc.raw = lc.raw.map(lambda x: '%0.4f' % x)
    lc.err = lc.err.map(lambda x: '%0.4f' % x)
    lc.trd = lc.trd.map(lambda x: '%0.4f' % x)
    lc.zpt = lc.zpt.map(lambda x: '%0.4f' % x)
    lc.x = lc.x.map(lambda x: '%d' % x)
    lc.y = lc.y.map(lambda x: '%d' % x)
    lc.nstars = lc.nstars.map(lambda x: '%d' % x)
    lc.airmass = lc.airmass.map(lambda x: '%0.3f' % x)

    # write out lc
    if row.chip < 10:
        lc.to_csv(Configuration.LIGHTCURVE_FIELD_DETREND_DIRECTORY + '/0' + str(row.chip) +
                  '/' + Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                  sep=" ", header=True, index=False)
    else:
        lc.to_csv(Configuration.LIGHTCURVE_FIELD_DETREND_DIRECTORY + '/' + str(row.chip) +
                  '/' + Configuration.FIELD + '_' + str(row.source_id) + '.lc',
                  sep=" ", header=True, index=False)
    del lc

# now close the file
f.close()

return

ls = LombScargle(lc[~ok_data.mask].jd.to_numpy(),
                 lc[~ok_data.mask].mag.to_numpy(),
                 dy=lc[~ok_data.mask].err.to_numpy())

frequency, power = ls.autopower()

shfl_mags = lc[~ok_data.mask].mag.to_numpy()

mx_power = np.zeros(1000)
for ii in range(0, 1000):
    ss = np.random.choice(shfl_mags, len(shfl_mags), replace=True)
    ss_f, ss_p = LombScargle(lc[~ok_data.mask].jd.to_numpy(),
                             ss,
                             dy=lc[~ok_data.mask].err.to_numpy()).autopower()
    mx_power[ii] = np.max(ss_p)

pwr5 = power[np.flip(np.argsort(power))][0:5]
prd5 = 1. / frequency[np.flip(np.argsort(power))][0:5]
fap5 = np.zeros(5)
for ii in range(0, 5):
    fap5[ii] = len(mx_power[mx_power >= pwr5[ii]])

# get J stet
wk = 1.0  # Weighting Factor

MeanMag = lc[~ok_data.mask].mag.mean()

mg = lc[~ok_data.mask].mag.to_numpy()
MeanMag = np.mean(mg)
er = lc[~ok_data.mask].err.to_numpy()
nms = len(lc[~ok_data.mask].jd.to_numpy())

Jt = np.arange(nms) * 0.0
Jb = np.arange(nms) * 0.0
Kt = np.arange(nms) * 0.0
Kb = np.arange(nms) * 0.0

for i in range(0, nms - 2, 2):

    Sigi = (mg[i] - MeanMag) / (er[i]) * (np.sqrt(nms / (nms - 1)))
    Sigj = (mg[i + 1] - MeanMag) / (er[i + 1]) * (np.sqrt(nms / (nms - 1)))

    Pk = Sigi * Sigj  # pg 853 Stetson 1996 Eq 2 Kinemuchi
    if Pk > 0.0:
        sgnPk = 1.0
    if Pk == 0.0:
        sgnPk = 0.0
    if Pk < 0.0:
        sgnPk = -1.0

    Jt[i] = wk * sgnPk * (np.sqrt(abs(Pk)))  # Kinemuchi eq.1 (Numerator)
    Jb[i] = (wk)  # Kinemuchi eq.1 (Denominator)
    Kt[i] = abs(Sigi)  # Kinemuchi eq.5 (Numerator)
    Kb[i] = abs(Sigi ** (2.0))  # Kinemuchi eq.5 (Denominator)

jstet = sum(Jt) / sum(Jb)  # Eq 1
kstet = ((1.0 / nms) * sum(Kt)) / (np.sqrt((1.0 / nms) * sum(Kb)))  # Eq 5
lstet = jstet * kstet / (0.7908)

dys = np.unique(lc[~ok_data.mask].jd.to_numpy().astype(int))
jstet_dys = np.zeros(len(dys))
lstet_dys = np.zeros(len(dys))
kstet_dys = np.zeros(len(dys))

for jj in range(0, len(dys)):
    # get J stet
    wk = 1.0  # Weighting Factor

    mg = lc[(~ok_data.mask) & (lc.jd.to_numpy().astype(int) == dys[jj])].mag.to_numpy()
    MeanMag = np.mean(mg)
    er = lc[(~ok_data.mask) & (lc.jd.to_numpy().astype(int) == dys[jj])].err.to_numpy()
    nms = len(lc[(~ok_data.mask) & (lc.jd.to_numpy().astype(int) == dys[jj])].jd.to_numpy())

    Jt = np.arange(nms) * 0.0
    Jb = np.arange(nms) * 0.0
    Kt = np.arange(nms) * 0.0
    Kb = np.arange(nms) * 0.0

    for i in range(0, nms - 2, 2):

        Sigi = (mg[i] - MeanMag) / (er[i]) * (np.sqrt(nms / (nms - 1)))
        Sigj = (mg[i + 1] - MeanMag) / (er[i + 1]) * (np.sqrt(nms / (nms - 1)))

        Pk = Sigi * Sigj  # pg 853 Stetson 1996 Eq 2 Kinemuchi
        if Pk > 0.0:
            sgnPk = 1.0
        if Pk == 0.0:
            sgnPk = 0.0
        if Pk < 0.0:
            sgnPk = -1.0

        Jt[i] = wk * sgnPk * (np.sqrt(abs(Pk)))  # Kinemuchi eq.1 (Numerator)
        Jb[i] = (wk)  # Kinemuchi eq.1 (Denominator)
        Kt[i] = abs(Sigi)  # Kinemuchi eq.5 (Numerator)
        Kb[i] = abs(Sigi ** (2.0))  # Kinemuchi eq.5 (Denominator)

    jstet_dys[jj] = sum(Jt) / sum(Jb)  # Eq 1
    kstet_dys[jj] = ((1.0 / nms) * sum(Kt)) / (np.sqrt((1.0 / nms) * sum(Kb)))  # Eq 5
    lstet_dys[jj] = jstet * kstet / (0.7908)

Utils.log("star clock", "info")
passer = 0
for idz, col in enumerate(col_nme[spr > 0.8]):

    yy = df[dd][col].to_numpy()

    reg = LinearRegression().fit(xx, yy).predict(xx)
    cur_std = df[dd].sub(reg, axis=0).std().sum()
    if cur_std < min_std:
        min_std = cur_std
        trd = reg
        passer = 1

Utils.log("end clock " + str(passer), "info")

try:
    lc.loc[dd, 'zpt'] = trd
except:
    print('hold')

# for idy, col in enumerate(col_nme):
# plt.plot(lc[dd].jd, df[dd][col],  c='k')
# plt.scatter(lc[dd].jd, lc[dd].mag - lc[dd].mag[0], c='r')
# plt.show()

scl_data = np.zeros((100, len(lc.jd[dd])))

sp = np.zeros(100)

for idy, col in enumerate(col_nme):
    sp[idy], _ = spearmanr(cols[col][dd], lc.mag[dd].to_numpy())

    if (sp[idy] >= 0):
        x0 = np.array([np.min(cols[col][dd]), np.max(cols[col][dd])])
    else:
        x0 = np.array([np.max(cols[col][dd]), np.min(cols[col][dd])])

    scl_data[idy, :] = ((cols[col][dd] - x0[0]) * (np.max(lc.mag[dd]) - np.min(lc.mag[dd]))) / (x0[1] - x0[0])

wmn_scl = np.average(scl_data[~np.isnan(sp) & (sp != 0)],
                     weights=np.abs(sp[~np.isnan(sp) & (sp != 0)]),
                     axis=0)


def scale_mags(x):
    scaled_data = ((wmn_scl - x[0]) * (np.max(lc.mag[dd]) - np.min(lc.mag[dd]))) / (x[1] - x[0])
    _, _, sc_std = scs(lc.mag[dd] - scaled_data, sigma=2)
    return sc_std  # np.std(lc.mag[dd] - scaled_data)


x0 = np.array([np.min(wmn_scl), np.max(wmn_scl)])
res = minimize(scale_mags, x0)
fin_scl = ((wmn_scl - res.x[0]) * (np.max(lc.mag[dd]) - np.min(lc.mag[dd]))) / (res.x[1] - res.x[0])
# lc.loc[dd, 'zpt'] = fin_scl

# Define your custom handler here.
# @gcn.include_notice_types(
#    gcn.notice_types.LVC_EARLY_WARNING,
#    gcn.notice_types.LVC_PRELIMINARY,
#    gcn.notice_types.LVC_INITIAL,
#    gcn.notice_types.LVC_UPDATE,
#    gcn.notice_types.LVC_RETRACTION)

if Configuration.TRANSIENT_LC == 'Y':
    trans = pd.Series(index=star_list.columns.to_list())
    master, master_head = fits.getdata(
        Configuration.MASTER_DIRECTORY + Configuration.FIELD + "_master.fits",
        header=True)
    w = WCS(master_head)
    ra = Configuration.TRANSIENT_RA
    dec = Configuration.TRANSIENT_DEC

    # convert to x, y
    x, y = w.all_world2pix(ra, dec, 0)

    # add the x/y to the star data frame
    trans['x'] = x
    trans['y'] = y

    # centroid on the master frame, but make sure it doesn't return an invalid result
    xcen, ycen = centroid_sources(master, trans.x.item(), trans.y.item(), box_size=5)
    if np.isnan(xcen) | np.isnan(ycen):
        trans['xcen'] = x
        trans['ycen'] = y
    else:
        trans['xcen'] = xcen[0]
        trans['ycen'] = ycen[0]

    # fill in the remainder of the data
    trans['star_id'] = Configuration.TRANSIENT_NAME
    trans['toros_field_id'] = Configuration.FIELD
    trans['source_id'] = '-999'
    trans['ra'] = Configuration.TRANSIENT_RA
    trans['dec'] = Configuration.TRANSIENT_DEC

    positions = np.transpose((trans['x'], trans['y']))

    aperture = CircularAperture(positions, r=Configuration.APER_SIZE)
    aperture_annulus = CircularAnnulus(positions,
                                       r_in=Configuration.ANNULI_INNER,
                                       r_out=Configuration.ANNULI_OUTER)
    apers = [aperture, aperture_annulus]
    phot_table = aperture_photometry(master, apers, method='exact')
    sky = np.median(master)
    flux = np.array(phot_table['aperture_sum_0'] - (sky * aperture.area))
    trans['master_flux'] = np.around(flux, decimals=6)
    star_error = np.sqrt(np.abs(phot_table['aperture_sum_0']))
    flux_er = np.array(np.sqrt(star_error ** 2))
    trans['master_flux_er'] = np.around(flux_er, decimals=6)

    # convert to magnitude
    trans['master_mag'] = np.around(25. - 2.5 * np.log10(flux), decimals=6)
    trans['master_mag_er'] = np.around((np.log(10.) / 2.5) * (flux_er / flux), decimals=6)

    star_list.loc[len(star_list)] = trans

def handler(payload, root):

    Utils.log("New " + root.attrib['role'] + ' event detected!', "info")
    if root.attrib['role'] == 'test':
        Utils.log("Test event found. No need to alert team.", "info")
        #return

    # Read all of the VOEvent parameters from the "What" section.
    params = {elem.attrib['name']: elem.attrib['value']
              for elem in root.iterfind('.//Param')}
    alert_type = params['AlertType']
    event_name = params['GraceID']

    # make sure the alert is not a retraciton
    if params['AlertType'] == 'Retraction':
        Utils.log("Alert for " + event_name + " was retracted. Deleting sky map files.", "info")
        os.system('rm ' + Configuration.ANALYSIS_DIRECTORY + event_name + '_SkyMap_toros_fields.txt')

        Utils.log("Alerting team of retraction.", "info")
        Alerts.alert_toros_team(alert_type=alert_type, event_name=event_name)
        return

    Utils.log("Alerting TOROS team.", "info")

    # alert the TOROS team
    Alerts.alert_toros_team(alert_type=root.attrib['role'],  event_name=event_name)

    # send in a request to NED to get the galaxy list
    ned_result = Alerts.query_ned(event_name)

    if ned_result is None:
        Utils.log("No NED galaxy priority yet. Waiting to re-query and extracting TOROS fields from SkyMap "
                  "in the meantime.", "info")
        wait_iter = 0
        while wait_iter < Configuration.LISTEN_NED_WAIT:
            if 'skymap_fits' in params:
                # check to see if field list from skymap exists
                if os.path.isfile(Configuration.ANALYSIS_DIRECTORY +
                                  event_name + '_SkyMap_toros_fields.txt') is False:
                    # Read the HEALPix sky map and the FITS header.
                    skymap, header = hp.read_map(params['skymap_fits'], h=True)
                    header = dict(header)

                    # pull in the toros fields
                    toros_fields = Priority.toros_field_generator(Configuration.FIELD_SIZE)

                    # generate a ranked list of toros fields within the skymap
                    toros_fields_prio = Priority.sort_toros_fields_skymap(toros_fields, skymap)
                    toros_fields_prio.to_csv(Configuration.ANALYSIS_DIRECTORY +
                                             event_name +
                                             '_SkyMap_toros_fields.txt',
                                             sep=',')

                    Utils.log("TOROS field list generated based on SkyMap from GCN. "
                              "Waiting 5 minutes to re-query NED.", "info")
                    # increase the wait iterations
                    wait_iter += 1
                    time.sleep(300)

                    # requery NED
                    ned_result = Alerts.query_ned(event_name)

                    if ned_result is None:
                        Utils.log("NED queried again after " + str(wait_iter * 5) +
                                  " minutes. Still no priority list.", "info")
                    else:
                        Utils.log("NED result obtained. Identifying necessary TOROS fields.", "info")
                        ligo_fields = Priority.return_toros_fields(ned_result, toros_fields)
                        ligo_fields.to_csv(Configuration.ANALYSIS_DIRECTORY +
                                           event_name +
                                           '_NED_toros_fields.txt',
                                           sep=',')
                        return ned_result
                else:
                    Utils.log("TOROS field list already generated from SkyMap. Skipping for now and "
                              "waiting 5 minutes to re-query NED.", "info")
                    wait_iter += 1
                    time.sleep(300)
                    ned_result = Alerts.query_ned(event_name)

                    if ned_result is None:
                        Utils.log("NED queried again after " + str(wait_iter * 5) +
                                  " minutes. Still no priority list.", "info")
                    else:
                        Utils.log("NED result obtained. Identifying necessary TOROS fields.", "info")
                        toros_fields = Priority.toros_field_generator(Configuration.FIELD_SIZE)
                        ligo_fields = Priority.return_toros_fields(ned_result, toros_fields)
                        ligo_fields.to_csv(Configuration.ANALYSIS_DIRECTORY +
                                           event_name + '_NED_toros_fields.txt', sep=',')
                        return
            else:
                Utils.log("SkyMap not found. Check manually.", "info")
                return

        return
    else:
        Utils.log("NED result obtained. Identifying necessary TOROS fields.", "info")
        # pull in the toros fields
        toros_fields = Priority.toros_field_generator(Configuration.FIELD_SIZE)
        ligo_fields = Priority.return_toros_fields(ned_result, toros_fields)

        ligo_fields.to_csv(Configuration.ANALYSIS_DIRECTORY + event_name + '_NED_toros_fields.txt', sep=',')

        return

    Utils.log("Maximum iterations for NED reached.", "info")
    return

# Listen for VOEvents until killed with Control-C.
gcn.listen(handler=handler)

from libraries.utils import Utils
from config import Configuration
import pandas as pd
import numpy as np
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt

# do necessary prep work such as making output directories
files = Utils.get_file_list(Configuration.LIGHTCURVE_DIRECTORY, '.lc')

nfiles = len(files)
mags = np.zeros(nfiles)
errs = np.zeros(nfiles)

for idx, file in enumerate(files):
    lc = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY + '/' + file, header=0, sep=' ')
    mmag, dmag, smag = sigma_clipped_stats(lc.cln, sigma=2.5)
    mags[idx] = dmag
    errs[idx] = smag
    if idx == 105:
        plt.scatter(lc.jd - 2459587, lc.cln-0.88, c='k', marker='.')
        plt.ylabel('Instrumental Magnitude [+offset]', size=15)
        plt.xlabel('JD - 2459587 [days]', size=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim([16.9, 16.5])
        plt.xlim([0.6, 0.9])
        plt.show()

plt.scatter(mags-0.88, errs, c='k')
plt.yscale('log')
plt.ylabel('rms')
plt.xlabel('Instrumental Magnitude [+offset]')
plt.show()

nms = ['s1', 's2', 's3', 's4', 's5']
corr = pd.DataFrame(columns=nms)
for idx, file in enumerate(files):

    dmag = np.abs(mags-mags[idx])
    ntrs = len(np.argwhere((dmag < 0.1) & (dmag > 0)))
    lc = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY + '/' + file, header=0, sep=' ')
    trd = np.zeros((ntrs, len(lc)))
    idz = 0
    for idy in range(0, nfiles):
        if (dmag[idy] < 0.1) & (dmag[idy] > 0):
            ss = pd.read_csv(Configuration.LIGHTCURVE_DIRECTORY + '/' + files[idy], header=0, sep=' ')
            trd[idz, :] = ss.cln.to_numpy()
            idz += 1
    tr = np.median(trd, axis=0)

    if idx == 105:
        plt.scatter(lc.jd - 2459587, lc.cln-0.85, c='k', marker='.')
        plt.ylabel('Instrumental Magnitude [+offset]', size=15)
        plt.xlabel('JD - 2459587 [days]', size=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim([16.9, 16.5])
        plt.show()

    mmag, dmag, smag = sigma_clipped_stats(lc.mag - (tr-np.median(tr)), sigma=2.5)
    mags[idx] = dmag
    errs[idx] = smag

plt.scatter(mags-0.85, errs, c='k')
#plt.yscale('log')
plt.ylabel('rms')
plt.xlabel('Instrumental Magnitude [+offset]')
plt.show()
