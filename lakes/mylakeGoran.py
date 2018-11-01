import numpy as np
import pandas as pd
import h5py
import datetime
import os
import math
import sys


variables = ['clt', 'hurs', 'tas', 'rsds', 'ps', 'pr', 'sfcWind']
models = {1: ('ICHEC-EC-EARTH', 'r1i1p1_KNMI-RACMO22E_v1_day'),
          2: ('ICHEC-EC-EARTH', 'r3i1p1_DMI-HIRHAM5_v1_day'),
          3: ('MPI-M-MPI-ESM-LR', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day'),
		  4: ('MOHC-HadGEM2-ES', 'r1i1p1_SMHI-RCA4_v1_day'),
		  5: ('IPSL-IPSL-CM5A-MR', 'r1i1p1_IPSL-INERIS-WRF331F_v1_day')}
scenarios = {1: ('historical', 1971, 'historical', 1976),
             2: ('historical', 2001, 'rcp45', 2006),
             3: ('rcp45', 2031, 'rcp45', 2036),
             4: ('rcp45', 2061, 'rcp45', 2066),
             5: ('rcp45', 2091, 'rcp45', 2096),
             6: ('rcp85', 2031, 'rcp85', 2036),
             7: ('rcp85', 2061, 'rcp85', 2066),
             8: ('rcp85', 2091, 'rcp85', 2096)}

cordexfolder = 'MDN_FishNiche_2017/cordex'
inflowfolder = '../sweden_inflow_data'
outputfolder = '../output'

def mylakeinit(max_depth, area, outpath):
    """
        create a file of a lake initiated with a max_depth and area.
        Assumes to have a cone shaped bathymetry curve
        :param max_depth: maximum depth in metre
        :param area: area in metre^2
        :param outpath: filename where an init file of Mylake will be written
        :type max_depth: int
        :type area: int
        :type outpath: str
        :return: string to be written to an init file of MyLake
    """
    depth_resolution = 1 # metres. NOTE: don't change this unless you know what you are doing. Changing it here will not make mylake run with a higher depth resolution, it will only change the init data

    depth_levels = np.arange(0, max_depth, depth_resolution)
    if not max_depth in depth_levels:
        depth_levels = np.concatenate((depth_levels, np.array([max_depth])))
    areas = area * (depth_levels - max_depth) ** 2 / max_depth ** 2
    lines = ['\t'.join([('%.2f' % d), ('%.0f' % a)] + ['4'] + ['0'] * 5 + ['8000'] + ['0'] * 5 + ['8000'] + ['0'] * 15) # Z, Az and T, ...., DOC, .... DO, ...
             for d, a in zip(depth_levels, areas)]
    # lines[0] = lines[0] + '\t0\t0'  # snow and ice, plus 16 dummies
    firstlines = '''-999	"MyLake init"
Z (m)	Az (m2)	Tz (deg C)	Cz	Sz (kg/m3)	TPz (mg/m3)	DOPz (mg/m3)	Chlaz (mg/m3)	DOCz (mg/m3)	TPz_sed (mg/m3)	Chlaz_sed (mg/m3)	"Fvol_IM (m3/m3	 dry w.)"	Hice (m)	Hsnow (m)	DO	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy'''
    lines = [firstlines] + lines
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))


def mylakepar(longitude, latitude, outpath):
    """
    create file of Mylake with parameters. uses the Minesota area and BV parameters -> sets NaNs
    atten_coeff: m-1
    :param longitude: longitude coordinate of Mylake in degrees.
    :param latitude: latitude coordinate of Mylake in degrees
    :param outpath: filename where a file of Mylake parameters will be written
    :type longitude: int
    :type latitude: int
    :type outpath: str
    :return: string to be written to a file
    """
    out = " "
    if (os.path.isfile("mylake_param_template.txt")):
        with open ("mylake_param_template.txt", "r") as infile:
            out = infile.read() % (latitude, longitude)
    else:
        out = '''-999	"Mylake parameters"			
Parameter	Value	Min	Max	Unit
dz	1.0	0.5	2	m
Kz_ak	NaN	NaN	NaN	(-)
Kz_ak_ice	0.000898	NaN	NaN	(-)
Kz_N0	7.00E-05	NaN	NaN	s-2
C_shelter	NaN	NaN	NaN	(-)
latitude	%.5f	NaN	NaN	dec.deg
longitude	%.5f	NaN	NaN	dec.deg
alb_melt_ice	0.6	NaN	NaN	(-)
alb_melt_snow	0.9	NaN	NaN	(-)
PAR_sat	3.00E-05	1.00E-05	1.00E-04	mol m-2 s-1
f_par	0.45	NaN	NaN	(-)
beta_chl	0.015	0.005	0.045	m2 mg-1
lamgbda_I	5	NaN	NaN	m-1
lambda_s	15	NaN	NaN	m-1
sed_sld	0.36	NaN	NaN	(m3/m3)
I_scV 	1.15	NaN	NaN	(-)
I_scT	0	NaN	NaN	deg C
I_scC	1	NaN	NaN	(-)
I_scS	1	1.1	1.9	(-)
I_scTP	1	0.4	0.8	(-)
I_scDOP	1	NaN	NaN	(-)
I_scChl	1	NaN	NaN	(-)
I_scDOC	1	NaN	NaN	(-)
swa_b0	2.5	NaN	NaN	m-1
swa_b1	1	0.8	1.3	m-1
S_res_epi	3.30E-07	7.30E-08	1.82E-06	m d-1 (dry mass)
S_res_hypo	3.30E-08	NaN	NaN	m d-1 (dry mass)
H_sed	0.3	NaN	NaN	m
Psat_Lang	2500	NaN	NaN	mg m-3
Fmax_Lang	8000	5000	10000	mg kg-1
Uz_Sz	0.3	0.1	1	m d-1
Uz_Chl	0.16	0.05	0.5	m d-1
Y_cp	1	NaN	NaN	(-)
m_twty	0.2	0.1	0.3	d-1
g_twty	1.5	1	1.5	d-1
k_sed_twty	2.00E-04	NaN	NaN	d-1
k_dop_twty	0	NaN	NaN	d-1
P_half	0.2	0.2	2	mg m-3
PAR_sat2	3.00E-05	NaN	NaN	mol m-2 s-1
beta_chl2	0.015	NaN	NaN	m2 mg-1
Uz_Chl2	0.16	NaN	NaN	m d-1
m_twty2	0.2	NaN	NaN	d-1
g_twty2	1.5	NaN	NaN	d-1
P_half2	0.2	NaN	NaN	mg m-3
oc_DOC	0.01	NaN	NaN	m2 mg-1
qy_DOC	0.1	NaN	NaN	mg mol-1
k_BOD	0.01	NaN	NaN	d-1
k_SOD	100	NaN	NaN	mg m-2
theta_BOD	1.047	NaN	NaN	(-)
theta_BOD_ice	1.13	NaN	NaN	(-)
theta_SOD	1	NaN	NaN	(-)
theta_SOD_ice	1	NaN	NaN	(-)
theta_T	4	NaN	NaN	deg.celcius
pH	5.2	NaN	NaN	(-)
I_scDIC	1	NaN	NaN	(-)
Mass_Ratio_C_Chl	100	NaN	NaN	(-)
SS_C	0.25	NaN NaN 57
density_org_H_nc	1.95	NaN NaN 58
density_inorg_H_nc	2.65	NaN NaN 59
I_scO	1	NaN NaN (-)
''' % (latitude, longitude)

    with open(outpath, 'w') as f:
        f.write(out)


def take5(pdict, dates, eh):
    """
    Create a dataFrame containing the values predicted of (clt,hurs,pr,ps,rsds,sfcWind,tas) for each dates
    :param pdict: dictionary of paths to HDF5 files (see pA and pB)
    :param dates: see datesA and datesB
    :param eh: ebhex number
    :type pdict: dict
    :type dates: pandas.DatetimeIndex
    :type eh: str
    :return: pandas.DataFrame
    """

    e = eh.lstrip('0x').lstrip('0')
    df = pd.DataFrame(dates, columns = ['date'])
    df['clt'] = h5py.File(pdict['clt'], mode='r')[e][:] * 0.01
    df['hurs'] = h5py.File(pdict['hurs'], mode='r')[e][:]
    df['pr'] = h5py.File(pdict['pr'], mode='r')[e][:] * (60 * 60 * 24)
    df['ps'] = h5py.File(pdict['ps'], mode='r')[e][:] * 0.01
    df['rsds'] = h5py.File(pdict['rsds'], mode='r')[e][:] * (60 * 60 * 24 * 1e-6)
    df['sfcWind'] = h5py.File(pdict['sfcWind'], mode='r')[e][:]
    df['tas'] = h5py.File(pdict['tas'], mode='r')[e][:] - 273.15
    return df

def inflow5(filename, dates, subId):
    """
    create a dataFrame containing the values of (Q,T,TP,DOP) for each dates
    :param filename: filename of the file containing inflow information
    :param dates: sum of datesA and dateB
    :param subId: Reference number
    :type filename: str
    :type dates: pandas.DatetimeIndex
    :type subId: int
    :return: pandas.DataFrame
    """
    dflow = pd.DataFrame(dates, columns = ['date'])
    dflow['Q'] = h5py.File(filename, mode='r')['%d/Q' % subId][:]
    dflow['T'] = h5py.File(filename, mode='r')['%d/T' % subId][:]
    dflow['TP'] = h5py.File(filename, mode='r')['%d/TP' % subId][:]
    dflow['DOP'] = h5py.File(filename, mode='r')['%d/DOP' % subId][:]
    return dflow

def mylakeinput(pA, pB, datesA, datesB, eh, subid, inflowfile, outpath):
    """
    create a file containing the informations relatively to Mylake
    :param pA: dictionary of paths to HDF5 files
    :param pB: dictionary of paths to HDF5 files
    :param datesA: pandas.date_range of time y1A to y2A
    :param datesB: pandas.date_range of time y1B to y2B
    :param eh: ebhex number
    :param subid: Reference number
    :param inflowfile: filename of the inflowfile
    :param outpath: filename where a file of Mylake input will be written
    :type pA: dict
    :type pB: dict
    :type datesA: pandas.date_range
    :type datesB: pandas.date_range
    :type eh: str
    :type subid: int
    :type inflowfile: str
    :type outpath: str
    :return: string to be written to a file
    """
    df = pd.concat([take5(pA, datesA, eh), take5(pB, datesB, eh)])
    ndays = len(datesA) + len(datesB)
    df.index = np.arange(ndays)

    dflow = inflow5(inflowfile, datesA + datesB, subid)

    repd = [datesA[0] + datetime.timedelta(d) for d in range(-(365 * 2), ndays)]
    mlyear = np.array([d.year for d in repd])
    mlmonth = np.array([d.month for d in repd])
    mlday = np.array([d.day for d in repd])
    mlndays = 365 + 365 + ndays
    repeati = list(range(365)) + list(range(365)) + list(range(ndays))
    spacer = np.repeat([0], repeats = ndays)[repeati].reshape((mlndays, 1))
    #stream_Q = np.repeat([2000], repeats = ndays)[repeati].reshape((mlndays, 1))
    #stream_T = np.repeat([10], repeats = ndays)[repeati].reshape((mlndays, 1))
    stream_O = np.repeat([8000], repeats = ndays)[repeati].reshape((mlndays, 1))
    stream_C = np.repeat([0.5], repeats = ndays)[repeati].reshape((mlndays, 1))
    #stream_TP = np.repeat([5], repeats = ndays)[repeati].reshape((mlndays, 1))
    #stream_DOP = np.repeat([1], repeats = ndays)[repeati].reshape((mlndays, 1))
    stream_SS = np.repeat([0.01], repeats = ndays)[repeati].reshape((mlndays, 1))
    stream_Chl = np.repeat([0.01], repeats = ndays)[repeati].reshape((mlndays, 1))
    stream_DOC = np.repeat([8000], repeats = ndays)[repeati].reshape((mlndays, 1))
    stream_DIC = np.repeat([20000], repeats = ndays)[repeati].reshape((mlndays, 1))
    temporarypath = '%s.temp' % outpath
    np.savetxt(temporarypath,
               np.concatenate((mlyear.reshape((mlndays, 1)),
                               mlmonth.reshape((mlndays, 1)),
                               mlday.reshape((mlndays, 1)),
                               df['rsds'][repeati].values.reshape((mlndays, 1)),
                               df['clt'][repeati].values.reshape((mlndays, 1)),
                               df['tas'][repeati].values.reshape((mlndays, 1)),
                               df['hurs'][repeati].values.reshape((mlndays, 1)),
                               df['ps'][repeati].values.reshape((mlndays, 1)),
                               #np.repeat([0], repeats = ndays)[repeati].reshape((mlndays, 1)),
                               df['sfcWind'][repeati].values.reshape((mlndays, 1)),
                               df['pr'][repeati].values.reshape((mlndays, 1)),
                               dflow['Q'][repeati].values.reshape((mlndays, 1)),
                               dflow['T'][repeati].values.reshape((mlndays, 1)),
                               stream_C, stream_SS,  # C, SS
                               dflow['TP'][repeati].values.reshape((mlndays, 1)),
                               dflow['DOP'][repeati].values.reshape((mlndays, 1)),
							   stream_Chl, stream_DOC,  # Chl, DOC
                               stream_DIC, stream_O, spacer, spacer,  # DIC, O, NO3, NH4
                               spacer, spacer, spacer, spacer,  # SO4, Fe, Ca, PH
                               spacer, spacer, spacer, spacer,  # CH4, Fe3, Al3, SiO4
                               spacer, spacer), axis = 1),  # SiO2, diatom
               fmt = ['%i', '%i', '%i',  # yy mm dd
                      '%.4g', '%.2f', '%.2f', '%i', '%i', '%.2f', '%.3f',  # rad, cloud, temp, hum, pres, wind, precip 
                      '%.3f', '%.3f', '%.3f', '%.3f',  # 
                      '%.3f', '%.3f', '%.3f', '%.3f',  # 
                      '%.3f', '%.3f', '%i', '%i',  # 
                      '%i', '%i', '%i', '%i',  # 
                      '%i', '%i', '%i', '%i',  # 
                      '%i', '%i'], #
               delimiter = '\t',
               header = 'mylake input\nYear	Month	Day	Global_rad (MJ/m2)	Cloud_cov (-)	Air_temp (deg C)	Relat_hum (%)	Air_press (hPa)	Wind_speed (m/s)	Precipitation (mm/day)	Inflow (m3/day)	Inflow_T (deg C)	Inflow_C	Inflow_S (kg/m3)	Inflow_TP (mg/m3)	Inflow_DOP (mg/m3)	Inflow_Chla (mg/m3)	Inflow_DOC (mg/m3)	DIC	DO	NO3	NH4	SO4	Fe2	Ca	pH	CH4	Fe3	Al3	SiO4	SiO2	diatom')
    with open(temporarypath) as f:
        with open(outpath, 'w') as g:
            g.write(f.read().replace('-99999999', 'NaN'))
    os.unlink(temporarypath)

def runlake(modelid, scenarioid, eh, subid, depth, area, longitude, latitude):
    """

    :param modelid: model used
    :param scenarioid: scenario used
    :param eh: ebhex number
    :param subid: Reference number
    :param depth: depth used for initiate Mylake (see mylakeinit())
    :param area: area used for initiate Mylake (see mylakeinit())
    :param longitude: longitude coordinate for Mylake (see mylakepar())
    :param latitude: latitude coordinate for Mylake (see mylakepar())
    :return:
    .. note:: see above lines for models and scenarios (dictionaries). eh: EBhex
    """
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2A = y1A + 4
    y2B = y1B + 4
    pA = {v: '%s/Lakes_%s_EUR-11_%s_%s_%s_%s0101-%s1231.h5' %
          (cordexfolder, v, m1, exA, m2, y1A, y2A) for v in variables}
    pB = {v: '%s/Lakes_%s_EUR-11_%s_%s_%s_%s0101-%s1231.h5' %
          (cordexfolder, v, m1, exB, m2, y1B, y2B) for v in variables}

    inflowfilename = '%s/sweden_inflow_data_20010101_20101231.h5' % inflowfolder    # This is hard coded for now

    datesA = pd.date_range(pd.datetime(y1A, 1, 1), pd.datetime(y2A, 12, 31), freq='d').tolist()
    datesB = pd.date_range(pd.datetime(y1B, 1, 1), pd.datetime(y2B, 12, 31), freq='d').tolist()

    eh = eh[2:] if eh[:2] == '0x' else eh
    while len(eh) < 6:
        eh = '0' + eh
    d1, d2, d3 = eh[:2], eh[:4], eh[:6]
    outdir = os.path.join(outputfolder, d1, d2, d3,
                          'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (
                              m1, exA, exB, m2, y1A, y2B))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    initp = os.path.join(outdir, '2017init')
    parp = os.path.join(outdir, '2017par')
    inputp = os.path.join(outdir, '2017input')
    if os.path.exists(os.path.join(outdir, '2017REDOCOMPLETE')):
        print('lake %s is already completed' % eh)
        ret = 0
    else:
        mylakeinit(depth, area, initp)
        mylakepar(longitude, latitude, parp)
        mylakeinput(pA, pB, datesA, datesB, eh, subid, inflowfilename, inputp)
        cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\');quit' % (
            initp, parp, inputp, y1A - 2, y2B, outdir)
        print(cmd)
        os.system(cmd)
        for f in [initp, parp, inputp]:
            os.system('bzip2 -f %s' % f)
        expectedfs = ['2017init.bz2', '2017par.bz2',
                      '2017input.bz2', 'Tzt.csv',
                      'O2zt.csv', 'Attn_zt.csv', 'Qst.csv','DOCzt.csv']
        flags = [os.path.exists(os.path.join(outdir, f)) for f in expectedfs]
        if all(flags):
            with open(os.path.join(outdir, '2017REDOCOMPLETE'), 'w') as f:
                f.write(datetime.datetime.now().isoformat())
        ret = 0 if all(flags) else 100
    return ret

if __name__ == '__main__':
    # model = 'EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_KNMI-RACMO22E_v1_day'
    # y1 = 1971
    # latitude = 59.97
    # longitude = 10.62
    # eh = '1cb62'
    # # eh = 'b516a3' ## a lake in the north
    # area = 1100000
    # depth = max(4.0, math.sqrt(area) / 100)
    # y2 = y1 + 4
    # y3 = y1 + 5
    # y4 = y1 + 9
    a = sys.argv


