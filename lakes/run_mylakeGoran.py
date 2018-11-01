"""
run_mylakeGoran.py
last modification: 2018-08-21 MC

Created input, init and parameters files for each lake giving with data coeffient calculated
Lauch matlab script 'mylakeGoran.mat' and verify if output files are created.

"""


import numpy as np
import pandas as pd
import h5py
import datetime
import os
import shutil
import bz2
import math
import sys


variables = ['clt', 'hurs', 'tas', 'rsds', 'ps', 'pr', 'sfcWind']
models = {1: ('ICHEC-EC-EARTH', 'r1i1p1_KNMI-RACMO22E_v1_day'),
          2: ('ICHEC-EC-EARTH', 'r3i1p1_DMI-HIRHAM5_v1_day'),
          3: ('MPI-M-MPI-ESM-LR', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day'),
          4: ('MOHC-HadGEM2-ES', 'r1i1p1_SMHI-RCA4_v1_day'),
          5: ('IPSL-IPSL-CM5A-MR', 'r1i1p1_IPSL-INERIS-WRF331F_v1_day'),
          6: ('CNRM-CERFACS-CNRM-CM5', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day')}
scenarios = {1: ('historical', 1971, 'historical', 1976),
             2: ('historical', 2001, 'rcp45', 2006),
             3: ('rcp45', 2031, 'rcp45', 2036),
             4: ('rcp45', 2061, 'rcp45', 2066),
             5: ('rcp45', 2091, 'rcp45', 2096),
             6: ('rcp85', 2031, 'rcp85', 2036),
             7: ('rcp85', 2061, 'rcp85', 2066),
             8: ('rcp85', 2091, 'rcp85', 2096)}

cordexfolder = 'G:\cordex' #5-24-2018 MC
inflowfolder = '../sweden_inflow_data' #5-9-2018 MC
outputfolder = '../output' #5-9-2018 MC


def mylakeinit(max_depth, area, outpath,I_scDOC=1):
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
    #5-7-2018 MC
    depth_resolution = 1  # metres. NOTE: don't change this unless you know what you are doing. Changing it here will
    #  not make mylake run with a higher depth resolution, it will only change the init data

    depth_levels = np.arange(0, max_depth, depth_resolution)
    if max_depth not in depth_levels:
        depth_levels = np.concatenate((depth_levels, np.array([max_depth])))
    areas = area * (depth_levels - max_depth) ** 2 / max_depth ** 2
    lines = [
        '\t'.join([('%.2f' % d), ('%.0f' % a)] + ['4'] + ['0'] * 5 + ['%s'%(2000*I_scDOC)] + ['0'] * 5 + ['12000'] + ['0'] * 15) #MC 06-01-2018 add I_scDOC and initial 8000 become 2000#MC 06-29-2018 12000
        # Z, Az and T, ...., DOC, .... DO, ...
        for d, a in zip(depth_levels, areas)]
    # lines[0] = lines[0] + '\t0\t0'  # snow and ice, plus 16 dummies
    firstlines = '''-999	"MyLake init"
Z (m)	Az (m2)	Tz (deg C)	Cz	Sz (kg/m3)	TPz (mg/m3)	DOPz (mg/m3)	Chlaz (mg/m3)	DOCz (mg/m3)	TPz_sed (mg/m3)	
Chlaz_sed (mg/m3)	"Fvol_IM (m3/m3	 dry w.)"	Hice (m)	Hsnow (m)	DO	dummy	dummy	dummy	dummy	dummy	
dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy'''
    lines = [firstlines] + lines
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))

def mylakepar(longitude, latitude, outpath,swa_b1=0.1,k_BOD=0.01,k_SOD=100,I_scDOC=1):
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
    #5-7-2018 MC

    if (os.path.isfile ( "LAE_para_all1.txt" )): #this file allows change of the four coefficients, if nothing is given, will uses initial values
        print('using file')
        with open ( "LAE_para_all1.txt", "r" ) as infile:
            out = infile.read () % (latitude, longitude, I_scDOC, swa_b1, k_BOD, k_SOD)

    else:
        out = '''-999	"Mylake parameters"			
    Parameter	Value	Min	Max	Unit
    dz	1.0	0.5	2	m
    Kz_ak	0.007	NaN	NaN	(-)
    Kz_ak_ice	0.003	NaN	NaN	(-)
    Kz_N0	7.00E-05	NaN	NaN	s-2
    C_shelter	NaN	NaN	NaN	(-)
    latitude	%.5f	NaN	NaN	dec.deg
    longitude	%.5f	NaN	NaN	dec.deg
    alb_melt_ice	0.6	NaN	NaN	(-)
    alb_melt_snow	0.9	NaN	NaN	(-)
    PAR_sat	3.00E-05	1.00E-05	1.00E-04	mol m-2 s-1
    f_par	0.89	NaN	NaN	(-)
    beta_chl	0.015	0.005	0.045	m2 mg-1
    lamgbda_I	5	NaN	NaN	m-1
    lambda_s	15	NaN	NaN	m-1
    sed_sld	0.36	NaN	NaN	(m3/m3)
    I_scV 	1.339	NaN	NaN	(-)
    I_scT	1.781	NaN	NaN	deg C
    I_scC	1	NaN	NaN	(-)
    I_scS	1	1.1	1.9	(-)
    I_scTP	1	0.4	0.8	(-)
    I_scDOP	1	NaN	NaN	(-)
    I_scChl	1	NaN	NaN	(-)
    I_scDOC	%s	NaN	NaN	(-)
    swa_b0	0.727	NaN	NaN	m-1
    swa_b1	%s	0.8	1.3	m-1
    S_res_epi	3.30E-07	7.30E-08	1.82E-06	m d-1 (dry mass)
    S_res_hypo	3.30E-08	NaN	NaN	m d-1 (dry mass)
    H_sed	0.03	NaN	NaN	m
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
    k_BOD	%s	NaN	NaN	d-1
    k_SOD	%s	NaN	NaN	mg m-2
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
    ''' % (latitude, longitude, I_scDOC, swa_b1, k_BOD, k_SOD)


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
    #5-7-2018 MC

    e = eh.lstrip('0x').lstrip('0')
    df = pd.DataFrame(dates, columns=['date'])
    try:
        df['clt'] = h5py.File(pdict['clt'], mode='r')[e][:] * 0.01
    except:
        df['clt'] = 0.65 #2018-08-02 MC Mean value found in literature
    try:
        df['hurs'] = h5py.File(pdict['hurs'], mode='r')[e][:]
    except:
        df['hurs'] = 50
    df['pr'] = h5py.File(pdict['pr'], mode='r')[e][:] * (60 * 60 * 24)
    df['ps'] = h5py.File(pdict['ps'], mode='r')[e][:] * 0.01
    df['rsds'] = h5py.File(pdict['rsds'], mode='r')[e][:] * (60 * 60 * 24 * 1e-6)
    df['sfcWind'] = h5py.File(pdict['sfcWind'], mode='r')[e][:]
    df['tas'] = h5py.File(pdict['tas'], mode='r')[e][:] - 273.15
    return df

def take5_missingdata(pdict, dates, eh): #5-24-2018 MC
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
    #5-7-2018 MC
    e = eh.lstrip('0x').lstrip('0')
    test = len(pdict)
    special = False
    if str(dates[-1].year) == '2099':
        df = pd.DataFrame ( index=range ( 1440 ), columns=['clt', 'hurs', 'pr', 'ps', 'rsds', 'sfcWind', 'tas'] )
        special = True
    else:
        df = pd.DataFrame(index=range(1800),columns=['clt','hurs','pr','ps','rsds','sfcWind','tas'])
    # 2018-08-02 MC add try to clt and hurs to compensate when the variables are missing
    try:
        df['clt'] = h5py.File(pdict['clt'], mode='r')[e][:] * 0.01
    except:
        df['clt'] = 0.65 #2018-08-02 MC Mean value found in literature
    try:
        df['hurs'] = h5py.File(pdict['hurs'], mode='r')[e][:]
    except:
        df['hurs'] = 50
    df['pr'] = h5py.File(pdict['pr'], mode='r')[e][:] * (60 * 60 * 24)
    df['ps'] = h5py.File(pdict['ps'], mode='r')[e][:] * 0.01
    df['rsds'] = h5py.File(pdict['rsds'], mode='r')[e][:] * (60 * 60 * 24 * 1e-6)
    df['sfcWind'] = h5py.File(pdict['sfcWind'], mode='r')[e][:]
    df['tas'] = h5py.File(pdict['tas'], mode='r')[e][:] - 273.15
    if len(dates) != len(df):
        if special:
            step = int((len(dates)-len(df))/4)
        else:
            step = int ( (len ( dates ) - len ( df )) / 5 )
        leapyear= int(str(dates[-1])[0:4])

        for i in dates:
            if str(i)[5:10]=='02-29':
                leapyear = int(str(i)[0:4])

        beforeleap = leapyear - int(str(dates[0])[0:4])
        row = -1
        time = beforeleap*365
        for i in np.arange((365/step)+row,time,(365/step)): #year/years before leap
            emptyrow = pd.DataFrame({'clt': np.nan,'hurs': np.nan,'pr': np.nan,'ps':np.nan,'rsds':np.nan,'sfcWind':np.nan,'tas':np.nan},index=[i])
            df = pd.concat ( [df.ix[:i - 1], emptyrow, df.ix[i:]] ).reset_index ( drop=True )
        row = row+time
        time = 366
        for i in np.arange((366/(step+1)+row),row+time+1,(366/(step+1))): # leap year
            emptyrow = pd.DataFrame (
                {'clt': np.nan, 'hurs': np.nan, 'pr': np.nan, 'ps': np.nan, 'rsds': np.nan, 'sfcWind': np.nan,
                 'tas': np.nan}, index=[i] )
            df = pd.concat ( [df.ix[:i - 1], emptyrow, df.ix[i:]] ).reset_index ( drop=True )
        row = row + 366
        time = (4-beforeleap)*365
        for i in np.arange((365/step)+row,row+time+1,(365/step)): #year/years after leap
            emptyrow = pd.DataFrame (
                {'clt': np.nan, 'hurs': np.nan, 'pr': np.nan, 'ps': np.nan, 'rsds': np.nan, 'sfcWind': np.nan,
                 'tas': np.nan}, index=[i] )
            df = pd.concat ( [df.ix[:i - 1], emptyrow, df.ix[i:]] ).reset_index ( drop=True )
    dfinal = pd.DataFrame ( dates, columns=['date'] )
    return pd.concat([dfinal,df],axis=1)

def nbrleapyears(start,end): #MC 2018-07-10
    """
    determine the number of leap years in the date range
    :param start: start year
    :param end: end year
    :return: number of leap year between the start and end years
    """
    nbryears=0
    while start <= end:
        if (start % 4 == 0 and start%100 !=0) or start%400 == 0:
            nbryears +=1
        start += 1
    return nbryears

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
    #5-7-2018 MC
    # MC 2018-07-10 Only if inflow of 2001-2010 is used for other date range. Once inflow will be choose in function of date range, this part will need to be modified
    if nbrleapyears(int(str(dates[0])[0:4]),int(str(dates[-1])[0:4])) != 2 \
            or dates[-1].year -dates[0].year != 9: #2018-08-30 in case time range is not 10 years
        d = pd.DataFrame(pd.date_range ( pd.datetime ( 2001, 1, 1 ), pd.datetime ( 2010, 12, 31 ), freq='d' ).tolist (), columns=['date'])
        d['Q'] = h5py.File ( filename, mode='r' )['%d/Q' % subId][:]
        d['T'] = h5py.File ( filename, mode='r' )['%d/T' % subId][:]
        d['TP'] = h5py.File ( filename, mode='r' )['%d/TP' % subId][:]
        d['DOP'] = h5py.File ( filename, mode='r' )['%d/DOP' % subId][:]

        dflow = pd.DataFrame(dates, columns=['date'])
        dflow.loc[:,'Q'] = d.loc[:,'Q']
        dflow.loc[3652, 'Q'] = d.loc[3651, 'Q']
        dflow.loc[:,'T'] = d.loc[:,'T']
        dflow.loc[3652,'T'] = d.loc[3651,'T']
        dflow.loc[:,'TP'] = d.loc[:,'TP']
        dflow.loc[3652,'TP'] = d.loc[3651,'TP']
        dflow.loc[:,'DOP'] = d.loc[:,'DOP']
        dflow.loc[3652,'DOP'] = d.loc[3651,'DOP']
        if str(dates[-1].year) == '2099':
            if str(dates[-1].month) == '11':
                dflow= dflow[:-396]
            else:
                dflow=dflow[:-365]

    else:
        dflow = pd.DataFrame ( dates, columns=['date'] )
        dflow['Q'] = h5py.File ( filename, mode='r' )['%d/Q' % subId][:]
        dflow['T'] = h5py.File ( filename, mode='r' )['%d/T' % subId][:]
        dflow['TP'] = h5py.File ( filename, mode='r' )['%d/TP' % subId][:]
        dflow['DOP'] = h5py.File ( filename, mode='r' )['%d/DOP' % subId][:]
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
    :type eh: str201
    :type subid: int
    :type inflowfile: str
    :type outpath: str
    :return: string to be written to a file
    """
    #5-7-2018 MC

    for v in variables:# 2018-08-03 MC modification to compensate if pr, ps or rsds are missing
        if v == 'pr' or v == 'ps' or v == 'rsds':
            if not os.path.isfile ( pA[v] ):
                if os.path.isfile ( pB[v] ):
                    pA[v] = pB[v]
            else:
                if not os.path.isfile ( pB[v] ):
                    pB[v] = pA[v]


    if pA['clt'].find('MOHC-HadGEM2-ES') != -1 and pA['clt'].find('r1i1p1_SMHI-RCA4_v1_day')!= -1 : # 5-24-2018 MC
        dfmissingdata = pd.concat([take5_missingdata(pA, datesA, eh), take5_missingdata(pB, datesB, eh)]) #5-24-2017 MC
        if datesB[-1] == pd.datetime(2099,11,30):#2018-08-01 MC
            dfmissingdata= dfmissingdata[:-31]
        df = dfmissingdata.interpolate() # 5-24-2018 MC
    else:
        df = pd.concat([take5(pA, datesA, eh), take5(pB, datesB, eh)])

    ndays = len(datesA) + len(datesB)
    df.index = np.arange(ndays)
    dflow = inflow5(inflowfile,  datesA + datesB , subid)
    repd = [datesA[0] + datetime.timedelta(d) for d in range ( -(365 * 2), ndays )]
    mlyear = np.array ( [d.year for d in repd] )
    mlmonth = np.array ( [d.month for d in repd] )
    mlday = np.array ( [d.day for d in repd] )
    mlndays = 365 + 365 + ndays
    repeati = list(range(365)) + list(range(365)) + list(range(ndays))
    spacer = np.repeat ( [0], repeats=ndays )[repeati].reshape ( (mlndays, 1) )
    # stream_Q = np.repeat([2000], repeats = ndays)[repeati].reshape((mlndays, 1))
    # stream_T = np.repeat([10], repeats = ndays)[repeati].reshape((mlndays, 1))
    stream_O = np.repeat ( [12000], repeats=ndays )[repeati].reshape ( (mlndays, 1) ) #MC 06-01-2018 initial parameters stream_O:8000
    stream_C = np.repeat ( [0.5], repeats=ndays )[repeati].reshape ( (mlndays, 1) )
    # stream_TP = np.repeat([5], repeats = ndays)[repeati].reshape((mlndays, 1))
    # stream_DOP = np.repeat([1], repeats = ndays)[repeati].reshape((mlndays, 1))
    stream_SS = np.repeat ( [0.01], repeats=ndays )[repeati].reshape ( (mlndays, 1) )
    stream_Chl = np.repeat ( [0.01], repeats=ndays )[repeati].reshape ( (mlndays, 1) )
    stream_DOC = np.repeat ( [2000], repeats=ndays )[repeati].reshape ( (mlndays, 1) )#MC 06-01-2018 initial parameters 8000
    stream_DIC = np.repeat ( [20000], repeats=ndays )[repeati].reshape ( (mlndays, 1) )
    temporarypath = '%s.temp' % outpath
    np.savetxt ( temporarypath,
                 np.concatenate ( (mlyear.reshape ( (mlndays, 1) ),
                                   mlmonth.reshape ( (mlndays, 1) ),
                                   mlday.reshape ( (mlndays, 1) ),
                                   df['rsds'][repeati].values.reshape ( (mlndays, 1) ),
                                   df['clt'][repeati].values.reshape ( (mlndays, 1) ),
                                   df['tas'][repeati].values.reshape ( (mlndays, 1) ),
                                   df['hurs'][repeati].values.reshape ( (mlndays, 1) ),
                                   df['ps'][repeati].values.reshape ( (mlndays, 1) ),
                                   # np.repeat([0], repeats = ndays)[repeati].reshape((mlndays, 1)),
                                   df['sfcWind'][repeati].values.reshape ( (mlndays, 1) ),
                                   df['pr'][repeati].values.reshape ( (mlndays, 1) ),
                                   dflow['Q'][repeati].values.reshape ( (mlndays, 1) ),
                                   dflow['T'][repeati].values.reshape ( (mlndays, 1) ),
                                   stream_C, stream_SS,  # C, SS
                                   dflow['TP'][repeati].values.reshape ( (mlndays, 1) ),
                                   dflow['DOP'][repeati].values.reshape ( (mlndays, 1) ),
                                   stream_Chl, stream_DOC,  # Chl, DOC
                                   stream_DIC, stream_O, spacer, spacer,  # DIC, O, NO3, NH4
                                   spacer, spacer, spacer, spacer,  # SO4, Fe, Ca, PH
                                   spacer, spacer, spacer, spacer,  # CH4, Fe3, Al3, SiO4
                                   spacer, spacer), axis=1 ),  # SiO2, diatom
                 fmt=['%i', '%i', '%i',  # yy mm dd
                      '%.4g', '%.2f', '%.2f', '%i', '%i', '%.2f', '%.3f',  # rad, cloud, temp, hum, pres, wind, precip 
                      '%.3f', '%.3f', '%.3f', '%.3f',  # 
                      '%.3f', '%.3f', '%.3f', '%.3f',  # 
                      '%.3f', '%.3f', '%i', '%i',  # 
                      '%i', '%i', '%i', '%i',  # 
                      '%i', '%i', '%i', '%i',  # 
                      '%i', '%i'],  #
                 delimiter='\t',
                 header='mylake input\nYear	Month	Day	Global_rad (MJ/m2)	Cloud_cov (-)	Air_temp (deg C)	Relat_hum (%)	Air_press (hPa)	Wind_speed (m/s)	Precipitation (mm/day)	Inflow (m3/day)	Inflow_T (deg C)	Inflow_C	Inflow_S (kg/m3)	Inflow_TP (mg/m3)	Inflow_DOP (mg/m3)	Inflow_Chla (mg/m3)	Inflow_DOC (mg/m3)	DIC	DO	NO3	NH4	SO4	Fe2	Ca	pH	CH4	Fe3	Al3	SiO4	SiO2	diatom' )
    with open ( temporarypath ) as f:
        with open ( outpath, 'w' ) as g:
            g.write ( f.read ().replace ( '-99999999', 'NaN' ) )
    os.unlink ( temporarypath )

def runlake(modelid, scenarioid, eh, subid, depth, area, longitude, latitude,k_BOD=0.01,swa_b1=1,k_SOD=100,I_scDOC=1):
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
    #5-7-2018 MC
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2A = y1A + 4
    y2B = y1B + 4

    if modelid == 4: #5-18-2018 MC
        pA = {v: '%s/Lakes_%s_EUR-11_%s_%s_%s_%s0101-%s1230.h5' %
                 (cordexfolder, v, m1, exA, m2, y1A, y2A) for v in variables}
        pB = {v: '%s/Lakes_%s_EUR-11_%s_%s_%s_%s0101-%s1230.h5' %
                 (cordexfolder, v, m1, exB, m2, y1B, y2B) for v in variables}
    else:
        pA = {v: '%s/Lakes_%s_EUR-11_%s_%s_%s_%s0101-%s1231.h5' %
                 (cordexfolder, v, m1, exA, m2, y1A, y2A) for v in variables}
        pB = {v: '%s/Lakes_%s_EUR-11_%s_%s_%s_%s0101-%s1231.h5' %
                 (cordexfolder, v, m1, exB, m2, y1B, y2B) for v in variables}

    inflowfilename = '%s/sweden_inflow_data_20010101_20101231.h5' % inflowfolder  # This is hard coded for now
    datesA = pd.date_range ( pd.datetime ( y1A, 1, 1 ), pd.datetime ( y2A, 12, 31 ), freq='d' ).tolist ()
    if pA['clt'].find ( 'EUR-11_MOHC-HadGEM2-ES_rcp85_r1i1p1_SMHI-RCA4_v1_day_2091' ) != -1:
        datesB = pd.date_range ( pd.datetime ( y1B, 1, 1 ), pd.datetime ( y2B-1, 12, 31 ), freq='d' ).tolist ()
    elif pA['clt'].find ( 'EUR-11_MOHC-HadGEM2-ES_rcp45_r1i1p1_SMHI-RCA4_v1_day_2091' ) != -1:
        datesB = pd.date_range ( pd.datetime ( y1B, 1, 1 ), pd.datetime ( y2B-1, 11, 30 ), freq='d' ).tolist ()
    else:
        datesB = pd.date_range ( pd.datetime ( y1B, 1, 1 ), pd.datetime ( y2B, 12, 31 ), freq='d' ).tolist ()

    eh = eh[2:] if eh[:2] == '0x' else eh
    while len ( eh ) < 6:
        eh = '0' + eh
    d1, d2, d3 = eh[:2], eh[:4], eh[:6]
    outdir = os.path.join ( outputfolder, d1, d2, d3,
                            'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (
                                m1, exA, exB, m2, y1A, y2B) )

    if not os.path.exists ( outdir ):
        os.makedirs ( outdir )

    # creation of empty files before risks of bug: MC 2018-07-10

    initp = os.path.join ( outdir, '2017init' )
    parp = os.path.join ( outdir, '2017par' )
    inputp = os.path.join ( outdir, '2017input' )
    if os.path.exists ( os.path.join ( outdir, '2017REDOCOMPLETE' ) ):
        print ( 'lake %s is already completed' % eh )
        #with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
        #    f.write ( 'lake %s is already completed\n' % eh )
        #    f.close ()
        ret = 0
    else:
        # empty = pd.DataFrame(np.nan, index=np.arange(0,len(datesA+datesB)), columns=np.arange(1,int(depth)+1))
        # for i in ['Tzt.csv','O2zt.csv', 'Attn_zt.csv', 'Qst.csv', 'DOCzt.csv','lambdazt.csv']:
        #     empty.to_csv('%s/%s'%(outdir,i),na_rep='NA',header=False,index=False)
        # with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
        #     f.write ('empty files created\n')
        #     f.close ()
        mylakeinit ( depth, area, initp,I_scDOC)
        mylakepar ( longitude, latitude, parp,swa_b1,k_BOD,k_SOD,I_scDOC)
        mylakeinput ( pA, pB, datesA, datesB, eh, subid, inflowfilename, inputp )
        cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\');quit' % (initp, parp, inputp, y1A - 2, y2B, outdir)
        print ( cmd )
        os.system ( cmd )
        #for f in [initp, parp, inputp]:
        #    os.system ( 'bzip2 -f -k %s' % f )
        expectedfs = [ 'Tzt.csv','O2zt.csv', 'Attn_zt.csv', 'Qst.csv', 'DOCzt.csv','lambdazt.csv']
        flags = [os.path.exists ( os.path.join ( outdir, f ) ) for f in expectedfs]

        if all ( flags ):
            # with open ( os.path.join ( outdir, '2017REDOCOMPLETE' ), 'w' ) as f:
            #     f.write ( datetime.datetime.now ().isoformat () )
            ret=0
        ret = 0 if all ( flags ) else 100
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
