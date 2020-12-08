#!/usr/bin/env python

""" Script for MyLake - ISIMIP
Calls the init, input and par scripts to create the appropriate files for MyLake model
Then launches MyLake for the specified lake.
"""

__author__ = "Julien Bellavance and Marianne Cote"

import csv
import os
import numpy as np
import datetime
import netCDF4 as Ncdf
import pandas as pd
import statistics
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt, floor
import numpy as np
import pandas as pd
import h5py
import datetime
import os
import shutil
import bz2
import math
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from scipy import stats
import statsmodels.api as sm

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

cordexfolder = 'F:\MCote output\optimisation_2018_10_12\cordex' #5-24-2018 MC #Need to be change depending where the climatic files where
inflowfolder = r'C:\Users\macot620\Documents\GitHub\Fish_niche\sweden_inflow_data'
outputfolder = r'F:\output' #5-9-2018 MC
observation_folder = "Observations"
input_folder = "Inputs"

output_folder = r"D:\output_fish_niche"

lakes_data = pd.read_csv("2017SwedenList.csv",encoding='ISO-8859-1')
lakes_data = lakes_data.set_index("lake_id").to_dict()
lakes_list = list(lakes_data.get("name").keys())
lakes_list = list(lakes_data.get("ebhex").keys())


def simulation_years(modelid, scenarioid):
    """
    This function selects the initial and final years of the simulation in the function of the climatic model and
    scenario.
    :param modelid: One of the climatic model listed by ISIMIP.
    :param scenarioid: One of the climatic scenario listed by ISIMIP.
    :return y1, y2: initial and final years of simulation.
    """
    if modelid == "EWEMBI":
        y1, y2 = 1979, 2016
    elif scenarioid == 'piControl':
        if modelid == "GFDL-ESM2M":
            y1, y2 = 1661, 2099
        else:
            y1, y2 = 1661, 2299
    elif scenarioid == 'historical':
        y1, y2 = 1861, 2005
    elif scenarioid == 'rcp26':
        if modelid == "GFDL-ESM2M":
            y1, y2 = 2006, 2099
        else:
            y1, y2 = 2006, 2299
    elif modelid == "IPSL-CM5A-LR" and scenarioid == 'rcp85':
        y1, y2 = 2006, 2299
    else:
        y1, y2 = 2006, 2099

    return y1, y2


def calculatedensity(temp):
    """
    Calculate density by using equation giving in ISIMIP lake model protocol.
    :param temp: average daily temperature at a specific layer.
    :return density: Water density for the specific layer.
    """
    t = temp
    density = (999.842594 + (6.793952e-2 * t) - (9.095290e-3 * t ** 2) +
               (1.001685e-4 * t ** 3) - (1.120083e-6 * t ** 4) + (6.536336e-9 * t ** 5))
    return density


def outputfile(y1, y2, outfolder):
    """
    Function calculating all variables asked by ISIMIP and formates them into a txt file.
    :param y1: Initial year of the simulation
    :param y2: Final year of the simulation
    :param outfolder: folder directory to where the raw data produce by the model are.
    :return: None
    """
    try:
        dates = [d.strftime('%Y-%m-%d') for d in pd.date_range('%s-01-01' % y1, '%s-12-31' % y2)]
    except:
        dates = [d.strftime('%Y-%m-%d') for d in pd.period_range('%s-01-01' % y1, '%s-12-31' % y2, freq='D')]

    print("run output")

    if not os.path.exists(os.path.join(outfolder, "strat.csv")) or not os.path.exists(
            os.path.join(outfolder, "watertemp.csv")) \
            or not os.path.exists(os.path.join(outfolder, "thermodepth.csv")):
        with open("{}/Tzt.csv".format(outfolder), "r") as observation_file:
            try:
                rows = list(csv.reader(observation_file))
                strat = []
                maxdensity = []
                watertemp = []
                for i in range(0, len(rows)):
                    data = rows[i]
                    row = ["%s, 00:00:00" % dates[i]]
                    waterrow = ["%s, 00:00:00" % dates[i]]
                    maxrowdensity = ["%s, 00:00:00" % dates[i]]

                    if abs(calculatedensity(float(data[0])) - calculatedensity(float(data[-1]))) > 0.1:
                        row.append(1)
                    else:
                        row.append(0)

                    strat.append(row)
                    density = []

                    for j in range(0, len(data)):
                        density.append(calculatedensity(float(data[j])))
                        waterrow.append(float(data[j]) + 273.15)

                    watertemp.append(waterrow)
                    maxrowdensity.append(density.index(max(density)))
                    maxdensity.append(maxrowdensity)
                if not os.path.exists(os.path.join(outfolder, "strat.csv")):
                    try:
                        with open(os.path.join(outfolder, "strat.csv"), "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerows(strat)
                    except:
                        if os.path.exists(os.path.join(outfolder, "strat.csv")):
                            os.remove(os.path.join(outfolder, "strat.csv"))
                if not os.path.exists(os.path.join(outfolder, "watertemp.csv")):
                    try:
                        with open(os.path.join(outfolder, "watertemp.csv"), "w", newline="") as f2:
                            writer = csv.writer(f2)
                            writer.writerows(watertemp)
                    except:
                        if os.path.exists(os.path.join(outfolder, "watertemp.csv")):
                            os.remove(os.path.join(outfolder, "watertemp.csv"))

                if not os.path.exists(os.path.join(outfolder, "thermodepth.csv")):
                    try:
                        with open(os.path.join(outfolder, "thermodepth.csv"), "w", newline="") as f1:
                            writer = csv.writer(f1)
                            writer.writerows(maxdensity)
                    except:
                        if os.path.exists(os.path.join(outfolder, "thermodepth.csv")):
                            os.remove(os.path.join(outfolder, "thermodepth.csv"))
            except:
                print("error")

    if not os.path.exists(os.path.join(outfolder, "ice.csv")) or not os.path.exists(
            os.path.join(outfolder, "icetick.csv")) \
            or not os.path.exists(os.path.join(outfolder, "snowtick.csv")):
        try:
            with open("{}/His.csv".format(outfolder), "r") as ice_file:
                rows = list(csv.reader(ice_file))
                ice = []
                lakeicefrac = []
                snowtick = []
                for i in range(0, len(rows)):
                    data = rows[i]
                    ice.append(["%s, 00:00:00" % dates[i], float(data[6])])
                    lakeicefrac.append(["%s, 00:00:00" % dates[i], float(data[0])])
                    snowtick.append(["%s, 00:00:00" % dates[i], float(data[2])])
                if not os.path.exists(os.path.join(outfolder, "ice.csv")):
                    try:
                        with open(os.path.join(outfolder, "ice.csv"), "w", newline="") as f3:
                            writer = csv.writer(f3)
                            writer.writerows(ice)
                    except:
                        if os.path.exists(os.path.join(outfolder, "ice.csv")):
                            os.remove(os.path.join(outfolder, "ice.csv"))

                if not os.path.exists(os.path.join(outfolder, "icetick.csv")):
                    try:
                        with open(os.path.join(outfolder, "icetick.csv"), "w", newline="") as f4:
                            writer = csv.writer(f4)
                            writer.writerows(lakeicefrac)
                    except:
                        if os.path.exists(os.path.join(outfolder, "icetick.csv")):
                            os.remove(os.path.join(outfolder, "icetick.csv"))

                if not os.path.exists(os.path.join(outfolder, "snowtick.csv")):
                    try:
                        with open(os.path.join(outfolder, "snowtick.csv"), "w", newline="") as f4:
                            writer = csv.writer(f4)
                            writer.writerows(snowtick)
                    except:
                        if os.path.exists(os.path.join(outfolder, "snowtick.csv")):
                            os.remove(os.path.join(outfolder, "snowtick.csv"))
        except:
            print("errpr his")

    if not os.path.exists(os.path.join(outfolder, "sensheatf.csv")) or \
            not os.path.exists(os.path.join(outfolder, "latentheatf.csv")) or \
            not os.path.exists(os.path.join(outfolder, "lakeheatf.csv")) or \
            not os.path.exists(os.path.join(outfolder, "albedo.csv")):

        his = pd.read_csv("{}/His.csv".format(outfolder), names=['0', '1', '2', '3', '4', '5', '6', '7'])
        print(len(his), len(dates))
        if dates[0] == '1661-01-01':
            y1, y2 = 1681, 1780
            dates = [d.strftime('%Y-%m-%d') for d in
                     pd.period_range('%s-01-01' % y1, '%s-12-31' % y2, freq='D')]
            print(len(his), len(dates))

        elif dates[0] == '2211-01-01':
            y1, y2 = 2111, 2199
            dates = [d.strftime('%Y-%m-%d') for d in
                     pd.period_range('%s-01-01' % y1, '%s-12-31' % y2, freq='D')]
            print(len(his), len(dates))

        elif dates[0] == '2261-01-01':
            y1, y2 = 2161, 2199
            dates = [d.strftime('%Y-%m-%d') for d in
                     pd.period_range('%s-01-01' % y1, '%s-12-31' % y2, freq='D')]
            print(len(his), len(dates))

        his['date'] = dates
        his['datetime'] = pd.to_datetime(his['date'])

        try:
            with open("{}/Qst.csv".format(outfolder), "r") as qst_file:
                rows = list(csv.reader(qst_file))
                sensheatf = []
                latentheatf = []
                lakeheatf = []
                albedo = []
                for i in range(0, len(rows)):
                    data = rows[i]
                    sensheatf.append(["%s, 00:00:00" % dates[i], float(data[3])])
                    latentheatf.append(["%s, 00:00:00" % dates[i], float(data[4])])
                    lakeheatf.append(["%s, 00:00:00" % dates[i], float(data[0]) + float(data[1]) - float(data[2])])
                    albedo.append(["%s, 00:00:00" % dates[i], float(data[5])])
                if not os.path.exists(os.path.join(outfolder, "sensheatf.csv")):
                    try:
                        with open(os.path.join(outfolder, "sensheatf.csv"), "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerows(sensheatf)
                    except:
                        if os.path.exists(os.path.join(outfolder, "sensheatf.csv")):
                            os.remove(os.path.join(outfolder, "sensheatf.csv"))

                if not os.path.exists(os.path.join(outfolder, "latentheatf.csv")):
                    try:
                        with open(os.path.join(outfolder, "latentheatf.csv"), "w", newline="") as f2:
                            writer = csv.writer(f2)
                            writer.writerows(latentheatf)
                    except:
                        if os.path.exists(os.path.join(outfolder, "latentheatf.csv")):
                            os.remove(os.path.join(outfolder, "latentheatf.csv"))

                if not os.path.exists(os.path.join(outfolder, "lakeheatf.csv")):
                    try:
                        with open(os.path.join(outfolder, "lakeheatf.csv"), "w", newline="") as f1:
                            writer = csv.writer(f1)
                            writer.writerows(lakeheatf)
                    except:
                        if os.path.exists(os.path.join(outfolder, "lakeheatf.csv")):
                            os.remove(os.path.join(outfolder, "lakeheatf.csv"))

                if not os.path.exists(os.path.join(outfolder, "albedo.csv")):
                    try:
                        with open(os.path.join(outfolder, "albedo.csv"), "w", newline="") as f1:
                            writer = csv.writer(f1)
                            writer.writerows(albedo)
                    except:
                        if os.path.exists(os.path.join(outfolder, "albedo.csv")):
                            os.remove(os.path.join(outfolder, "albedo.csv"))
        except:
            print("issues with QST")

    if not os.path.exists(os.path.join(outfolder, "turbdiffheat.csv")):

        try:
            with open("{}/Kzt.csv".format(outfolder), "r") as qst_file:
                rows = list(csv.reader(qst_file))
                turbdiffheat = []

                for i in range(0, len(rows)):
                    data = rows[i]
                    turbrow = ["%s, 00:00:00" % dates[i]]

                    for j in range(0, len(data)):
                        turbrow.append(float(data[j]))
                    turbdiffheat.append(turbrow)

            with open(os.path.join(outfolder, "turbdiffheat.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(turbdiffheat)
        except:
            if os.path.exists(os.path.join(outfolder, "turbdiffheat.csv")):
                os.remove(os.path.join(outfolder, "turbdiffheat.csv"))

    if not os.path.exists(os.path.join(outfolder, "sedheatf.csv")):
        with open("{}/Qzt_sed.csv".format(outfolder), "r") as qst_file:
            rows = list(csv.reader(qst_file))
            sedheatf = []

            try:
                for i in range(0, len(rows)):
                    data = rows[i]
                    sedheat = ["%s, 00:00:00" % dates[i]]

                    for j in range(0, len(data)):
                        sedheat.append(float(data[j]))
                    sedheatf.append(sedheat)

                if 1 == 1:
                    with open(os.path.join(outfolder, "sedheatf.csv"), "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(sedheatf)
            except:
                if os.path.exists(os.path.join(outfolder, "sedheatf.csv")):
                    os.remove(os.path.join(outfolder, "sedheatf.csv"))


def find_init_temp_daily(observations, depth_levels):
    """
    J. Bellavance 2018/12/18
    For ISI-MIP
    With temperature .csv file opened, searches for the specified date in the timestamp column. Then checks if the data
    set for that date is complete (is there a temperature value for every known depth level for this lake).
    If not, interpolate the missing data (with missing_temp).

    :param observations: Type list. A list made from an opened .csv file with sub-daily temperatures.
    :param depth_levels: Type list. The depth levels obtained from the hypsometry file. Depth levels values are floats.

    :return: Type list. A complete set of mean temperatures for init files, ordered by depth levels.
    """
    if len(observations) == 0:
        print("Date not found, using dummy temperatures")
        return list("4" * len(depth_levels))

    obs_list = []

    for observation in observations:

        if int(observation[2][4:]) < 101 or int(observation[2][4:]) > 101 + 20:
            continue
        elif not obs_list:
            obs_list.append(observation)
        elif observation[2][4:] == obs_list[0][2][4:]:
            obs_list.append(observation)

    w_temp = []
    m = 0
    try:
        if len(obs_list) > 0:
            for depth in depth_levels:

                try:
                    if float(obs_list[m][3]) == depth or (m == 0 and float(obs_list[m][3]) < 1):
                        w_temp.append(float(obs_list[m][4]))
                        m += 1

                    else:
                        w_temp.append("")
                except IndexError:
                    w_temp.append("")
        else:
            for i in range(len(depth_levels)):
                w_temp.append(4)

        if "" in w_temp:
            return missing_temp(w_temp, depth_levels)
        else:
            return w_temp
    except:
        print("Date not found, using dummy temperatures")
        return list("4" * len(depth_levels))


def find_init_temp_subdaily(observations, depth_levels, date_init):
    """
    J. Bellavance 2018/12/18
    For ISI-MIP
    With temperature .csv file opened, searches for the specified date in the time stamp column.
    Then checks if the data set for that date is complete (is there a temperature value for every known depth level
    for this lake). If not, interpolate the missing data (with missing_temp).

    :param observations: Type list. A list made from an opened .csv file.
    :param depth_levels: Type list. The depth levels obtained from the hypsometry file. Depth levels values are floats.
    :param date_init: Type int. Dates used to initialise data. Must be in the form of 'MMDD'.
                                Years must not be specified.
    :return: Type list. A complete set of mean temperatures for init files, ordered by depth levels.
    """
    if len(observations) == 0:
        print("Date not found, using dummy temperatures")
        return list("4" * len(depth_levels))

    obs_list = []

    for observation in observations:
        if int(observation[2][4:8]) < date_init or int(observation[2][4:8]) > date_init + 20:
            continue
        elif obs_list is []:
            obs_list.append(observation)
        elif observation[2][4:] == obs_list[0][2][4:]:
            obs_list.append(observation)

    w_temp = []
    m = 0

    try:
        if len(obs_list) > 0:
            for depth in depth_levels:
                try:
                    if float(obs_list[m][3]) == depth or (m == 0 and float(obs_list[m][3]) < 1):
                        w_temp.append(float(obs_list[m][4]))
                        m += 1

                    else:
                        w_temp.append("")

                except IndexError:
                    w_temp.append("")
        else:
            for i in range(len(depth_levels)):
                w_temp.append(4)

        if "" in w_temp:
            return missing_temp(w_temp, depth_levels)
        else:
            return w_temp
    except:
        print("Date not found, using dummy temperatures")
        return list("4" * len(depth_levels))


def missing_temp(temp_list, depth_levels):
    """
    J. Bellavance 2019/10/01
    Interpolates missing temperatures for find_init_temp
    :param temp_list: Type list. The list of initial temperatures from find_init_temp, with empty strings where
    temperatures are missing.
    :param depth_levels: Type list. The list of depth levels used in find_init_temp.
    :return: Type list. The list of initial temperatures with the interpolated values.
    """
    observed_depths = []

    for depth in depth_levels:
        if temp_list[depth_levels.index(depth)] != "":
            observed_depths.append(depth)

    while "" in temp_list:
        temp_list.remove("")

    for depth in depth_levels:
        if depth in observed_depths:
            continue

        else:
            if depth < observed_depths[0]:
                temp_list.insert(0, temp_list[0])
                observed_depths.insert(0, depth)
            elif depth > observed_depths[-1]:
                temp_list.append(temp_list[-1])
                observed_depths.append(depth)

            else:
                temp_list.insert(depth_levels.index(depth), np.interp(depth, observed_depths, temp_list))
                observed_depths.insert(depth_levels.index(depth), depth)

    return temp_list


def get_longitude(lake_name, forcing_data_directory):
    """
    Obtains longitude from a given ncdf file.
    :param lake_name: string.
    :param forcing_data_directory: string. The directory with the ncdf files.
    :return: float. the longitude of the lake.
    """
    if lake_name == 'Mozhaysk':
        lake_name = 'Mozaisk'
    try:
        print(forcing_data_directory + "/hurs_EWEMBI_historical_{}.allTS.nc".format(lake_name))
        ncdf_file = Ncdf.Dataset(
            forcing_data_directory + "/hurs_EWEMBI_historical_{}.allTS.nc".format(lake_name), "r", format="NETCDF4")

        return ncdf_file.variables["lon"][0]
    except:
        print("problem!")


def get_latitude(lake_name, forcing_data_directory):
    """
    Obtains latitude from a given ncdf file.
    :param lake_name: string.
    :param forcing_data_directory: string. The directory with the ncdf files.
    :return: float. the latitude of the lake.
    """
    if lake_name == 'Mozhaysk':
        lake_name = 'Mozaisk'
    try:
        print(forcing_data_directory + "/hurs_EWEMBI_historical_{}.allTS.nc".format(lake_name))
        ncdf_file = Ncdf.Dataset(
            forcing_data_directory + "/hurs_EWEMBI_historical_{}.allTS.nc".format(lake_name), "r", format="NETCDF4")

        return ncdf_file.variables["lat"][0]
    except:
        print("problem!")


class LakeInfo:
    """
    Class containing all info needed by the script to run model and calibration.
    """

    def __init__(self, lake_name, lake_id, subid, ebhex, area, depth, longitude, latitude, volume, scenarioid=2):
        """
        initiate parameters
        :param lake_name: string containing the long name of the lake
        """
        self.lake_name = lake_name
        self.ebhex = ebhex
        self.lake_id = lake_id
        self.subid = subid
        self.area = area
        self.depth = depth
        self.volume = volume
        self.longitude = longitude
        self.latitude = latitude
        self.observation_file = os.path.join(inflowfolder, "Validation_data_for_lookup.xlsx")

        ebhex = ebhex[2:] if ebhex[:2] == '0x' else ebhex
        while len(ebhex) < 6:
            ebhex = '0' + ebhex
        d1, d2, d3 = ebhex[:2], ebhex[:4], ebhex[:6]
        outdir = os.path.join(outputfolder, d1, d2, d3)

        self.output_path = outdir

        exA, y1A, exB, y1B = scenarios[scenarioid]
        self.start_year = y1A
        self.end_year = y1B + 4
        self.calibration_path = os.path.join(self.output_path,
                                             "EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231")
        if os.path.exists(os.path.join(self.calibration_path, "Calibration_Complete.csv")):
            data = pd.read_csv(os.path.join(self.calibration_path, "Calibration_Complete.csv"), header=None)
            if os.path.exists(os.path.join(self.calibration_path, "Calibration_CompleteOXY.csv")):
                dataOXY = pd.read_csv(os.path.join(self.calibration_path, "Calibration_CompleteOXY.csv"), header=None)

                if os.path.exists(os.path.join(self.calibration_path, "2017par")):
                    with open(os.path.join(self.calibration_path, "2017par")) as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.split(sep="	")
                            line[0] = line[0].replace(" ", "")

                            if line[0] == "Kz_N0":
                                self.kz_n0 = data.iloc[0, 0]

                            elif line[0] == "C_shelter":
                                self.c_shelter = data.iloc[0, 1]


                            elif line[0] == "alb_melt_ice":
                                self.alb_melt_ice = data.iloc[0, 2]

                            elif line[0] == "alb_melt_snow":
                                self.alb_melt_snow = data.iloc[0, 3]

                            elif line[0] == "I_scV":
                                self.i_scv = data.iloc[0, 4]

                            elif line[0] == "I_scT":
                                self.i_sct = data.iloc[0, 5]

                            elif line[0] == "I_scO":
                                self.i_sco = dataOXY.iloc[0, 3]

                            elif line[0] == "I_scDOC":
                                self.i_sc_doc = dataOXY.iloc[0, 0]

                            elif line[0] == "swa_b0":
                                self.swa_b0 = data.iloc[0, 6]

                            elif line[0] == "swa_b1":
                                self.swa_b1 = data.iloc[0, 7]

                            elif line[0] == "k_BOD":
                                self.k_bod = dataOXY.iloc[0, 1]

                            elif line[0] == "k_SOD":
                                self.k_sod = dataOXY.iloc[0, 2]

                elif os.path.exists(os.path.join(self.calibration_path, "2020par")):
                    with open(os.path.join(self.calibration_path, "2020par")) as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.split(sep="	")
                            line[0] = line[0].replace(" ", "")

                            if line[0] == "Kz_N0":
                                self.kz_n0 = data.iloc[0, 0]

                            elif line[0] == "C_shelter":
                                self.c_shelter = data.iloc[0, 1]


                            elif line[0] == "alb_melt_ice":
                                self.alb_melt_ice = data.iloc[0, 2]

                            elif line[0] == "alb_melt_snow":
                                self.alb_melt_snow = data.iloc[0, 3]

                            elif line[0] == "I_scV":
                                self.i_scv = data.iloc[0, 4]

                            elif line[0] == "I_scT":
                                self.i_sct = data.iloc[0, 5]

                            elif line[0] == "I_scO":
                                self.i_sco = dataOXY.iloc[0, 3]

                            elif line[0] == "I_scDOC":
                                self.i_sc_doc = dataOXY.iloc[0, 0]

                            elif line[0] == "swa_b0":
                                self.swa_b0 = data.iloc[0, 6]

                            elif line[0] == "swa_b1":
                                self.swa_b1 = data.iloc[0, 7]

                            elif line[0] == "k_BOD":
                                self.k_bod = dataOXY.iloc[0, 1]

                            elif line[0] == "k_SOD":
                                self.k_sod = dataOXY.iloc[0, 2]

                else:
                    self.kz_n0 = data.iloc[0, 0]
                    self.c_shelter = data.iloc[0, 1]
                    self.alb_melt_ice = data.iloc[0, 2]
                    self.alb_melt_snow = data.iloc[0, 3]
                    self.i_scv = data.iloc[0, 4]
                    self.i_sct = data.iloc[0, 5]
                    self.swa_b0 = data.iloc[0, 6]
                    self.swa_b1 = data.iloc[0, 7]
                    self.k_bod = dataOXY.iloc[0, 1]
                    self.k_sod = dataOXY.iloc[0, 2]
                    self.i_sc_doc = dataOXY.iloc[0, 0]
                    self.i_sco = dataOXY.iloc[0, 3]

            else:
                if os.path.exists(os.path.join(self.calibration_path, "2017par")):
                    with open(os.path.join(self.calibration_path, "2017par")) as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.split(sep="	")
                            line[0] = line[0].replace(" ", "")

                            if line[0] == "Kz_N0":
                                self.kz_n0 = data.iloc[0, 0]

                            elif line[0] == "C_shelter":
                                self.c_shelter = data.iloc[0, 1]


                            elif line[0] == "alb_melt_ice":
                                self.alb_melt_ice = data.iloc[0, 2]

                            elif line[0] == "alb_melt_snow":
                                self.alb_melt_snow = data.iloc[0, 3]

                            elif line[0] == "I_scV":
                                self.i_scv = data.iloc[0, 4]

                            elif line[0] == "I_scT":
                                self.i_sct = data.iloc[0, 5]

                            elif line[0] == "I_scO":
                                self.i_sco = line[1]


                            elif line[0] == "I_scDOC":
                                self.i_sc_doc = line[1]

                            elif line[0] == "swa_b0":
                                self.swa_b0 = data.iloc[0, 6]

                            elif line[0] == "swa_b1":
                                self.swa_b1 = data.iloc[0, 7]

                            elif line[0] == "k_BOD":
                                self.k_bod = line[1]

                            elif line[0] == "k_SOD":
                                self.k_sod = line[1]

                elif os.path.exists(os.path.join(self.calibration_path, "2020par")):
                    with open(os.path.join(self.calibration_path, "2020par")) as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.split(sep="	")
                            line[0] = line[0].replace(" ", "")

                            if line[0] == "Kz_N0":
                                self.kz_n0 = data.iloc[0, 0]

                            elif line[0] == "C_shelter":
                                self.c_shelter = data.iloc[0, 1]


                            elif line[0] == "alb_melt_ice":
                                self.alb_melt_ice = data.iloc[0, 2]

                            elif line[0] == "alb_melt_snow":
                                self.alb_melt_snow = data.iloc[0, 3]

                            elif line[0] == "I_scV":
                                self.i_scv = data.iloc[0, 4]

                            elif line[0] == "I_scT":
                                self.i_sct = data.iloc[0, 5]

                            elif line[0] == "I_scO":
                                self.i_sco = line[1]

                            elif line[0] == "I_scDOC":
                                self.i_sc_doc = line[1]

                            elif line[0] == "swa_b0":
                                self.swa_b0 = data.iloc[0, 6]

                            elif line[0] == "swa_b1":
                                self.swa_b1 = data.iloc[0, 7]

                            elif line[0] == "k_BOD":
                                self.k_bod = line[1]

                            elif line[0] == "k_SOD":
                                self.k_sod = line[1]

                else:
                    self.kz_n0 = data.iloc[0, 0]
                    self.c_shelter = data.iloc[0, 1]
                    self.alb_melt_ice = data.iloc[0, 2]
                    self.alb_melt_snow = data.iloc[0, 3]
                    self.i_scv = data.iloc[0, 4]
                    self.i_sct = data.iloc[0, 5]
                    self.i_sco = 1
                    self.swa_b0 = data.iloc[0, 6]
                    self.swa_b1 = data.iloc[0, 7]
                    self.k_bod = 500
                    self.k_sod = 0.1
                    self.i_sc_doc = 1

        else:
            if os.path.exists(os.path.join(self.calibration_path, "2017par")):
                with open(os.path.join(self.calibration_path, "2017par")) as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.split(sep="	")
                        line[0] = line[0].replace(" ", "")

                        if line[0] == "Kz_N0":
                            self.kz_n0 = line[1]

                        elif line[0] == "C_shelter":
                            self.c_shelter = line[1]


                        elif line[0] == "alb_melt_ice":
                            self.alb_melt_ice = line[1]

                        elif line[0] == "alb_melt_snow":
                            self.alb_melt_snow = line[1]

                        elif line[0] == "I_scV":
                            self.i_scv = line[1]

                        elif line[0] == "I_scT":
                            self.i_sct = line[1]

                        elif line[0] == "I_scO":
                            self.i_sco = line[1]


                        elif line[0] == "I_scDOC":
                            self.i_sc_doc = line[1]

                        elif line[0] == "swa_b0":
                            self.swa_b0 = line[1]

                        elif line[0] == "swa_b1":
                            self.swa_b1 = line[1]

                        elif line[0] == "k_BOD":
                            self.k_bod = line[1]

                        elif line[0] == "k_SOD":
                            self.k_sod = line[1]

            elif os.path.exists(os.path.join(self.calibration_path, "2020par")):
                with open(os.path.join(self.calibration_path, "2020par")) as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.split(sep="	")
                        line[0] = line[0].replace(" ", "")

                        if line[0] == "Kz_N0":
                            self.kz_n0 = line[1]

                        elif line[0] == "C_shelter":
                            self.c_shelter = line[1]


                        elif line[0] == "alb_melt_ice":
                            self.alb_melt_ice = line[1]

                        elif line[0] == "alb_melt_snow":
                            self.alb_melt_snow = line[1]

                        elif line[0] == "I_scV":
                            self.i_scv = line[1]

                        elif line[0] == "I_scT":
                            self.i_sct = line[1]

                        elif line[0] == "I_scO":
                            self.i_sco = line[1]

                        elif line[0] == "I_scDOC":
                            self.i_sc_doc = line[1]

                        elif line[0] == "swa_b0":
                            self.swa_b0 = line[1]

                        elif line[0] == "swa_b1":
                            self.swa_b1 = line[1]

                        elif line[0] == "k_BOD":
                            self.k_bod = line[1]

                        elif line[0] == "k_SOD":
                            self.k_sod = line[1]

            else:
                self.kz_n0 = 7.00E-05
                self.c_shelter = "NaN"
                self.alb_melt_ice = 0.6
                self.alb_melt_snow = 0.9
                self.i_scv = 1.339
                self.i_sct = 1.781
                self.i_sco = 1
                self.swa_b0 = 2.5
                self.swa_b1 = 1
                self.k_bod = 0.1
                self.k_sod = 500
                self.i_sc_doc = 1

    def runlake(self, modelid, scenarioid):
        """

        :param modelid: model used
        :param scenarioid: scenario used
        :param ebhex: ebhex number
        :param subid: Reference number
        :param depth: depth used for initiate Mylake (see mylakeinit())
        :param area: area used for initiate Mylake (see mylakeinit())
        :param longitude: longitude coordinate for Mylake (see mylakepar())
        :param latitude: latitude coordinate for Mylake (see mylakepar())
        :return:
        .. note:: see above lines for models and scenarios (dictionaries). ebhex: EBhex
        """
        # 5-7-2018 MC
        ret = True
        exA, y1A, exB, y1B = scenarios[scenarioid]
        m1, m2 = models[modelid]
        y2A = y1A + 4
        y2B = y1B + 4

        if modelid == 4:  # 5-18-2018 MC
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
        datesA = pd.date_range(pd.datetime(y1A, 1, 1), pd.datetime(y2A, 12, 31), freq='d').tolist()
        if pA['clt'].find('EUR-11_MOHC-HadGEM2-ES_rcp85_r1i1p1_SMHI-RCA4_v1_day_2091') != -1:
            datesB = pd.date_range(pd.datetime(y1B, 1, 1), pd.datetime(y2B - 1, 12, 31), freq='d').tolist()
        elif pA['clt'].find('EUR-11_MOHC-HadGEM2-ES_rcp45_r1i1p1_SMHI-RCA4_v1_day_2091') != -1:
            datesB = pd.date_range(pd.datetime(y1B, 1, 1), pd.datetime(y2B - 1, 11, 30), freq='d').tolist()
        else:
            datesB = pd.date_range(pd.datetime(y1B, 1, 1), pd.datetime(y2B, 12, 31), freq='d').tolist()


        outdir = os.path.join(self.output_path,
                              'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (
                                  m1, exA, exB, m2, y1A, y2B))

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # creation of empty files before risks of bug: MC 2018-07-10

        initp = os.path.join(outdir, '2020init')
        if os.path.exists(os.path.join(outdir, '2017par')):
            parp = os.path.join(outdir, '2020par')
        else:
            parp = os.path.join(outdir, '2020par')
        inputp = os.path.join(outdir, '2020input')
        if os.path.exists(os.path.join(outdir, '2020REDOCOMPLETE')):
            print('lake %s is already completed' % self.ebhex)
            # with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
            #    f.write ( 'lake %s is already completed\n' % ebhex )
            #    f.close ()
            ret = 0
        else:
            # empty = pd.DataFrame(np.nan, index=np.arange(0,len(datesA+datesB)), columns=np.arange(1,int(depth)+1))
            # for i in ['Tzt.csv','O2zt.csv', 'Attn_zt.csv', 'Qst.csv', 'DOCzt.csv','lambdazt.csv']:
            #     empty.to_csv('%s/%s'%(outdir,i),na_rep='NA',header=False,index=False)
            # with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
            #     f.write ('empty files created\n')
            #     f.close ()

            self.mylakeinit(initp)
            self.mylakepar(parp)
            self.mylakeinput(pA, pB, datesA, datesB,inflowfilename, inputp)
            cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran_optimize(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\',%d);quit' % (
            initp, parp, inputp, y1A , y2B, outdir,1)
            print(cmd)
            os.system(cmd)
        #     # for f in [initp, parp, inputp]:
        #     #    os.system ( 'bzip2 -f -k %s' % f )
            expectedfs = ['Tzt.csv', 'O2zt.csv', 'Attn_zt.csv', 'Qst.csv', 'DOCzt.csv', 'PARMaxt.csv', 'PARzt.csv', 'His.csv', 'lambdazt.csv']
            flags = [os.path.exists(os.path.join(outdir, f)) for f in expectedfs]

            if all(flags):
                with open(os.path.join(outdir, '2020REDOCOMPLETE'), 'w') as f:
                    f.write(datetime.datetime.now().isoformat())
                ret = False
            #         ret = 0
        #     ret = 0 if all(flags) else

        return ret

    def lake_input(self, modelid, scenarioid):
        """

        :param modelid: model used
        :param scenarioid: scenario used
        :param ebhex: ebhex number
        :param subid: Reference number
        :param depth: depth used for initiate Mylake (see mylakeinit())
        :param area: area used for initiate Mylake (see mylakeinit())
        :param longitude: longitude coordinate for Mylake (see mylakepar())
        :param latitude: latitude coordinate for Mylake (see mylakepar())
        :return:
        .. note:: see above lines for models and scenarios (dictionaries). ebhex: EBhex
        """
        # 5-7-2018 MC
        exA, y1A, exB, y1B = scenarios[scenarioid]
        m1, m2 = models[modelid]
        y2A = y1A + 4
        y2B = y1B + 4

        if modelid == 4:  # 5-18-2018 MC
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
        datesA = pd.date_range(pd.datetime(y1A, 1, 1), pd.datetime(y2A, 12, 31), freq='d').tolist()
        if pA['clt'].find('EUR-11_MOHC-HadGEM2-ES_rcp85_r1i1p1_SMHI-RCA4_v1_day_2091') != -1:
            datesB = pd.date_range(pd.datetime(y1B, 1, 1), pd.datetime(y2B - 1, 12, 31), freq='d').tolist()
        elif pA['clt'].find('EUR-11_MOHC-HadGEM2-ES_rcp45_r1i1p1_SMHI-RCA4_v1_day_2091') != -1:
            datesB = pd.date_range(pd.datetime(y1B, 1, 1), pd.datetime(y2B - 1, 11, 30), freq='d').tolist()
        else:
            datesB = pd.date_range(pd.datetime(y1B, 1, 1), pd.datetime(y2B, 12, 31), freq='d').tolist()

        outdir = os.path.join(self.output_path,
                              'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (
                                  m1, exA, exB, m2, y1A, y2B))

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # creation of empty files before risks of bug: MC 2018-07-10

        initp = os.path.join(outdir, '2020init')
        if os.path.exists(os.path.join(outdir, '2017par')):
            parp = os.path.join(outdir, '2020par')
        else:
            parp = os.path.join(outdir, '2020par')
        inputp = os.path.join(outdir, '2020input')
        if os.path.exists(os.path.join(outdir, 'Calibration_Complete.csv')):
            print('lake %s is already completed' % self.ebhex)
            # with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
            #    f.write ( 'lake %s is already completed\n' % ebhex )
            #    f.close ()
            ret = 0
        #else:
            # empty = pd.DataFrame(np.nan, index=np.arange(0,len(datesA+datesB)), columns=np.arange(1,int(depth)+1))
            # for i in ['Tzt.csv','O2zt.csv', 'Attn_zt.csv', 'Qst.csv', 'DOCzt.csv','lambdazt.csv']:
            #     empty.to_csv('%s/%s'%(outdir,i),na_rep='NA',header=False,index=False)
            # with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
            #     f.write ('empty files created\n')
            #     f.close ()

        self.mylakeinit(initp)
        self.mylakepar(parp)
        self.mylakeinput(pA, pB, datesA, datesB, inflowfilename, inputp)
        #     cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran_optimize(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\',%d);quit' % (
        #         initp, parp, inputp, y1A, y2B, outdir, 1)
        #     print(cmd)
        #     os.system(cmd)
        #     #     # for f in [initp, parp, inputp]:
        #     #     #    os.system ( 'bzip2 -f -k %s' % f )
        #     expectedfs = ['Tzt.csv', 'O2zt.csv', 'Attn_zt.csv', 'Qst.csv', 'DOCzt.csv', 'PARMaxt.csv', 'PARzt.csv',
        #                   'His.csv', 'lambdazt.csv']
        #     flags = [os.path.exists(os.path.join(outdir, f)) for f in expectedfs]
        #
        #     if all(flags):
        #         with open(os.path.join(outdir, '2020REDOCOMPLETE'), 'w') as f:
        #             f.write(datetime.datetime.now().isoformat())
        # #         ret = 0
        # #     ret = 0 if all(flags) else 100
        # # return ret

    def mylakeinit(self,outdir):
        """
            create a file of a lake initiated with a max_depth and area.
            Assumes to have a cone shaped bathymetry curve

            :return: string to be written to an init file of MyLake
        """
        # 5-7-2018 MC
        depth_resolution = 1  # metres. NOTE: don't change this unless you know what you are doing. Changing it here will
        #  not make mylake run with a higher depth resolution, it will only change the init data

        depth_levels = np.arange(0, self.depth, depth_resolution)
        if self.depth not in depth_levels:
            depth_levels = np.concatenate((depth_levels, np.array([self.depth])))  # a enlever
        areas = self.area * (depth_levels - self.depth) ** 2 / self.depth ** 2
        lines = [
            '\t'.join(
                [('%.2f' % d), ('%.0f' % a)] + ['4'] + ['0'] * 5 + ['%s' % (2000 * float(self.i_sc_doc))] + ['0'] * 5 + ['12000'] + [
                    '0'] * 15)  # MC 06-01-2018 add I_scDOC and initial 8000 become 2000#MC 06-29-2018 12000
            # Z, Az and T, ...., DOC, .... DO, ...
            for d, a in zip(depth_levels, areas)]
        # lines[0] = lines[0] + '\t0\t0'  # snow and ice, plus 16 dummies
        firstlines = '''-999	"MyLake init"
        Z (m)	Az (m2)	Tz (deg C)	Cz	Sz (kg/m3)	TPz (mg/m3)	DOPz (mg/m3)	Chlaz (mg/m3)	DOCz (mg/m3)	TPz_sed (mg/m3)	
        Chlaz_sed (mg/m3)	"Fvol_IM (m3/m3	 dry w.)"	Hice (m)	Hsnow (m)	DO	dummy	dummy	dummy	dummy	dummy	
        dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy'''

        lines = [firstlines] + lines

        with open(os.path.join(outdir), 'w') as f:
            f.write('\n'.join(lines))

    def mylakepar(self,outdir):
        """
        Creates the MyLake parameter file. If the file LAE_para_all1.txt is present, it will be used to prepare the
        parameters. Otherwise, the string in this function using the parameter's value from the class will be used.
        :return: None
        """

        if os.path.isfile(
                "LAE_para_all2.txt"):
            print('using file')
            with open("LAE_para_all2.txt", "r") as infile:
                out = infile.read() % (
                    self.latitude, self.longitude, self.kz_n0, self.c_shelter, self.alb_melt_ice,
                    self.alb_melt_snow,
                    self.i_scv, self.i_sct, self.i_sc_doc, self.swa_b0, self.swa_b1, self.k_bod, self.k_sod)



        else:
            out = '''-999	"Mylake parameters"			
                Parameter	Value	Min	Max	Unit
                dz	1.0	0.5	2	m
                Kz_ak	NaN	NaN	NaN	(-)
                Kz_ak_ice	0.000898	NaN	NaN	(-)
                Kz_N0	%s	NaN	NaN	s-2
                C_shelter	%s	NaN	NaN	(-)
                latitude	%s	NaN	NaN	dec.deg
                longitude	%s	NaN	NaN	dec.deg
                alb_melt_ice	%s	NaN	NaN	(-)
                alb_melt_snow	%s	NaN	NaN	(-)
                PAR_sat	3.00E-05	1.00E-05	1.00E-04	mol m-2 s-1
                f_par	0.45	NaN	NaN	(-)
                beta_chl	0.015	0.005	0.045	m2 mg-1
                lamgbda_I	5	NaN	NaN	m-1
                lambda_s	15	NaN	NaN	m-1
                sed_sld	0.36	NaN	NaN	(m3/m3)
                I_scV 	%s	NaN	NaN	(-)
                I_scT	%s	NaN	NaN	deg C
                I_scC	1	NaN	NaN	(-)
                I_scS	1	1.1	1.9	(-)
                I_scTP	1	0.4	0.8	(-)
                I_scDOP	1	NaN	NaN	(-)
                I_scChl	1	NaN	NaN	(-)
                I_scDOC	%s	NaN	NaN	(-)
                swa_b0	%s	NaN	NaN	m-1
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
                I_scDIC	2	NaN	NaN	(-)
                Mass_Ratio_C_Chl	100	NaN	NaN	(-)
                SS_C	0.25	NaN	NaN	 
                density_org_H_nc	1.95	NaN	NaN	 
                density_inorg_H_nc	2.65	NaN	NaN	 
                I_scO	1	NaN	NaN	(-)
            ''' %(
                self.kz_n0, self.c_shelter, self.latitude, self.longitude, self.alb_melt_ice,
                self.alb_melt_snow,
                self.i_scv, self.i_sct, self.i_sc_doc, self.swa_b0, self.swa_b1, self.k_bod, self.k_sod)

        outpath = outdir

        with open(outpath, 'w') as f:
            f.write(out)

        print("{} Done".format(outpath))

        return outpath


    def mylakeinput(self,pA, pB, datesA, datesB,inflowfile, outpath):
        """
        create a file containing the informations relatively to Mylake
        :param pA: dictionary of paths to HDF5 files
        :param pB: dictionary of paths to HDF5 files
        :param datesA: pandas.date_range of time y1A to y2A
        :param datesB: pandas.date_range of time y1B to y2B
        :param ebhex: ebhex number
        :param subid: Reference number
        :param inflowfile: filename of the inflowfile
        :param outpath: filename where a file of Mylake input will be written
        :type pA: dict
        :type pB: dict
        :type datesA: pandas.date_range
        :type datesB: pandas.date_range
        :type ebhex: str201
        :type subid: int
        :type inflowfile: str
        :type outpath: str
        :return: string to be written to a file
        """
        # 5-7-2018 MC

        for v in variables:  # 2018-08-03 MC modification to compensate if pr, ps or rsds are missing
            if v == 'pr' or v == 'ps' or v == 'rsds':
                if not os.path.isfile(pA[v]):
                    if os.path.isfile(pB[v]):
                        pA[v] = pB[v]
                else:
                    if not os.path.isfile(pB[v]):
                        pB[v] = pA[v]

        if pA['clt'].find('MOHC-HadGEM2-ES') != -1 and pA['clt'].find('r1i1p1_SMHI-RCA4_v1_day') != -1:  # 5-24-2018 MC
            dfmissingdata = pd.concat(
                [self.take5_missingdata(pA, datesA, self.ebhex), self.take5_missingdata(pB, datesB, self.ebhex)])  # 5-24-2017 MC
            if datesB[-1] == pd.datetime(2099, 11, 30):  # 2018-08-01 MC
                dfmissingdata = dfmissingdata[:-31]
            df = dfmissingdata.interpolate()  # 5-24-2018 MC
        else:
            df = pd.concat([self.take5(pA, datesA, self.ebhex), self.take5(pB, datesB, self.ebhex)])

        ndays = len(datesA) + len(datesB)
        calibration_time = (datesA[0]-pd.datetime(datesA[0].year-8, datesA[0].month, datesA[0].day)).days

        df.index = np.arange(ndays)
        dflow = self.inflow5(inflowfile, datesA + datesB, self.subid)
        # repd = [datesA[0] + datetime.timedelta(d) for d in range(-(365 * 2), ndays)] # old version MC
        repd = [datesA[0] + datetime.timedelta(d) for d in range(-(calibration_time), ndays)]
        mlyear = np.array([d.year for d in repd])
        mlmonth = np.array([d.month for d in repd])
        mlday = np.array([d.day for d in repd])
        # mlndays = 366 + 365 + ndays # odl version MC
        mlndays = calibration_time + ndays
        # repeati = list(range(366)) + list(range(365)) + list(range(ndays)) # old version MC
        repeati = list(range(calibration_time)) + list(range(ndays))
        spacer = np.repeat([0], repeats=ndays)[repeati].reshape((mlndays, 1))
        # stream_Q = np.repeat([2000], repeats = ndays)[repeati].reshape((mlndays, 1))
        # stream_T = np.repeat([10], repeats = ndays)[repeati].reshape((mlndays, 1))
        stream_O = np.repeat([12000], repeats=ndays)[repeati].reshape(
            (mlndays, 1))  # MC 06-01-2018 initial parameters stream_O:8000
        stream_C = np.repeat([0.5], repeats=ndays)[repeati].reshape((mlndays, 1))
        # stream_TP = np.repeat([5], repeats = ndays)[repeati].reshape((mlndays, 1))
        # stream_DOP = np.repeat([1], repeats = ndays)[repeati].reshape((mlndays, 1))
        stream_SS = np.repeat([0.01], repeats=ndays)[repeati].reshape((mlndays, 1))
        stream_Chl = np.repeat([0.01], repeats=ndays)[repeati].reshape((mlndays, 1))
        stream_DOC = np.repeat([2000], repeats=ndays)[repeati].reshape(
            (mlndays, 1))  # MC 06-01-2018 initial parameters 8000
        stream_DIC = np.repeat([20000], repeats=ndays)[repeati].reshape((mlndays, 1))
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
                                   # np.repeat([0], repeats = ndays)[repeati].reshape((mlndays, 1)),
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
                                   spacer, spacer), axis=1),  # SiO2, diatom
                   fmt=['%i', '%i', '%i',  # yy mm dd
                        '%.4g', '%.2f', '%.2f', '%i', '%i', '%.2f', '%.3f',  # rad, cloud, temp, hum, pres, wind, precip
                        '%.3f', '%.3f', '%.3f', '%.3f',  #
                        '%.3f', '%.3f', '%.3f', '%.3f',  #
                        '%.3f', '%.3f', '%i', '%i',  #
                        '%i', '%i', '%i', '%i',  #
                        '%i', '%i', '%i', '%i',  #
                        '%i', '%i'],  #
                   delimiter='\t',
                   header='mylake input\nYear	Month	Day	Global_rad (MJ/m2)	Cloud_cov (-)	Air_temp (deg C)	Relat_hum (%)	Air_press (hPa)	Wind_speed (m/s)	Precipitation (mm/day)	Inflow (m3/day)	Inflow_T (deg C)	Inflow_C	Inflow_S (kg/m3)	Inflow_TP (mg/m3)	Inflow_DOP (mg/m3)	Inflow_Chla (mg/m3)	Inflow_DOC (mg/m3)	DIC	DO	NO3	NH4	SO4	Fe2	Ca	pH	CH4	Fe3	Al3	SiO4	SiO2	diatom')
        with open(temporarypath) as f:
            with open(outpath, 'w') as g:
                g.write(f.read().replace('-99999999', 'NaN'))
        os.unlink(temporarypath)

    def take5(self,pdict, dates, ebhex):
        """
        Create a dataFrame containing the values predicted of (clt,hurs,pr,ps,rsds,sfcWind,tas) for each dates
        :param pdict: dictionary of paths to HDF5 files (see pA and pB)
        :param dates: see datesA and datesB
        :param ebhex: ebhex number
        :type pdict: dict
        :type dates: pandas.DatetimeIndex
        :type ebhex: str
        :return: pandas.DataFrame
        """
        # 5-7-2018 MC

        e = ebhex.lstrip('0x').lstrip('0')
        df = pd.DataFrame(dates, columns=['date'])
        try:
            df['clt'] = h5py.File(pdict['clt'], mode='r')[e][:] * 0.01
        except:
            df['clt'] = 0.65  # 2018-08-02 MC Mean value found in literature
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

    def take5_missingdata(self,pdict, dates, ebhex):  # 5-24-2018 MC
        """
        Create a dataFrame containing the values predicted of (clt,hurs,pr,ps,rsds,sfcWind,tas) for each dates
        :param pdict: dictionary of paths to HDF5 files (see pA and pB)
        :param dates: see datesA and datesB
        :param ebhex: ebhex number
        :type pdict: dict
        :type dates: pandas.DatetimeIndex
        :type ebhex: str
        :return: pandas.DataFrame
        """
        # 5-7-2018 MC
        e = ebhex.lstrip('0x').lstrip('0')
        test = len(pdict)
        special = False
        if str(dates[-1].year) == '2099':
            df = pd.DataFrame(index=range(1440), columns=['clt', 'hurs', 'pr', 'ps', 'rsds', 'sfcWind', 'tas'])
            special = True
        else:
            df = pd.DataFrame(index=range(1800), columns=['clt', 'hurs', 'pr', 'ps', 'rsds', 'sfcWind', 'tas'])
        # 2018-08-02 MC add try to clt and hurs to compensate when the variables are missing
        try:
            df['clt'] = h5py.File(pdict['clt'], mode='r')[e][:] * 0.01
        except:
            df['clt'] = 0.65  # 2018-08-02 MC Mean value found in literature
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
                step = int((len(dates) - len(df)) / 4)
            else:
                step = int((len(dates) - len(df)) / 5)
            leapyear = int(str(dates[-1])[0:4])

            for i in dates:
                if str(i)[5:10] == '02-29':
                    leapyear = int(str(i)[0:4])

            beforeleap = leapyear - int(str(dates[0])[0:4])
            row = -1
            time = beforeleap * 365
            for i in np.arange((365 / step) + row, time, (365 / step)):  # year/years before leap
                emptyrow = pd.DataFrame(
                    {'clt': np.nan, 'hurs': np.nan, 'pr': np.nan, 'ps': np.nan, 'rsds': np.nan, 'sfcWind': np.nan,
                     'tas': np.nan}, index=[i])
                df = pd.concat([df.ix[:i - 1], emptyrow, df.ix[i:]]).reset_index(drop=True)
            row = row + time
            time = 366
            for i in np.arange((366 / (step + 1) + row), row + time + 1, (366 / (step + 1))):  # leap year
                emptyrow = pd.DataFrame(
                    {'clt': np.nan, 'hurs': np.nan, 'pr': np.nan, 'ps': np.nan, 'rsds': np.nan, 'sfcWind': np.nan,
                     'tas': np.nan}, index=[i])
                df = pd.concat([df.ix[:i - 1], emptyrow, df.ix[i:]]).reset_index(drop=True)
            row = row + 366
            time = (4 - beforeleap) * 365
            for i in np.arange((365 / step) + row, row + time + 1, (365 / step)):  # year/years after leap
                emptyrow = pd.DataFrame(
                    {'clt': np.nan, 'hurs': np.nan, 'pr': np.nan, 'ps': np.nan, 'rsds': np.nan, 'sfcWind': np.nan,
                     'tas': np.nan}, index=[i])
                df = pd.concat([df.ix[:i - 1], emptyrow, df.ix[i:]]).reset_index(drop=True)
        dfinal = pd.DataFrame(dates, columns=['date'])
        return pd.concat([dfinal, df], axis=1)

    def nbrleapyears(self,start, end):  # MC 2018-07-10
        """
        determine the number of leap years in the date range
        :param start: start year
        :param end: end year
        :return: number of leap year between the start and end years
        """
        nbryears = 0
        while start <= end:
            if (start % 4 == 0 and start % 100 != 0) or start % 400 == 0:
                nbryears += 1
            start += 1
        return nbryears

    def inflow5(self,filename, dates, subId):
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
        # 5-7-2018 MC
        # MC 2018-07-10 Only if inflow of 2001-2010 is used for other date range. Once inflow will be choose in function of date range, this part will need to be modified
        if self.nbrleapyears(int(str(dates[0])[0:4]), int(str(dates[-1])[0:4])) != 2 \
                or dates[-1].year - dates[0].year != 9:  # 2018-08-30 in case time range is not 10 years
            d = pd.DataFrame(pd.date_range(pd.datetime(2001, 1, 1), pd.datetime(2010, 12, 31), freq='d').tolist(),
                             columns=['date'])
            d['Q'] = h5py.File(filename, mode='r')['%d/Q' % subId][:]
            d['T'] = h5py.File(filename, mode='r')['%d/T' % subId][:]
            d['TP'] = h5py.File(filename, mode='r')['%d/TP' % subId][:]
            d['DOP'] = h5py.File(filename, mode='r')['%d/DOP' % subId][:]

            dflow = pd.DataFrame(dates, columns=['date'])
            dflow.loc[:, 'Q'] = d.loc[:, 'Q']
            dflow.loc[3652, 'Q'] = d.loc[3651, 'Q']
            dflow.loc[:, 'T'] = d.loc[:, 'T']
            dflow.loc[3652, 'T'] = d.loc[3651, 'T']
            dflow.loc[:, 'TP'] = d.loc[:, 'TP']
            dflow.loc[3652, 'TP'] = d.loc[3651, 'TP']
            dflow.loc[:, 'DOP'] = d.loc[:, 'DOP']
            dflow.loc[3652, 'DOP'] = d.loc[3651, 'DOP']
            if str(dates[-1].year) == '2099':
                if str(dates[-1].month) == '11':
                    dflow = dflow[:-396]
                else:
                    dflow = dflow[:-365]

        else:
            dflow = pd.DataFrame(dates, columns=['date'])
            dflow['Q'] = h5py.File(filename, mode='r')['%d/Q' % subId][:]
            dflow['T'] = h5py.File(filename, mode='r')['%d/T' % subId][:]
            dflow['TP'] = h5py.File(filename, mode='r')['%d/TP' % subId][:]
            dflow['DOP'] = h5py.File(filename, mode='r')['%d/DOP' % subId][:]
            #dflow['T'] = np.nan #remove inflow temperature; issue for some lake with the ice covert formation.
        return dflow

    def performance_analysis(self):
        """
        Opens the comparison file created by make_comparison_file, and prints the results of analysis functions.
        :return: Score, a float representing the overall performance of the current simulation.
        """


        if os.path.exists("{}/Tztcompare.csv".format(self.calibration_path)):
            with open("{}/Tztcompare.csv".format(self.calibration_path), "r") as file:
                reader = list(csv.reader(file))

                date_list = []
                depth_list = []
                obs_list = []
                sims_list = []


            for item in reader[1:]:
                date_list.append(item[0])
                depth_list.append(item[1])
                obs_list.append(item[2])
                sims_list.append(item[3])


            sosT = sums_of_squares(obs_list, sims_list)

            rmseT,rmsenT = root_mean_square(obs_list, sims_list)

            r_sqT = r_squared(obs_list, sims_list)
        else:
            sosT = np.nan
            rmseT,rmsenT = np.nan,np.nan
            r_sqT = np.nan

        if os.path.exists("{}/O2ztcompare.csv".format(self.calibration_path)):
            with open("{}/O2ztcompare.csv".format(self.calibration_path), "r") as file:
                reader = list(csv.reader(file))

                date_list = []
                depth_list = []
                obs_list = []
                sims_list = []

            for item in reader[1:]:
                date_list.append(item[0])
                depth_list.append(item[1])
                obs_list.append(item[2])
                sims_list.append(item[3])

            sosO = sums_of_squares(obs_list, sims_list)

            rmseO,rmsenO = root_mean_square(obs_list, sims_list)

            r_sqO = r_squared(obs_list, sims_list)
        else:
            sosO = np.nan
            rmseO,rmsenO = np.nan,np.nan
            r_sqO = np.nan

        if os.path.exists("{}/Secchicompare.csv".format(self.calibration_path)):
            with open("{}/Secchicompare.csv".format(self.calibration_path), "r") as file:
                reader = list(csv.reader(file))

                date_list = []
                depth_list = []
                obs_list = []
                sims_list = []

            for item in reader[1:]:
                date_list.append(item[0])
                depth_list.append(item[1])
                obs_list.append(item[2])
                sims_list.append(item[3])

            sosS = sums_of_squares(obs_list, sims_list)

            rmseS,rmsenS = root_mean_square(obs_list, sims_list)

            r_sqS = r_squared(obs_list, sims_list)
        else:
            sosS = np.nan
            rmseS,rmsenS = np.nan,np.nan
            r_sqS = np.nan



        print("Analysis of {}.".format(self.lake_name))
        print("Sums of squares : {}, {}, {}".format(sosT,sosO,sosS))
        print("RMSE : {}, {}, {}".format(rmseT,rmseO,rmseS))
        print("R squared : {}, {}, {}".format(r_sqT,r_sqO,r_sqS))

        return [rmseT,rmseO,rmseS],[rmsenT,rmsenO,rmsenS], [r_sqT,r_sqO,r_sqS]

    def outputfile(self):
        """
        Function calculating all variables asked by ISIMIP and formates them into a txt file.
        :param y1: Initial year of the simulation
        :param y2: Final year of the simulation
        :param outfolder: folder directory to where the raw data produce by the model are.
        :return: None
        """
        y1,y2 = self.start_year-8, self.end_year
        outfolder = self.calibration_path
        try:
            dates = [d.strftime('%Y-%m-%d') for d in pd.date_range('%s-01-01' % y1, '%s-12-31' % y2)]
        except:
            dates = [d.strftime('%Y-%m-%d') for d in pd.period_range('%s-01-01' % y1, '%s-12-31' % y2, freq='D')]

        print("run output")

        if not os.path.exists(os.path.join(outfolder, "strat.csv")) or not os.path.exists(
                os.path.join(outfolder, "watertemp.csv")) \
                or not os.path.exists(os.path.join(outfolder, "thermodepth.csv")):
            with open("{}/Tzt.csv".format(outfolder), "r") as observation_file:
                try:
                    rows = list(csv.reader(observation_file))
                    strat = []
                    maxdensity = []
                    watertemp = []
                    for i in range(0, len(rows)):
                        data = rows[i]
                        row = ["%s, 00:00:00" % dates[i]]
                        waterrow = ["%s, 00:00:00" % dates[i]]
                        maxrowdensity = ["%s, 00:00:00" % dates[i]]

                        if abs(calculatedensity(float(data[0])) - calculatedensity(float(data[-1]))) > 0.1:
                            row.append(1)
                        else:
                            row.append(0)

                        strat.append(row)
                        density = []

                        for j in range(0, len(data)):
                            density.append(calculatedensity(float(data[j])))
                            waterrow.append(float(data[j]) + 273.15)

                        watertemp.append(waterrow)
                        maxrowdensity.append(density.index(max(density)))
                        maxdensity.append(maxrowdensity)
                    if not os.path.exists(os.path.join(outfolder, "strat.csv")):
                        try:
                            with open(os.path.join(outfolder, "strat.csv"), "w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerows(strat)
                        except:
                            if os.path.exists(os.path.join(outfolder, "strat.csv")):
                                os.remove(os.path.join(outfolder, "strat.csv"))
                    if not os.path.exists(os.path.join(outfolder, "watertemp.csv")):
                        try:
                            with open(os.path.join(outfolder, "watertemp.csv"), "w", newline="") as f2:
                                writer = csv.writer(f2)
                                writer.writerows(watertemp)
                        except:
                            if os.path.exists(os.path.join(outfolder, "watertemp.csv")):
                                os.remove(os.path.join(outfolder, "watertemp.csv"))

                    if not os.path.exists(os.path.join(outfolder, "thermodepth.csv")):
                        try:
                            with open(os.path.join(outfolder, "thermodepth.csv"), "w", newline="") as f1:
                                writer = csv.writer(f1)
                                writer.writerows(maxdensity)
                        except:
                            if os.path.exists(os.path.join(outfolder, "thermodepth.csv")):
                                os.remove(os.path.join(outfolder, "thermodepth.csv"))
                except:
                    print("error")

    def comparison_obs_sims(self,thermocline):
        """
        Opens the comparison file created by make_comparison_file, and prints the results of analysis functions.
        :return: Score, a float representing the overall performance of the current simulation.
        """


        if os.path.exists("{}/Tztcompare.csv".format(self.calibration_path)):
            variable = "Temperature"
            data = pd.read_csv("{}/Tztcompare.csv".format(self.calibration_path),header=None)
            data.columns = ['Dates', 'Depth','Observed %s (°C)'%variable, 'Modeled %s (°C)'%variable]
            all_data = pd.read_csv("{}/Tzt.csv".format(self.calibration_path),header=None)
            jj = ["%s" % (i) for i in np.arange(0.5,len(all_data),1)]
            all_data.columns = ["%s" % (i) for i in np.arange(0.5,len(all_data),1)]
            thermocline = data['Depth'].median()
            dataunder = data[data['Depth'] <= thermocline]
            dataover = data[data['Depth'] > thermocline]
            alldataover = []
            alldataunder = []
            depthunder = []
            depthover = []
            for i in data['Depth'].unique():
                if i <= thermocline:
                    depthunder.append(i)
                    if i in np.arange(0.5, len(all_data), 1):
                        dataa = all_data['%s' % i].tolist()
                    else:
                        if (round(i) - floor(i)) == 1:
                            dataa= [findYPoint(round(i) - 0.5, round(i), all_data.iloc[round(i) - 0.5, y],
                                        all_data.iloc[round(i), y], i) for y in range(1, len(all_data))]
                        else:
                            dataa = [findYPoint(round(i), round(i)+0.5, all_data.iloc[round(i) , y],
                                                all_data.iloc[round(i)+0.5, y], i) for y in range(1, len(all_data))]
                    alldataunder.append(dataa)
                else:
                    depthover.append(i)
                    if i in np.arange(0.5, len(all_data), 1):
                        dataa = all_data['%s' % i].tolist()
                    else:
                        if (round(i) - floor(i)) == 1:
                            dataa= [findYPoint(round(i) - 0.5, round(i), all_data.iloc[round(i) - 0.5, y],
                                        all_data.iloc[round(i), y], i) for y in range(1, len(all_data))]
                        else:
                            dataa = [findYPoint(round(i), round(i) + 0.5, all_data.iloc[round(i), y],
                                                all_data.iloc[round(i) + 0.5, y], i) for y in range(1, len(all_data))]
                    alldataover.append(dataa)




            colorpalette = sns.color_palette("colorblind")

            sns.set(font_scale=2)
            sns.set_style("ticks")

            plt.grid(False)
            plt.figure(figsize=(20, 10))
            #sns.lineplot(x="Dates", y='Observed %s (°C)' % variable, data=dataunder, color="black")
            for i in np.arange(0,len(alldataunder),1):
                sns.lineplot(x="Dates", y='Modeled %s (°C)' % variable, data=alldataunder[i], color=colorpalette[i])
                sns.scatterplot(x="Dates", y='Modeled %s (°C)' % variable, data=alldataunder[i], markers="-o",
                                color=colorpalette[i], label='Modeled %s(%s m)' % (variable, depthunder[i]))

            sns.scatterplot(x="Dates", y='Observed %s (°C)' % variable, data=dataunder,markers="-o-", color="black", label ='Observation %s (0-%s m)' % (variable,thermocline))

            plt.xticks(rotation=15)
            ax = plt.axes()
            ax.xaxis.set_major_locator(plt.MaxNLocator(15))
            plt.savefig("comparison_%s_%s_epi.png" % (self.lake_name, variable))
            plt.close()
            sns.set(font_scale=2)
            sns.set_style("ticks")
            plt.grid(False)
            plt.figure(figsize=(20, 10))
            #sns.lineplot(x="Dates", y='Observed %s (°C)' % variable, data=dataover, color="black")
            for i in np.arange(0,len(alldataover),1):
                sns.lineplot(x="Dates", y='Modeled %s (°C)' % variable, data=alldataover[i], color=colorpalette[i])
                sns.scatterplot(x="Dates", y='Modeled %s (°C)' % variable, data=alldataover[i], markers="-o",
                                color=colorpalette[i], label='Modeled %s(%s m)' % (variable, depthover[i]))

            sns.lineplot(x="Dates", y='Modeled %s (°C)' % variable, data=dataover, color=colorpalette[3])
            sns.scatterplot(x="Dates", y='Modeled %s (°C)' % variable, data=dataover, markers="-o",
                            color=colorpalette[3], label='Modeled %s (%s-%s °C)' % (variable, thermocline, self.depth))

            sns.scatterplot(x="Dates", y='Observed %s (°C)' % variable, data=dataover, markers="-o-", color="black",
                            label='Observed %s (%s-%s °C)' % (variable,thermocline,self.depth))


            plt.xticks(rotation=15)
            ax = plt.axes()
            ax.xaxis.set_major_locator(plt.MaxNLocator(15))
            plt.savefig("comparison_%s_%s_hypo.png" % (self.lake_name, variable))
            plt.close()

        if os.path.exists("{}/O2ztcompare.csv".format(self.calibration_path)):
            variable = "Oxygen"
            data = pd.read_csv("{}/O2ztcompare.csv".format(self.calibration_path), header=None)
            data.columns = ['Dates', 'Depth', 'Observed %s (mg/L)' % variable, 'Modeled %s (mg/L)' % variable]
            thermocline = data['Depth'].median()
            dataunder = data[data['Depth'] <= thermocline]
            dataover = data[data['Depth'] > thermocline]

            colorpalette = sns.color_palette("colorblind", 7)
            sns.set(font_scale=2)
            sns.set_style("ticks")

            plt.grid(False)
            plt.figure(figsize=(20, 10))
            sns.lineplot(x="Dates", y='Observed %s (mg/L)' % variable, data=dataunder, color="black")
            sns.lineplot(x="Dates", y='Modeled %s (mg/L)' % variable, data=dataunder, color=colorpalette[0])
            sns.scatterplot(x="Dates", y='Observed %s (mg/L)' % variable, data=dataunder, markers="-o-", color="black",
                            label='Observed %s (0-%s mg/L) ' % (variable,thermocline))

            sns.scatterplot(x="Dates", y='Modeled %s (mg/L)' % variable, data=dataunder, markers="-o",
                            color=colorpalette[0], label='Modeled %s (0-%s mg/L) ' % (variable,thermocline))
            plt.xticks(rotation=15)
            ax = plt.axes()
            ax.xaxis.set_major_locator(plt.MaxNLocator(15))
            plt.savefig("comparison_%s_%s_epi.png" % (self.lake_name, variable))
            plt.close()
            sns.set(font_scale=2)
            sns.set_style("ticks")

            plt.grid(False)
            plt.figure(figsize=(20, 10))
            sns.lineplot(x="Dates", y='Observed %s (mg/L)' % variable, data=dataover, color="black")
            sns.lineplot(x="Dates", y='Modeled %s (mg/L)' % variable, data=dataover, color=colorpalette[3])
            sns.scatterplot(x="Dates", y='Observed %s (mg/L)' % variable, data=dataover, markers="-o-", color="black",
                            label='Observed %s (%s-%s mg/L) ' % (variable,thermocline,self.depth))

            sns.scatterplot(x="Dates", y='Modeled %s (mg/L)' % variable, data=dataover, markers="-o",
                            color=colorpalette[3], label='Modeled %s (%s-%s mg/L) ' % (variable,thermocline,self.depth))

            plt.xticks(rotation=15)
            ax = plt.axes()
            ax.xaxis.set_major_locator(plt.MaxNLocator(15))
            plt.savefig("comparison_%s_%s_hypo.png" % (self.lake_name, variable))

            plt.close()





def sums_of_squares(obs_list, sims_list):
    """
    Finds the sums of squares for all temperatures listed in the comparison file.
    :param obs_list: A list of observed temperatures.
    :param sims_list: A list of simulated temperatures.
    :return: The result of the sums of square as a float.
    """
    sums = 0
    for x in range(len(obs_list)):
        if obs_list[x] == 'None':
            continue
        sums += (float(obs_list[x]) - float(sims_list[x])) ** 2

    return sums


def root_mean_square(obs_list, sims_list):
    """
    Finds the root_mean_square for the temperatures listed in the comparison file.
    :param obs_list: A list of observed temperatures.
    :param sims_list: A list of simulated temperatures.
    :return: The result of the root mean square as a float.
    """
    d = pd.DataFrame(list(zip(obs_list, sims_list)),
               columns =['obs', 'sim'])

    d["obs"] = d["obs"].astype(float)
    d["sim"] = d["sim"].astype(float)
    try:
        results = mean_squared_error(d["obs"],d["sim"])
        results = sqrt(results)
        resultsnormalise = sqrt(mean_squared_error(d["obs"],d["sim"]))/(max(d["obs"])-min(d["obs"]))
    except:
        results = np.nan
        resultsnormalise = np.nan
    return results,resultsnormalise


def r_squared(obs_list, sims_list):
    """
    Find the R squared for the simulations compared to the expected observations
    :param obs_list: A list of observed temperatures.
    :param sims_list: A list of simulated temperatures.
    :return: results of R squared, as a float
    """

    x = []
    y = []

    for i in obs_list:
        try:

            x.append(float(i))
            y.append(float(sims_list[obs_list.index(i)]))

        except ValueError:
            continue
        except IndexError:
            break
    try:
        rsquare = r2_score(x,y)
    except:
        rsquare = np.nan
    return rsquare


def standard_deviation(obs_list):
    """
    Find the standard deviation of the observations
    :param obs_list: Type list. The list of observed temperatures
    :return: The standard deviation of obs_list
    """
    observations = []
    for obs in obs_list:
        try:
            observations.append(float(obs))
        except ValueError:
            continue

    return statistics.stdev(observations)


def rmse_by_sd(obs_list, rmse):
    """
    Divides RMSE of the simulations by the SD of the observations
    :param obs_list: A list of observed temperatures.
    :param rmse: Float
    :return: A float, RMSE / SD
    """
    try:
        results = rmse / standard_deviation(obs_list)
    except ZeroDivisionError:
        results = "Error_Zero_Division"
    return results


def calculatedensity(temp):
    """
    Calculate density by using equation giving in ISIMIP lake model protocol.
    :param temp: average daily temperature at a specific layer.
    :return density: Water density for the specific layer.
    """
    t = temp
    density = (999.842594 + (6.793952e-2 * t) - (9.095290e-3 * t ** 2) +
               (1.001685e-4 * t ** 3) - (1.120083e-6 * t ** 4) + (6.536336e-9 * t ** 5))
    return density



def graphique(x,y,xerr,yerr,r_value,slope,intercept):
    sns.set(font_scale=2)
    sns.set_style("ticks")

    plt.grid(False)

    colorpalette = sns.color_palette("dark", 10)
    lineStart = 0
    lineEnd = 14
    fig, ax = plt.subplots(figsize=(15.0, 14.5))
    plt.plot ( [lineStart, lineEnd], [lineStart, lineEnd], 'k-', color=colorpalette[9],label="y= x",linewidth=4 )
    plt.xlim(0, 14)
    plt.ylim(0, 14)
    (_, caps, _)=plt.errorbar ( x,y, xerr=xerr, yerr=yerr, fmt='o',color=colorpalette[3],markersize=8, capsize=20, linewidth= 4,elinewidth=4 )
    for cap in caps:
        cap.set_markeredgewidth ( 4 )

    fig.suptitle("")
    fig.tight_layout(pad=2)



    x = sm.add_constant(x) # constant intercept term
    # Model: y ~ x + c
    model = sm.OLS(y, x)
    fitted = model.fit()
    x_pred = np.linspace(x.min(), x.max(), 50)
    x_pred2 = sm.add_constant(x_pred)
    y_pred = fitted.predict(x_pred2)

    ax.plot(x_pred, y_pred, '-', color='k', linewidth=4,label="linear regression (y = %0.3f x + %0.3f) \n R\u00b2 : %0.3f "%(slope,intercept,r_value))


    print(fitted.params)     # the estimated parameters for the regression line
    print(fitted.summary())  # summary statistics for the regression

    y_hat = fitted.predict(x) # x is an array from line 12 above
    y_err = y - y_hat
    mean_x = x.T[1].mean()
    n = len(x)
    dof = n - fitted.df_model - 1

    t = stats.t.ppf(1-0.025, df=dof)
    s_err = np.sum(np.power(y_err, 2))
    conf = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((x_pred-mean_x),2)/((np.sum(np.power(x_pred,2))) - n*(np.power(mean_x,2))))))
    upper = y_pred + abs(conf)
    lower = y_pred - abs(conf)
    ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.4,label="Confidence interval")


    sdev, lower, upper = wls_prediction_std(fitted, exog=x_pred2, alpha=0.025)
    ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.1,label="Prediction interval")
    plt.xlabel ( "Average Observed Secchi_Depth (m)" )
    plt.ylabel ( "Average Modeled Secchi Depth (m)" )
    plt.ylim ()
    plt.xlim()

    ax.legend (loc='lower right')
    fig.savefig('Secchi_mean_comparison.png', dpi=125)
    plt.close()


def graphiqueTO(x,y,z,symbol,r_value,slope,intercept,variable):
    sns.set(font_scale=2)
    sns.set_style("ticks")

    rmse,nrmse = root_mean_square([item for sublist in x for item in sublist],[item for sublist in y for item in sublist])
    plt.grid(False)

    colorpalette = sns.color_palette("dark", 10)
    lineStart = 0
    if variable == "Temperature (°C)":
        lineEnd = 28
    else:
        lineEnd = 28
    fig, ax = plt.subplots(figsize=(15.0, 14.5))

    if variable == "Temperature (°C)":
        plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color=colorpalette[0], label="y= x", linewidth=4)
        plt.xlim(-1,28)
        plt.ylim(-1,28)
        ccmap = 'Blues'
    else:
        plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color=colorpalette[3], label="y= x", linewidth=4)
        plt.xlim(-1,20)
        plt.ylim(-1,28)
        ccmap = 'Reds'
    markers = [ "o", "v", "^","s", "P","*","+","X","D","1","p","d"]
    for i, c in enumerate(np.unique(symbol)):
        cs = plt.scatter(x[i], y[i], c=z[i], marker=markers[i], s=90,cmap=ccmap, linewidths=1, edgecolors = 'k',alpha=0.8)
    cb = plt.colorbar(cs)
    cb.ax.tick_params(labelsize=14)


    fig.suptitle("")
    fig.tight_layout(pad=2)


    x,y =[item for sublist in x for item in sublist],[item for sublist in y for item in sublist]
    x = sm.add_constant(x) # constant intercept term
    # Model: y ~ x + c
    model = sm.OLS(y, x)
    fitted = model.fit()
    x_pred = np.linspace(x.min(), x.max(), 50)
    x_pred2 = sm.add_constant(x_pred)
    y_pred = fitted.predict(x_pred2)

    ax.plot(x_pred, y_pred, '-', color='k', linewidth=4,label="linear regression (y = %0.3f x + %0.3f) \n R\u00b2 : %0.3f RMSE: %0.3f"%(slope,intercept,r_value,rmse))


    print(fitted.params)     # the estimated parameters for the regression line
    print(fitted.summary())  # summary statistics for the regression

    y_hat = fitted.predict(x) # x is an array from line 12 above
    y_err = y - y_hat
    mean_x = x.T[1].mean()
    n = len(x)
    dof = n - fitted.df_model - 1

    t = stats.t.ppf(1-0.025, df=dof)
    s_err = np.sum(np.power(y_err, 2))
    conf = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((x_pred-mean_x),2)/((np.sum(np.power(x_pred,2))) - n*(np.power(mean_x,2))))))
    upper = y_pred + abs(conf)
    lower = y_pred - abs(conf)
    ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.4,label="Confidence interval")


    sdev, lower, upper = wls_prediction_std(fitted, exog=x_pred2, alpha=0.025)
    ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.1,label="Prediction interval")
    plt.xlabel ( "Average Observed %s" %variable)
    plt.ylabel ( "Average Modeled %s"%variable )
    plt.ylim ()
    plt.xlim()

    ax.legend (loc='upper left')
    if variable == "Temperature (°C)":
        fig.savefig('Temperature_comparison.png', dpi=125)
    else:
        fig.savefig('Oxygen_comparison.png', dpi=125)
    plt.close()

def findYPoint(xa, xb, ya, yb, xc):
    """
    Function used to calculate the temperature at a depth non simulated by the model. MyLake simulates the temperature
     at each meter (atarting at 0.5) and this function permit to comparer the temperature at the same depth that it has
      been measured.
    :param xa: Closest depth simulated below the wanted depth.
    :param xb: Closest depth simulated over the wanted depth.
    :param ya: Temperature at the depth xa.
    :param yb: Temperature at the depth xb.
    :param xc: Depth at which the temparature is wanted
    :return: Temperature at the depth yc.
    """
    m = (float(ya) - float(yb)) / (float(xa) - float(xb))
    yc = (float(xc) - (float(xb) + 0.5)) * m + float(yb)
    return yc


if __name__ == "__main__":
    lake_info = LakeInfo("test")

