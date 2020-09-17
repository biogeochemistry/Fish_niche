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

cordexfolder = '..\cordex' #5-24-2018 MC #Need to be change depending where the climatic files where
inflowfolder = r'C:\Users\macot620\Documents\GitHub\Fish_niche\sweden_inflow_data'
outputfolder = '../output' #5-9-2018 MC
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

    def __init__(self, lake_name, lake_id, subid, ebhex, area, depth, longitude, latitude, volume):
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
        self.obervation_file = pd.ExcelFile(os.path.join(inflowfolder,"Validation_data_for_lookup.xlsx"))

        ebhex = ebhex[2:] if ebhex[:2] == '0x' else ebhex
        while len(ebhex) < 6:
            ebhex = '0' + ebhex
        d1, d2, d3 = ebhex[:2], ebhex[:4], ebhex[:6]
        outdir = os.path.join(outputfolder, d1, d2, d3)

        self.output_path = outdir
        self.start_year = 1971
        self.end_year = 1980
        self.calibration_path = os.path.join(self.output_path, "EUR-11_ICHEC-EC-EARTH_historical-historical_r3i1p1_DMI-HIRHAM5_v1_day_19710101-19801231")

        if os.path.exists(os.path.join(self.calibration_path, "2020_par")):
            with open(os.path.join(self.calibration_path, "2020_par")) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(sep="	")
                    line[0] = line[0].replace(" ", "")
                    if line[0] == "swa_b1":
                        self.swa_b1 = line[1]

                    elif line[0] == "C_shelter":
                        self.c_shelter = line[1]

                    elif line[0] == "Kz_N0":
                        self.kz_n0 = line[1]

                    elif line[0] == "alb_melt_ice":
                        self.alb_melt_ice = line[1]

                    elif line[0] == "alb_melt_snow":
                        self.alb_melt_snow = line[1]

                    elif line[0] == "swa_b0":
                        self.swa_b0 = line[1]

                    elif line[0] == "K_SOD":
                        self.k_sod = line[1]
                        break

                    elif line[0] == "I_scV":
                        self.k_bod = line[1]
                    elif line[0] == "K_SOD":
                        self.k_sod = line[1]
                        break

                    elif line[0] == "I_scT":
                        self.k_bod = line[1]
        else:
            self.kz_n0 = 7.00E-05
            self.c_shelter = "NaN"
            self.alb_melt_ice = 0.6
            self.alb_melt_snow = 0.9
            self.i_scv = 1.15
            self.i_sct = 0
            self.swa_b0 = 2.5
            self.swa_b1 = 1
            self.k_bod = 0.1
            self.k_sod = 500
            self.i_sc_doc = 1


    def initiate_init(self, date_init=101):
        """
        J. Bellavance 2018/11/19
        For ISI-MIP
        Opens hypsometry and temperature csv files for a lake in the observations directory. Obtains the depth levels,
        observed bathymetric area for each level, and the first observed mean temperature for each level.
        It also prepares the output path for the init file.

        :param: date_init : Type int. By default = 0101, corresponding at the month and day of the starting date of
        the init file.

        :return: class InitInfo: class containing the information needed to create the file containing the initial
        state of the model.
        """
        if os.path.exists("{}/{}_hypsometry_modified.csv".format(self.observation_path, self.lake_name)):
            with open("{}/{}_hypsometry_modified.csv".format(self.observation_path, self.lake_name), "r") as obs:
                reader = list(csv.reader(obs))[1:]
                depth_levels = []
                areas = []

                for row in reader:
                    depth_levels.append((float(row[2])))
                    areas.append(float(row[3]))

            if os.path.exists("{}/{}_temp_daily.csv".format(self.observation_path, self.lake_name)):

                with open("{}/{}_temp_daily.csv".format(self.observation_path, self.lake_name), "r") as obs:
                    reader = list(csv.reader(obs))[1:]

                w_temp = find_init_temp_daily(reader, depth_levels)

            else:
                found_date = False
                for file in os.listdir(self.observation_path):
                    with open("{}/{}".format(self.observation_path, file), "r") as obs:

                        reader = list(csv.reader(obs))[1:]

                    for observation in reader:
                        if int(observation[2][4:8]) > date_init and int(
                                observation[2][4:8]) < date_init + 20 and found_date is False:
                            found_date = True

                    if found_date is True:
                        break

                w_temp = find_init_temp_subdaily(reader, depth_levels,
                                                 date_init)  # need to be modified to included spin off and last
                # temperature simulated by other temperature

            if not os.path.exists(self.input_path):
                os.makedirs(self.input_path)

            return InitInfo(self.lake_name, depth_levels, areas, w_temp)
        else:
            print("{} doesn't have hypsometry")

    def initiate_par(self, kz_n0=0.00007, c_shelter="NaN", alb_melt_ice=0.6, alb_melt_snow=0.9, i_scv=1.15, i_sct=0,
                     swa_b0=2.5, swa_b1=1, k_bod=0.1, k_sod=500, i_sc_doc=1):
        """
        Function initiating the class ParInfo. Give to the class the value for each parameters needed to be
        calibrating (the default value is comming from the calibration that have been done on the lake Langtjern from
        previous analysis)

       :param kz_n0: Type float: by default 0.00007. Min. stability frequency (s-2)
       :param c_shelter: Type float: by default empty. Wind shelter parameter (-)
       :param alb_melt_ice: Type float: by default 0.6. Albedo of melting ice (-)
       :param alb_melt_snow:Type float: by default 0.9. Albedo of melting ice (-)
       :param i_scv:Type float: by default 1.15. Scaling factor for inflow volume (-)
       :param i_sct:Type float: by default 0. Scaling coefficient for inflow temperature (-)
       :param swa_b0:Type float: by default 2.5. Non-PAR light atteneuation coefficient (m-1).
       :param swa_b1:Type float: by default 1. PAR light atteneuation coefficient (m-1).
       :param k_bod:Type float: by default 0.1. Organic decomposition rate (1/d)
       :param k_sod:Type float: by default 500. Sedimentary oxygen demand (mg m-2 d-1).
       :param i_sc_doc:Type float: by default 1. Scaling factor for inflow concentration of DOC  (-)
       :return: class ParInfo.
       """

        return ParInfo(self.lake_name, kz_n0, c_shelter, alb_melt_ice, alb_melt_snow, i_scv,
                       i_sct, swa_b0, swa_b1, k_bod, k_sod, i_sc_doc)

    def initiate_input(self, model, scenario):
        """
        initiale the file containing the file containing the value in input.
        :param model: Climatic model used to generate the input values
        :param scenario: Climatic scenario used to generate the input values
        :return: class InputInfo
        """
        return InputInfo(self.lake_name, model, scenario)

    def generate_input_files(self, model="EWEMBI", scenario="historical", kz_n0=0.00007, c_shelter="NaN",
                             alb_melt_ice=0.6, alb_melt_snow=0.9, i_scv=1.15, i_sct=0, swa_b0=2.5, swa_b1=1,
                             k_bod=0.1,
                             k_sod=500, i_sc_doc=1):
        """
        Creates all files needed for a run of mylake model with a single lake. The input function will generate all
        needed input files(one for each combination of climatic model and scenario)
        :param model: Type string. model name
        :param scenario: Type string. scenario name
        :param kz_n0: Type float: by default 0.00007. Min. stability frequency (s-2)
        :param c_shelter: Type float: by default empty. Wind shelter parameter (-)
        :param alb_melt_ice: Type float: by default 0.6. Albedo of melting ice (-)
        :param alb_melt_snow:Type float: by default 0.9. Albedo of melting ice (-)
        :param i_scv:Type float: by default 1.15. Scaling factor for inflow volume (-)
        :param i_sct:Type float: by default 0. Scaling coefficient for inflow temperature (-)
        :param swa_b0:Type float: by default 2.5. Non-PAR light atteneuation coefficient (m-1).
        :param swa_b1:Type float: by default 1. PAR light atteneuation coefficient (m-1).
        :param k_bod:Type float: by default 0.1. Organic decomposition rate (1/d)
        :param k_sod:Type float: by default 500. Sedimentary oxygen demand (mg m-2 d-1).
        :param i_sc_doc:Type float: by default 1. Scaling factor for inflow concentration of DOC  (-)
        :return: None
        """

        self.initiate_init().init_file()

        if not os.path.exists(r"D:\output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(self.region,
                                                                                                   self.lake_name)):
            self.initiate_par(kz_n0, c_shelter, alb_melt_ice, alb_melt_snow, i_scv, i_sct, swa_b0, swa_b1, k_bod,
                              k_sod,
                              i_sc_doc).mylakepar()
        else:
            print(r"D:\output/{}/{}/EWEMBI/historical/Calibration_Complete.txt already exists".format(self.region,
                                                                                                      self.lake_name))
        self.initiate_input(model, scenario).mylakeinput()

    def run_mylake(self, modelid, scenarioid, flag=None):
        """
        Runs the MyLake simulation using the input, init, and parameter files.
        Makes a single run for a combination of lake, model, and scenario.
        :param modelid: string. model used
        :param scenarioid: string. scenario used
        :param flag: None or string, determines if the run is for calibration or simulation.
        Should be set to None or "calibration".

        :return: None
        """

        prefix = self.prefix
        print(self.input_path, "{}_init".format(prefix))

        init_file = os.path.join(self.input_path, "{}_init".format(prefix))
        parameter_file = os.path.join(self.input_path, "{}_par".format(prefix))
        input_file = os.path.join(self.input_path, r"{}_{}_{}_input".format(prefix, modelid, scenarioid))
        outfolder = os.path.join(self.output_path, modelid, scenarioid)
        expectedfs = ['Tzt.csv', 'O2zt.csv', 'Attn_zt.csv', 'Qst.csv', 'DOCzt.csv', 'lambdazt.csv', 'Kzt.csv',
                      'Qzt_sed.csv']
        expectedfsvariables = ["strat.csv", "watertemp.csv", "thermodepth.csv", "ice.csv", "icetick.csv",
                               "snowtick.csv", "sensheatf.csv", "latentheatf.csv", "lakeheatf.csv", "albedo.csv",
                               "turbdiffheat.csv", "sedheatf.csv"]
        ret = False

        if flag == "calibration":  # Loop to run for calibrating the model
            if not os.path.exists(os.path.join(outfolder, 'RunComplete1')):

                # Set the period of simulation

                y1, y2 = self.start_year, self.end_year # possible period for the calibration analysis

                if not os.path.exists(outfolder):
                    os.makedirs(outfolder)

                # Call matlab and lauch the simulation
                cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\');quit' % \
                      (init_file, parameter_file, input_file, y1, y2, outfolder)
                print(cmd)
                os.system(cmd)

                # Validation of the creation of all files by the model

                flags = [os.path.exists(os.path.join(outfolder, f)) for f in expectedfs]
                if all(flags):
                    ret = True
                    with open(os.path.join(outfolder, 'RunComplete'), 'w') as f:
                        f.write(datetime.datetime.now().isoformat())

            else:
                ret = True

        else:  # Loop to run the model to produce the wanted variables
            y1, y2 = simulation_years(modelid, scenarioid)
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)

            if y2 - y1 > 100:
                alldate1 = 0
                alldate = 0
                years = [y1]
                if y1 == 1971:
                    yinit = 1980
                else:
                    yinit = y1 + 100
                years = years + (list(range(yinit, y2, 100)))
                years.append(y2)
                all_files = []
                if not os.path.exists(os.path.join(outfolder, 'RunComplete2')):
                    for i in range(0, len(years) - 1):
                        alldate += 1

                        if i + 1 != len(years) - 1:
                            yinit = years[i]
                            yend = years[i + 1] - 1

                            print(yinit, yend)
                        else:
                            yinit = years[i]
                            yend = years[i + 1]
                            print(yinit, yend)
                        outfolder2 = os.path.join(outfolder, "%s_%s" % (yinit, yend))
                        all_files.append(outfolder2)
                        if not os.path.exists(outfolder2):
                            os.makedirs(outfolder2)
                        if not os.path.exists(os.path.join(outfolder2, 'RunComplete1')):

                            if not os.path.exists(os.path.join(outfolder2, 'RunComplete')):
                                cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,' \
                                      '\'%s\');quit' % (init_file, parameter_file, input_file, yinit, yend, outfolder2)
                                print(cmd)

                                os.system(cmd)

                            flags = [os.path.exists(os.path.join(outfolder2, f)) for f in expectedfs]

                            if all(flags):
                                outputfile(yinit, yend, outfolder2)
                                with open(os.path.join(outfolder2, 'RunComplete'), 'w') as f:
                                    f.write(datetime.datetime.now().isoformat())

                            flags = [os.path.exists(os.path.join(outfolder2, f)) for f in expectedfsvariables]

                            if all(flags):
                                with open(os.path.join(outfolder2, 'RunComplete1'), 'w') as f:
                                    f.write(datetime.datetime.now().isoformat())
                                alldate1 += 1
                            else:
                                output = "incomplete_%s_%s" % (yinit, yend)
                                for variable in np.arange(0, len(flags)):
                                    if flags[variable] is False:
                                        output = output + '_' + expectedfs[i]

                                with open(os.path.join(outfolder, output), 'w') as f:
                                    f.write(datetime.datetime.now().isoformat())

                        else:
                            alldate1 += 1

                    if alldate == alldate1:
                        ret = True
                        with open(os.path.join(outfolder, 'RunComplete2'), 'w') as f:
                            f.write(datetime.datetime.now().isoformat())
                    else:
                        with open(os.path.join(outfolder, 'Incomplete'), 'w') as f:
                            f.write(datetime.datetime.now().isoformat())

                else:
                    ret = True

            else:
                if not os.path.exists(os.path.join(outfolder, 'RunComplete1')):
                    if not os.path.exists(os.path.join(outfolder, 'RunComplete')):
                        cmd = 'matlab -wait -r -nosplash -nodesktop MyLake_optimizer(%d,%d,\'%s\',\'%s\',\'%s\',\'%s\',%d,%d);' \
                              'quit' % (self.start_year, self.end_year, parameter_file, input_file, init_file, outfolder, self.latitude, self.longitude)
                        print(cmd)

                        os.system(cmd)

                    flags = [os.path.exists(os.path.join(outfolder, f)) for f in expectedfs]

                    if all(flags):
                        outputfile(y1, y2, outfolder)
                        with open(os.path.join(outfolder, 'RunComplete'), 'w') as f:
                            f.write(datetime.datetime.now().isoformat())

                    flags = [os.path.exists(os.path.join(outfolder, f)) for f in expectedfsvariables]

                    if all(flags):
                        ret = True
                        with open(os.path.join(outfolder, 'RunComplete1'), 'w') as f:
                            f.write(datetime.datetime.now().isoformat())

                    else:
                        output = "incomplete_%s_%s" % (y1, y2)
                        for i in np.arange(0, len(flags)):
                            if flags[i] is False:
                                output = output + '_' + expectedfs[i]
                        with open(os.path.join(outfolder, output), 'w') as f:
                            f.write(datetime.datetime.now().isoformat())

                else:
                    ret = True

        if ret is True:
            ret = 0
        else:
            ret = 100
        return ret


class InitInfo(LakeInfo):
    """
    Class containing all info needed to create the init file.
    """

    def __init__(self, lake_name, depth, areas, w_temp):
        """
        :param lake_name: Type string. Long lake's name,
        :param depth: Type list. The depth levels obtained from the hypsometry file. values are floats.
        :param areas: Type list. The areas obtained from the hypsometry file at each depth levels. values are floats.
        :param w_temp: Type list. A complete set of mean temperatures for init files, ordered by depth levels
        """
        LakeInfo.__init__(self, lake_name)
        self.depth_levels = depth
        self.areas = areas
        self.w_temp = w_temp

    def init_file(self, i_sc_doc=1):
        """
        For ISI-MIP
        Creates the init file. Uses a dictionary to find the values for each parameters.
        :param i_sc_doc: Type float: by default 1. Scaling factor for inflow concentration of DOC (-)
        :return None
        """

        lines = [
            '\t'.join(
                [('%.2f' % d), ('%.0f' % a), ('%.f' % float(w_t))] + ['0'] * 5 + ['%s' % (2000 * i_sc_doc)] + [
                    '0'] * 5 + ['12000']
                + ['0'] * 15)  # MC 06-01-2018 add i_sc_doc and initial 8000 become 2000#MC 06-29-2018 12000
            # Z, Az and T, ...., DOC, .... DO, ...
            for d, a, w_t in zip(self.depth_levels, self.areas, self.w_temp)]

        # lines[0] = lines[0] + '\t0\t0'  # snow and ice, plus 16 dummies
        firstlines = '''-999   "MyLake init"
            Z (m)  Az (m2)    Tz (deg C) Cz Sz (kg/m3) TPz (mg/m3)    DOPz (mg/m3)   Chlaz (mg/m3)  DOCz (mg/m3)   
            TPz_sed (mg/m3)    Chlaz_sed (mg/m3)  "Fvol_IM (m3/m3     dry w.)"  Hice (m)   Hsnow (m)  DO dummy  dummy  
            dummy  dummy  dummy  dummy  dummy  dummy  dummy  dummy  dummy  dummy  dummy  dummy  dummy'''
        lines = [firstlines] + lines

        with open(r"{}/{}_init".format(self.input_path,self.prefix), 'w') as f:
            f.write('\n'.join(lines))

        print("{} Done".format(self.input_path))


class ParInfo(LakeInfo):
    """
    Class containing all info needed to create the parameters file.
    """

    def __init__(self, lake_name, kz_n0=0.00007, c_shelter="NaN", alb_melt_ice=0.6, alb_melt_snow=0.9, i_scv=1.15,
                 i_sct=0, swa_b0=2.5, swa_b1=1, k_bod=0.1, k_sod=500, i_sc_doc=1):
        """
        :param lake_name: Type string. Long lake's name.
        :param kz_n0: Type float: by default 0.00007. Min. stability frequency (s-2)
        :param c_shelter: Type float: by default empty. Wind shelter parameter (-)
        :param alb_melt_ice: Type float: by default 0.6. Albedo of melting ice (-)
        :param alb_melt_snow: Type float: by default 0.9. Albedo of melting ice (-)
        :param i_scv: Type float: by default 1.15. Scaling factor for inflow volume (-)
        :param i_sct: Type float: by default 0. Scaling coefficient for inflow temperature (-)
        :param swa_b0: Type float: by default 2.5. Non-PAR light atteneuation coefficient (m-1).
        :param swa_b1: Type float: by default 1. PAR light atteneuation coefficient (m-1).
        :param k_bod: Type float: by default 0.1. Organic decomposition rate (1/d)
        :param k_sod: Type float: by default 500. Sedimentary oxygen demand (mg m-2 d-1).
        :param i_sc_doc: Type float: by default 1. Scaling factor for inflow concentration of DOC  (-)
        """

        LakeInfo.__init__(self, lake_name)
        self.kz_N0 = kz_n0
        self.c_shelter = c_shelter
        self.alb_melt_ice = alb_melt_ice
        self.alb_melt_snow = alb_melt_snow
        self.i_scv = i_scv
        self.i_sct = i_sct
        self.swa_b0 = swa_b0
        self.swa_b1 = swa_b1
        self.k_BOD = k_bod
        self.k_SOD = k_sod
        self.I_scDOC = i_sc_doc

    def mylakepar(self):
        """
        Creates the MyLake parameter file. If the file LAE_para_all1.txt is present, it will be used to prepare the
        parameters. Otherwise, the string in this function using the parameter's value from the class will be used.
        :return: None
        """

        if os.path.isfile(
                "LAE_para_all1.txt"):
            print('using file')
            with open("LAE_para_all1.txt", "r") as infile:
                out = infile.read() % (
                    self.latitude, self.longitude, self.kz_N0, self.c_shelter, self.alb_melt_ice,
                    self.alb_melt_snow,
                    self.i_scv, self.i_sct, self.I_scDOC, self.swa_b0, self.swa_b1, self.k_BOD, self.k_SOD)



        else:
            out = '''-999	"Mylake parameters"			
            Parameter	Value	Min	Max	Unit
            dz	1.0	0.5	2	m
            Kz_ak	NaN	NaN	NaN	(-)
            Kz_ak_ice	0.0009	NaN	NaN	(-)
            Kz_N0	%f	NaN	NaN	s-2                 #7.00E-05
            C_shelter	%s	NaN	NaN	(-)
            latitude	%.5f	NaN	NaN	dec.deg
            longitude	%.5f	NaN	NaN	dec.deg
            alb_melt_ice	%f	NaN	NaN	(-)
            alb_melt_snow	%f	NaN	NaN	(-)
            PAR_sat	3.00E-05	1.00E-05	1.00E-04	mol m-2 s-1
            f_par	0.89	NaN	NaN	(-)
            beta_chl	0.015	0.005	0.045	m2 mg-1
            lamgbda_I	5	NaN	NaN	m-1
            lambda_s	15	NaN	NaN	m-1
            sed_sld	0.36	NaN	NaN	(m3/m3)
            I_scV 	%f	NaN	NaN	(-)
            I_scT	%f	NaN	NaN	deg C
            I_scC	1	NaN	NaN	(-)
            I_scS	1	1.1	1.9	(-)
            I_scTP	1	0.4	0.8	(-)
            I_scDOP	1	NaN	NaN	(-)
            I_scChl	1	NaN	NaN	(-)
            I_scDOC	%s	NaN	NaN	(-)
            swa_b0	%f	NaN	NaN	m-1
            swa_b1	%f	0.8	1.3	m-1
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
            ''' %(
                self.kz_N0, self.c_shelter, self.latitude, self.longitude, self.alb_melt_ice,
                self.alb_melt_snow,
                self.i_scv, self.i_sct, self.I_scDOC, self.swa_b0, self.swa_b1, self.k_BOD, self.k_SOD)

        outpath = self.input_path + r"\{}_par".format(self.prefix)

        with open(outpath, 'w') as f:
            f.write(out)

        print("{} Done".format(outpath))

        return outpath


class InputInfo(LakeInfo):
    """
    Class containing all info needed to create the input file.
    """

    def __init__(self, lake_name, modelid, scenarioid):
        """
        :param lake_name: Type string. Long lake's name.
        :param model: Climatic model used to generate the input values.
        :param scenario: Climatic scenario used to generate the input values.
        """

        LakeInfo.__init__(self, lake_name)
        self.model = models[modelid]
        self.scenario = scenarios[scenarioid]
        exA, y1A, exB, y1B = self.scenario
        m1, m2 = self.model
        y2A = y1A + 4
        y2B = y1B + 4

        if modelid == 4:  # 5-18-2018 MC
            self.pA = {v: '%s/Lakes_%s_EUR-11_%s_%s_%s_%s0101-%s1230.h5' %
                     (cordexfolder, v, m1, exA, m2, y1A, y2A) for v in variables}
            self.pB = {v: '%s/Lakes_%s_EUR-11_%s_%s_%s_%s0101-%s1230.h5' %
                     (cordexfolder, v, m1, exB, m2, y1B, y2B) for v in variables}
        else:
            self.pA = {v: '%s/Lakes_%s_EUR-11_%s_%s_%s_%s0101-%s1231.h5' %
                     (cordexfolder, v, m1, exA, m2, y1A, y2A) for v in variables}
            self.pB = {v: '%s/Lakes_%s_EUR-11_%s_%s_%s_%s0101-%s1231.h5' %
                     (cordexfolder, v, m1, exB, m2, y1B, y2B) for v in variables}

        inflowfilename = '%s/sweden_inflow_data_20010101_20101231.h5' % inflowfolder  # This is hard coded for now
        self.datesA = pd.date_range(pd.datetime(y1A, 1, 1), pd.datetime(y2A, 12, 31), freq='d').tolist()
        if pA['clt'].find('EUR-11_MOHC-HadGEM2-ES_rcp85_r1i1p1_SMHI-RCA4_v1_day_2091') != -1:
            self.datesB = pd.date_range(pd.datetime(y1B, 1, 1), pd.datetime(y2B - 1, 12, 31), freq='d').tolist()
        elif pA['clt'].find('EUR-11_MOHC-HadGEM2-ES_rcp45_r1i1p1_SMHI-RCA4_v1_day_2091') != -1:
            self.datesB = pd.date_range(pd.datetime(y1B, 1, 1), pd.datetime(y2B - 1, 11, 30), freq='d').tolist()
        else:
            self.datesB = pd.date_range(pd.datetime(y1B, 1, 1), pd.datetime(y2B, 12, 31), freq='d').tolist()


        outdir = os.path.join(self.output_path,'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (m1, exA, exB, m2, y1A, y2B))

        if not os.path.exists(outdir):
            os.makedirs(outdir)

    def mylakeinput(pA, pB, datesA, datesB, ebhex, subid, inflowfile, outpath):
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
                [take5_missingdata(pA, datesA, ebhex), take5_missingdata(pB, datesB, ebhex)])  # 5-24-2017 MC
            if datesB[-1] == pd.datetime(2099, 11, 30):  # 2018-08-01 MC
                dfmissingdata = dfmissingdata[:-31]
            df = dfmissingdata.interpolate()  # 5-24-2018 MC
        else:
            df = pd.concat([take5(pA, datesA, ebhex), take5(pB, datesB, ebhex)])

        ndays = len(datesA) + len(datesB)
        df.index = np.arange(ndays)
        dflow = inflow5(inflowfile, datesA + datesB, subid)
        repd = [datesA[0] + datetime.timedelta(d) for d in range(-(365 * 2), ndays)]
        mlyear = np.array([d.year for d in repd])
        mlmonth = np.array([d.month for d in repd])
        mlday = np.array([d.day for d in repd])
        mlndays = 365 + 365 + ndays
        repeati = list(range(365)) + list(range(365)) + list(range(ndays))
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


if __name__ == "__main__":
    lake_info = LakeInfo("test")



def mylakeinit(max_depth, area, outpath, I_scDOC=1):
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
        depth_levels = np.concatenate((depth_levels, np.array([max_depth]))) #a enlever
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

def take5(pdict, dates, ebhex):
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
    #5-7-2018 MC

    e = ebhex.lstrip('0x').lstrip('0')
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

def take5_missingdata(pdict, dates, ebhex): #5-24-2018 MC
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
    #5-7-2018 MC
    e = ebhex.lstrip('0x').lstrip('0')
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

def mylakeinput(pA, pB, datesA, datesB, ebhex, subid, inflowfile, outpath):
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
        dfmissingdata = pd.concat([take5_missingdata(pA, datesA, ebhex), take5_missingdata(pB, datesB, ebhex)]) #5-24-2017 MC
        if datesB[-1] == pd.datetime(2099,11,30):#2018-08-01 MC
            dfmissingdata= dfmissingdata[:-31]
        df = dfmissingdata.interpolate() # 5-24-2018 MC
    else:
        df = pd.concat([take5(pA, datesA, ebhex), take5(pB, datesB, ebhex)])

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

def runlake(modelid, scenarioid, ebhex, subid, depth, area, longitude, latitude,k_BOD=0.01,swa_b1=1,k_SOD=100,I_scDOC=1):
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

    ebhex = ebhex[2:] if ebhex[:2] == '0x' else ebhex
    while len ( ebhex ) < 6:
        ebhex = '0' + ebhex
    d1, d2, d3 = ebhex[:2], ebhex[:4], ebhex[:6]
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
        print ( 'lake %s is already completed' % ebhex )
        #with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
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
        mylakeinit ( depth, area, initp,I_scDOC)
        mylakepar ( longitude, latitude, parp,swa_b1,k_BOD,k_SOD,I_scDOC)
        mylakeinput ( pA, pB, datesA, datesB, ebhex, subid, inflowfilename, inputp )
        cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\');quit' % (initp, parp, inputp, y1A - 2, y2B, outdir)
        print ( cmd )
        os.system ( cmd )
        #for f in [initp, parp, inputp]:
        #    os.system ( 'bzip2 -f -k %s' % f )
        expectedfs = [ 'Tzt.csv','O2zt.csv', 'Attn_zt.csv', 'Qst.csv', 'DOCzt.csv','lambdazt.csv']
        flags = [os.path.exists ( os.path.join ( outdir, f ) ) for f in expectedfs]

        if all ( flags ):
            with open ( os.path.join ( outdir, '2017REDOCOMPLETE' ), 'w' ) as f:
                f.write ( datetime.datetime.now ().isoformat () )
            ret=0
        ret = 0 if all ( flags ) else 100
    return ret
