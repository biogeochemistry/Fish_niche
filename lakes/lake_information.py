#!/usr/bin/env python

""" Script for MyLake - ISIMIP
Calls the init, input and par scripts to create the appropriate files for MyLake model
Then launches MyLake for the specified lake.
"""

__author__ = "Julien Bellavance and Marianne Cote"

import csv
import datetime
import os
import statistics

import h5py
import math
import netCDF4 as Ncdf
import numpy as np
import pandas as pd

from math import sqrt, floor, log10, log
from sklearn.metrics import r2_score, mean_squared_error

from Graphics import Graphics

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

cordexfolder = r'F:\MCote output\optimisation_2018_10_12\cordex'  # 5-24-2018 MC #Need to be change depending where
#                                                                   the climatic files where
inflowfolder = r'C:\Users\macot620\Documents\GitHub\Fish_niche\sweden_inflow_data'
outputfolder = r'F:\output'  # 5-9-2018 MC
observation_folder = "Observations"
input_folder = "Inputs"

output_folder = r"D:\output_fish_niche"

lakes_data = pd.read_csv("2017SwedenList.csv", encoding='ISO-8859-1')
lakes_data = lakes_data.set_index("lake_id").to_dict()
# lakes_list = list(lakes_data.get("name").keys())
lakes_list = list(lakes_data.get("ebhex").keys())

def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

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


years_by_ebhex = {
    310: [2006, 2010],
    698: [2005, 2009],
    6950: [2006, 2010],
    14939: [2005, 2009],
    16765: [2006, 2010],
    30704: [2005, 2010],
    31895: [2005, 2009],
    32276: [2006, 2010],
    33494: [2006, 2010],
    33590: [2006, 2010],
    67035: [2003, 2007],
    99045: [2006, 2010]}


def get_key(val):
    for key, value in years_by_ebhex.items():
        # print(val, key)
        if val == key:
            return value

    return "key doesn't exist"


class LakeInfo:
    """
    Class containing all info needed by the script to run model and calibration.
    """

    def __init__(self, lake_name, lake_id, subid, ebhex, area, depth, mean, longitude, latitude, volume, turnover,
                 swab1='default', swab0='default', cshelter='default', isct='default', iscv='default',
                 isco='default', iscdoc='default', ksod='default', kbod='default', kzn0='default', albice='default',
                 albsnow='default', scenarioid=2, modelid=2, calibration=False, outputfolder=r'F:\output',old = False):
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
        self.mean = mean
        self.volume = volume
        self.longitude = longitude
        self.latitude = latitude
        self.observation_file = os.path.join(inflowfolder, "Validation_data_for_lookup.xlsx")
        self.turnover = turnover
        if turnover <= 3:
            self.spin_up = 3
        elif turnover <= 5:
            self.spin_up = 5
        else:
            self.spin_up = 5

        ebhex = ebhex[2:] if ebhex[:2] == '0x' else ebhex
        while len(ebhex) < 6:
            ebhex = '0' + ebhex
        d1, d2, d3 = ebhex[:2], ebhex[:4], ebhex[:6]
        outdir = os.path.join(outputfolder, d1, d2, d3)

        self.output_path = outdir
        self.old = old
        self.old_calibration_path = os.path.join(self.output_path,
                                                 "EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231",
                                                 "calibration_result_old")
        exA, y1A, exB, y1B = scenarios[scenarioid]
        m1, m2 = models[modelid]
        y2B = y1B + 4
        if get_key(lake_id) != "key doesn't exist":
            years = get_key(lake_id)
            self.start_year = years[0]
            self.end_year = years[1]
        else:
            self.start_year = y1A
            self.end_year = y1B + 4
        self.start_year = 2001
        self.end_year = 2010

        # oldcalibration_path = os.path.join(self.output_path,
        #                                      "EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231")

        if old:
            self.calibration_path = self.old_calibration_path
        else:
            self.calibration_path = os.path.join(self.output_path,
                                             "EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231",
                                             "calibration_result")

        self.outdir = os.path.join(self.output_path,
                                   'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (
                                       m1, exA, exB, m2, y1A, y2B))

        #
        # for file in["Observed_Secchi.csv","Observed_Oxygen.csv","Observed_Temperature.csv","Calibration_Complete.csv","Calibration_CompleteOXY.csv"]:
        #     if not os.path.exists(self.calibration_path):
        #         os.makedirs(self.calibration_path)
        #     if os.path.exists(os.path.join(oldcalibration_path, file)):
        #         shutil.move(os.path.join(oldcalibration_path, file),self.calibration_path)
        # for path, currentDirectory, files in os.walk(oldcalibration_path):
        #     if path == self.calibration_path:
        #         break
        #     for file in files:
        #         if file.startswith("Calibration_C"):
        #             shutil.move(os.path.join(oldcalibration_path, file),self.calibration_path)
        print([swab1, swab0, cshelter, iscv, isct, isco, iscdoc, ksod, kbod, kzn0, albice, albsnow],
              all(e == 'default' for e in
                  [swab1, swab0, cshelter, iscv, isct, isco, iscdoc, ksod, kbod, kzn0, albice, albsnow]))
        for e in [swab1, swab0, cshelter, iscv, isct, isco, iscdoc, ksod, kbod, kzn0, albice, albsnow]:
            if e != 'default':
                print("here")

        if not all(e == 'default' for e in
                   [swab1, swab0, cshelter, iscv, isct, isco, iscdoc, ksod, kbod, kzn0, albice, albsnow]):

            # self.start_year = y1A
            # self.end_year = y1B + 4

            self.swa_b1 = swab1
            self.swa_b0 = swab0
            self.c_shelter = cshelter
            self.i_scv = iscv
            self.i_sco = isco
            self.i_sc_doc = iscdoc
            self.k_sod = ksod
            self.k_bod = kbod
            self.kz_n0 = kzn0
            self.alb_melt_snow = albsnow
            self.alb_melt_ice = albice
            self.i_sct = isct

        else:
            if not calibration:
                try:
                    if os.path.exists(os.path.join(self.outdir, "Observed_Secchi.csv")):
                        swa_b1value = pd.read_csv(os.path.join(self.outdir, "Observed_Secchi.csv"))
                        mean_swa_b1 = round(1.48 / (swa_b1value.iloc[:, 1].mean()), 4)
                    else:
                        mean_swa_b1 = 1
                except:
                    mean_swa_b1 = 1

                if os.path.exists(os.path.join(self.outdir, "Calibration_Complete.csv")):
                    data = pd.read_csv(os.path.join(self.outdir, "Calibration_Complete.csv"), header=None)
                    if os.path.exists(os.path.join(self.outdir, "Calibration_CompleteOXY.csv")):
                        dataOXY = pd.read_csv(os.path.join(self.outdir, "Calibration_CompleteOXY.csv"),
                                              header=None)

                        if os.path.exists(os.path.join(self.outdir, "2017par")):
                            with open(os.path.join(self.outdir, "2017par")) as f:
                                lines = f.readlines()
                                for line in lines:
                                    line = line.split(sep="	")
                                    line[0] = line[0].replace(" ", "")

                                    if line[0] == "Kz_N0":
                                        self.kz_n0 = data.iloc[0, 0]

                                    elif line[0] == "C_shelter":
                                        self.c_shelter = data.iloc[0, 1]


                                    elif line[0] == "alb_melt_ice":
                                        self.alb_melt_ice = line[1]  # data.iloc[0, 2]

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
                                        # self.swa_b0 = 2.5
                                        self.swa_b0 = data.iloc[0, 6]

                                    elif line[0] == "swa_b1":
                                        try:
                                            self.swa_b1 = data.iloc[0, 7]
                                        except:
                                            self.swa_b1 = mean_swa_b1
                                        # self.swa_b1 = data.iloc[0, 7]
                                        # self.swa_b1 = data.iloc[0, 6]]

                                    elif line[0] == "k_BOD":
                                        self.k_bod = dataOXY.iloc[0, 1]

                                    elif line[0] == "k_SOD":
                                        self.k_sod = dataOXY.iloc[0, 2]

                        elif os.path.exists(os.path.join(self.outdir, "2020par")):
                            with open(os.path.join(self.outdir, "2020par")) as f:
                                lines = f.readlines()
                                for line in lines:
                                    line = line.split(sep="	")
                                    line[0] = line[0].replace(" ", "")

                                    if line[0] == "Kz_N0":
                                        self.kz_n0 = data.iloc[0, 0]

                                    elif line[0] == "C_shelter":
                                        self.c_shelter = data.iloc[0, 1]


                                    elif line[0] == "alb_melt_ice":
                                        self.alb_melt_ice = line[1]  # data.iloc[0, 2]

                                    elif line[0] == "alb_melt_snow":
                                        self.alb_melt_snow = line[1]  # data.iloc[0, 3]

                                    elif line[0] == "I_scV":
                                        self.i_scv = line[1]  # data.iloc[0, 4]

                                    elif line[0] == "I_scT":
                                        self.i_sct = line[1]  # data.iloc[0, 5]

                                    elif line[0] == "I_scO":
                                        self.i_sco = dataOXY.iloc[0, 3]

                                    elif line[0] == "I_scDOC":
                                        self.i_sc_doc = dataOXY.iloc[0, 0]

                                    elif line[0] == "swa_b0":
                                        # self.swa_b0 = 2.5
                                        self.swa_b0 = data.iloc[0, 2]  # data.iloc[0, 6]

                                    elif line[0] == "swa_b1":
                                        try:
                                            self.swa_b1 = data.iloc[0, 3]  # data.iloc[0, 7]
                                        except:
                                            self.swa_b1 = mean_swa_b1
                                        # self.swa_b1 = data.iloc[0, 7]
                                        # self.swa_b1 = data.iloc[0, 6]

                                    elif line[0] == "k_BOD":
                                        self.k_bod = dataOXY.iloc[0, 1]

                                    elif line[0] == "k_SOD":
                                        self.k_sod = dataOXY.iloc[0, 2]

                        else:
                            self.kz_n0 = data.iloc[0, 0]
                            self.c_shelter = data.iloc[0, 1]
                            self.alb_melt_ice = 0.6  # data.iloc[0, 2]
                            self.alb_melt_snow = 0.9  # data.iloc[0, 3]
                            self.i_scv = data.iloc[0, 4]
                            self.i_sct = data.iloc[0, 5]
                            # self.swa_b0 = data.iloc[0, 6]
                            self.swa_b0 = 2.5
                            self.swa_b1 = mean_swa_b1
                            # self.swa_b1 = data.iloc[0, 7]
                            self.k_bod = dataOXY.iloc[0, 1]
                            self.k_sod = dataOXY.iloc[0, 2]
                            self.i_sc_doc = dataOXY.iloc[0, 0]
                            self.i_sco = dataOXY.iloc[0, 3]

                    else:
                        if os.path.exists(os.path.join(self.outdir, "2017par")):
                            with open(os.path.join(self.outdir, "2017par")) as f:
                                lines = f.readlines()
                                for line in lines:
                                    line = line.split(sep="	")
                                    line[0] = line[0].replace(" ", "")

                                    if line[0] == "Kz_N0":
                                        self.kz_n0 = data.iloc[0, 0]

                                    elif line[0] == "C_shelter":
                                        self.c_shelter = data.iloc[0, 1]


                                    elif line[0] == "alb_melt_ice":
                                        self.alb_melt_ice = line[1]  # data.iloc[0, 2]

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
                                        # self.swa_b0 = 2.5
                                        self.swa_b0 = data.iloc[0, 6]

                                    elif line[0] == "swa_b1":
                                        try:
                                            self.swa_b1 = data.iloc[0, 7]
                                        except:
                                            self.swa_b1 = mean_swa_b1
                                        # self.swa_b1 = data.iloc[0, 7]
                                        # self.swa_b1 = data.iloc[0, 6]

                                    elif line[0] == "k_BOD":
                                        self.k_bod = line[1]

                                    elif line[0] == "k_SOD":
                                        self.k_sod = line[1]

                        elif os.path.exists(os.path.join(self.outdir, "2020par")):
                            with open(os.path.join(self.outdir, "2020par")) as f:
                                lines = f.readlines()
                                for line in lines:
                                    line = line.split(sep="	")
                                    line[0] = line[0].replace(" ", "")

                                    if line[0] == "Kz_N0":
                                        self.kz_n0 = data.iloc[0, 0]

                                    elif line[0] == "C_shelter":
                                        self.c_shelter = data.iloc[0, 1]


                                    elif line[0] == "alb_melt_ice":
                                        self.alb_melt_ice = line[1]  # data.iloc[0, 2]

                                    elif line[0] == "alb_melt_snow":
                                        self.alb_melt_snow = line[1]  # data.iloc[0, 3]

                                    elif line[0] == "I_scV":
                                        self.i_scv = line[1]  # data.iloc[0, 4]

                                    elif line[0] == "I_scT":
                                        self.i_sct = line[1]  # data.iloc[0, 5]

                                    elif line[0] == "I_scO":
                                        self.i_sco = line[1]

                                    elif line[0] == "I_scDOC":
                                        self.i_sc_doc = line[1]

                                    elif line[0] == "swa_b0":
                                        # self.swa_b0 = 2.5
                                        self.swa_b0 = data.iloc[0, 2]  # data.iloc[0, 6]

                                    elif line[0] == "swa_b1":
                                        try:
                                            self.swa_b1 = data.iloc[0, 3]  # data.iloc[0, 7]
                                        except:
                                            self.swa_b1 = mean_swa_b1
                                        # self.swa_b1 = data.iloc[0, 7]
                                        # self.swa_b1 = data.iloc[0, 6]

                                    elif line[0] == "k_BOD":
                                        self.k_bod = line[1]

                                    elif line[0] == "k_SOD":
                                        self.k_sod = line[1]

                        else:
                            self.kz_n0 = data.iloc[0, 0]
                            self.c_shelter = data.iloc[0, 1]
                            try:
                                self.alb_melt_ice = 0.6  # data.iloc[0, 2]
                                self.alb_melt_snow = 0.9  # data.iloc[0, 3]
                                self.i_scv = 1.15  # data.iloc[0, 4]
                                self.i_sct = 0  # data.iloc[0, 5]
                                self.i_sco = 1
                                self.swa_b0 = 2.5  # data.iloc[0, 6]
                            except:
                                self.alb_melt_ice = 0.6
                                self.alb_melt_snow = 0.9
                                self.i_scv = 1.15
                                self.i_sct = 0
                                self.i_sco = 1
                                self.swa_b0 = data.iloc[0, 3]

                            try:
                                self.swa_b1 = data.iloc[0, 7]
                            except:
                                self.swa_b1 = mean_swa_b1
                            # self.swa_b1 = data.iloc[0, 7]
                            self.k_bod = 500
                            self.k_sod = 0.1
                            self.i_sc_doc = 1

                else:
                    if os.path.exists(os.path.join(self.outdir, "2017par")):
                        with open(os.path.join(self.outdir, "2017par")) as f:
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

                                    self.swa_b1 = mean_swa_b1

                                    # self.swa_b1 = line[1]

                                elif line[0] == "k_BOD":
                                    self.k_bod = line[1]

                                elif line[0] == "k_SOD":
                                    self.k_sod = line[1]

                    elif os.path.exists(os.path.join(self.outdir, "2020par")):
                        with open(os.path.join(self.outdir, "2020par")) as f:
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
                                    self.swa_b1 = mean_swa_b1
                                    # self.swa_b1 = line[1]

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
                        self.k_bod = 0.001
                        self.k_sod = 100
                        self.i_sc_doc = 4.75

            else:
                # if get_key(lake_id) != "key doesn't exist":
                #     years = get_key(lake_id)
                #     self.start_year = years[0]
                #     self.end_year = years[1]
                # else:
                #     self.start_year = y1A
                #     self.end_year = y1B + 4

                try:
                    if os.path.exists(os.path.join(self.calibration_path, "Observed_Secchi.csv")):
                        swa_b1value = pd.read_csv(os.path.join(self.calibration_path, "Observed_Secchi.csv"))
                        mean_swa_b1 = round(1.48 / (swa_b1value.iloc[:, 1].mean()), 4)
                    else:
                        mean_swa_b1 = 1
                except:
                    mean_swa_b1 = 1

                if os.path.exists(os.path.join(self.calibration_path, "Calibration_Complete.csv")):
                    data = pd.read_csv(os.path.join(self.calibration_path, "Calibration_Complete.csv"), header=None)
                    if os.path.exists(os.path.join(self.calibration_path, "Calibration_CompleteOXY.csv")):
                        dataOXY = pd.read_csv(os.path.join(self.calibration_path, "Calibration_CompleteOXY.csv"),
                                              header=None)

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
                                        # self.swa_b0 = 2.5
                                        self.swa_b0 = data.iloc[0, 6]

                                    elif line[0] == "swa_b1":
                                        try:
                                            self.swa_b1 = data.iloc[0, 7]
                                        except:
                                            self.swa_b1 = mean_swa_b1
                                        # self.swa_b1 = data.iloc[0, 7]
                                        # self.swa_b1 = data.iloc[0, 6]]

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
                                        # self.swa_b0 = 2.5
                                        self.swa_b0 = data.iloc[0, 6]

                                    elif line[0] == "swa_b1":
                                        try:
                                            self.swa_b1 = data.iloc[0, 7]
                                        except:
                                            self.swa_b1 = mean_swa_b1
                                        #   self.swa_b1 = data.iloc[0, 7]
                                        # self.swa_b1 = data.iloc[0, 6]

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
                            #self.swa_b0 = 2.5
                            #self.swa_b1 = mean_swa_b1
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
                                        self.alb_melt_ice =  data.iloc[0, 2]

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
                                        # self.swa_b0 = 2.5
                                        self.swa_b0 = data.iloc[0, 6]

                                    elif line[0] == "swa_b1":
                                        try:
                                            self.swa_b1 = data.iloc[0, 7]
                                        except:
                                            self.swa_b1 = mean_swa_b1
                                        # self.swa_b1 = data.iloc[0, 7]
                                        # self.swa_b1 = data.iloc[0, 6]

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
                                        # self.swa_b0 = 2.5
                                        self.swa_b0 = data.iloc[0, 6]

                                    elif line[0] == "swa_b1":
                                        try:
                                            self.swa_b1 = data.iloc[0, 7]
                                        except:
                                            self.swa_b1 = mean_swa_b1
                                        # self.swa_b1 = data.iloc[0, 7]
                                        # self.swa_b1 = data.iloc[0, 6]

                                    elif line[0] == "k_BOD":
                                        self.k_bod = line[1]

                                    elif line[0] == "k_SOD":
                                        self.k_sod = line[1]

                        else:
                            self.kz_n0 = data.iloc[0, 0]
                            self.c_shelter = data.iloc[0, 1]
                            try:
                                self.alb_melt_ice = data.iloc[0, 2]
                                self.alb_melt_snow = data.iloc[0, 3]
                                self.i_scv = data.iloc[0, 4]
                                self.i_sct = data.iloc[0, 5]
                                self.i_sco = 1
                                self.swa_b0 = data.iloc[0, 6]
                            except:
                                self.alb_melt_ice = 0.6
                                self.alb_melt_snow = 0.9
                                self.i_scv = 1.15
                                self.i_sct = 0
                                self.i_sco = 1
                                self.swa_b0 = data.iloc[0, 3]

                            try:
                                self.swa_b1 = data.iloc[0, 7]
                            except:
                                self.swa_b1 = mean_swa_b1
                            # self.swa_b1 = data.iloc[0, 7]
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

                                    self.swa_b1 = mean_swa_b1

                                    # self.swa_b1 = line[1]

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
                                    self.swa_b1 = mean_swa_b1
                                    # self.swa_b1 = line[1]

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
                        self.k_bod = 0.001
                        self.k_sod = 100
                        self.i_sc_doc = 4.75

    def variables_by_depth(self,start=2001,end=2010,calibration = False, old=False):
        """
        Creates a new csv file with the observed temperatures separated in columns by depths.
        :return: None
        """
        lake_id = "%s"%self.lake_id
        observation_file = self.observation_file
        obs_file = pd.read_excel(observation_file,lake_id)
        obs_file['date'] = pd.to_datetime(obs_file['date'])
        obs_file = obs_file[(obs_file['date'] >= pd.datetime(int(start), 1, 1)) & (obs_file['date'] <= pd.datetime(int(end), 12, 31))]

        axisx = obs_file.iloc[:,0].unique()
        axisy = obs_file.iloc[:,1].unique()
        datasetT = []
        datasetO = []
        datasetS = []
        for date in axisx:
            valueatdepthT = []
            valueatdepthO = []
            valueatdepthS = []
            for depth in axisy:
                row = obs_file.loc[(obs_file['date'] == date) & (obs_file['depth(max)'] == depth)]
                if not row.empty:
                    if len(list(row.index.values)) != 1:
                        valueatdepthT.append(float(row.iloc[:, 3].mean(skipna=True)))
                        valueatdepthO.append(float(row.iloc[:, 2].mean(skipna=True)))
                        valueatdepthS.append(float(row.iloc[:, 4].mean(skipna=True)))
                    else:
                        try:
                            valueatdepthT.append(float(row.iloc[:,3]))
                            valueatdepthO.append(float(row.iloc[:,2]))
                            valueatdepthS.append(float(row.iloc[:,4]))
                        except:
                            print("rer")
                else:
                    valueatdepthT.append(np.nan)
                    valueatdepthO.append(np.nan)
                    valueatdepthS.append(np.nan)
            datasetT.append(valueatdepthT)
            datasetO.append(valueatdepthO)
            datasetS.append(valueatdepthS)

        if calibration:
            if old:
                outdir_path = self.old_calibration_path
            else:
                outdir_path = self.calibration_path
        else:
            outdir_path = self.output_path

        temperature = pd.DataFrame(index=axisx,columns=axisy, data=datasetT)
        print(len(list(temperature.index.values)))
        temperature = temperature.dropna(axis=1, how='all')
        temperature = temperature.dropna(axis=0, how='all')

        print(len(list(temperature.index.values)))
        temperature.to_csv(os.path.join(outdir_path, "Observed_Temperature.csv"))


        oxygen = pd.DataFrame(index=axisx, columns=axisy, data=datasetO)
        oxygen = oxygen.dropna(axis=1, how='all')
        oxygen = oxygen.dropna(axis=0, how='all')
        oxygen.to_csv(os.path.join(outdir_path, "Observed_Oxygen.csv"))

        secchi = pd.DataFrame(index=axisx, columns=axisy, data=datasetS)
        secchi = secchi.dropna(axis=1, how='all')
        secchi = secchi.dropna(axis=0, how='all')
        secchi.to_csv(os.path.join(outdir_path, "Observed_Secchi.csv"))


        print("observation done ... ... ... ... ")

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
        if os.path.exists(os.path.join(outdir, '20210507REDOCOMPLETE')):
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
            self.mylakeinput(pA, pB, datesA, datesB, inflowfilename, inputp)
            cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran_optimize(\'%s\',\'%s\',\'%s\',%d,%d,%d,\'%s\',%d);quit' % (
                initp, parp, inputp, y1A, y2B, self.spin_up, outdir, 1)
            print(cmd)
            os.system(cmd)
            #     # for f in [initp, parp, inputp]:
            #     #    os.system ( 'bzip2 -f -k %s' % f )
            expectedfs = ['Tzt.csv', 'O2zt.csv', 'Attn_zt.csv', 'Qst.csv', 'DOCzt.csv', 'PARMaxt.csv', 'PARzt.csv',
                          'His.csv', 'lambdazt.csv']
            flags = [os.path.exists(os.path.join(outdir, f)) for f in expectedfs]

            if all(flags):
                with open(os.path.join(outdir, '20210507REDOCOMPLETE'), 'w') as f:
                    f.write(datetime.datetime.now().isoformat())
                ret = False
            #         ret = 0
        #     ret = 0 if all(flags) else

        return ret

    def runlakefinal(self, modelid, scenarioid, calibration=False,old=False):
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

        if calibration:
            if old:
                outdir = self.old_calibration_path
            else:
                outdir = self.calibration_path
            calibration = 1
        else:
            outdir = self.outdir
            calibration = 0

        if self.lake_id in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 698, 16765, 67035]:
            calibration = 1
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # creation of empty files before risks of bug: MC 2018-07-10

        initp = os.path.join(outdir, '2020init')
        if os.path.exists(os.path.join(outdir, '2017par')):
            parp = os.path.join(outdir, '2020par')
        else:
            parp = os.path.join(outdir, '2020par')
        inputp = os.path.join(outdir, '2020input')
        if os.path.exists(os.path.join(outdir, '20210602REDOCOMPLETE')):
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
            self.mylakeinput(pA, pB, datesA, datesB, inflowfilename, inputp)
            cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran_optimizefinal_all(\'%s\',\'%s\',\'%s\',%d,%d,%d,\'%s\',%d);quit' % (
                initp, parp, inputp, y1A, y2B, self.spin_up, outdir, calibration)
            print(cmd)
            os.system(cmd)
            if self.lake_id in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 698, 16765, 67035] and m2 == 'r3i1p1_DMI-HIRHAM5_v1_day' and y1A == 2001:
                cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                    outdir, y1A,y2B)
                print(cmd)
                os.system(cmd)
            #     # for f in [initp, parp, inputp]:
            #     #    os.system ( 'bzip2 -f -k %s' % f )
            expectedfs = ['Tzt.csv', 'O2zt.csv', 'Attn_zt.csv', 'Qst.csv', 'DOCzt.csv', 'PARMaxt.csv', 'PARzt.csv',
                          'His.csv', 'lambdazt.csv']
            flags = [os.path.exists(os.path.join(outdir, f)) for f in expectedfs]

            if all(flags):
                with open(os.path.join(outdir, '20210602REDOCOMPLETE'), 'w') as f:
                    f.write(datetime.datetime.now().isoformat())
                ret = False
            #         ret = 0
        #     ret = 0 if all(flags) else

        return ret

    def lake_input(self, modelid, scenarioid, calibration=False):
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

        if calibration:
            outdir = self.calibration_path

        else:
            outdir = self.outdir

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
        # else:
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

    def mylakeinit(self, outdir):
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
        # areas = self.area * (depth_levels - self.depth) ** 2 / self.depth ** 2
        areas = np.array([round(Area_base_i(i, self.area, self.depth, self.mean), 3) for i in depth_levels])
        lines = [
            '\t'.join(
                [('%.2f' % d), ('%.0f' % a)] + ['4'] + ['0'] * 5 + ['%s' % (2000 * float(self.i_sc_doc))] + [
                    '0'] * 5 + ['12000'] + ['0'] * 15)
            # MC 06-01-2018 add I_scDOC and initial 8000 become 2000#MC 06-29-2018 12000
            # Z,        Az and          T, ...., DOC, .... DO, ...
            for d, a in zip(depth_levels, areas)]
        # lines[0] = lines[0] + '\t0\t0'  # snow and ice, plus 16 dummies
        firstlines = '''-999	"MyLake init"
        Z (m)	Az (m2)	Tz (deg C)	Cz	Sz (kg/m3)	TPz (mg/m3)	DOPz (mg/m3)	Chlaz (mg/m3)	DOCz (mg/m3)	TPz_sed (mg/m3)	
        Chlaz_sed (mg/m3)	"Fvol_IM (m3/m3	 dry w.)"	Hice (m)	Hsnow (m)	DO	dummy	dummy	dummy	dummy	dummy	
        dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy'''

        lines = [firstlines] + lines

        with open(os.path.join(outdir), 'w') as f:
            f.write('\n'.join(lines))

    def mylakepar(self, outdir):
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
                    self.i_scv, self.i_sct, self.i_sc_doc, self.swa_b0, self.swa_b1, self.k_bod, self.k_sod,self.i_sco)



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
                I_scO	%s	NaN	NaN	(-)
            ''' % (
                self.kz_n0, self.c_shelter, self.latitude, self.longitude, self.alb_melt_ice,
                self.alb_melt_snow,
                self.i_scv, self.i_sct, self.i_sc_doc, self.swa_b0, self.swa_b1, self.k_bod, self.k_sod,self.i_sco)

        outpath = outdir

        with open(outpath, 'w') as f:
            f.write(out)

        print("{} Done".format(outpath))

        return outpath

    def mylakeinput(self, pA, pB, datesA, datesB, inflowfile, outpath):
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
                [self.take5_missingdata(pA, datesA, self.ebhex),
                 self.take5_missingdata(pB, datesB, self.ebhex)])  # 5-24-2017 MC
            if datesB[-1] == pd.datetime(2099, 11, 30):  # 2018-08-01 MC
                dfmissingdata = dfmissingdata[:-31]
            df = dfmissingdata.interpolate()  # 5-24-2018 MC
        else:
            df = pd.concat([self.take5(pA, datesA, self.ebhex), self.take5(pB, datesB, self.ebhex)])

        ndays = len(datesA) + len(datesB)
        calibration_time = (datesA[0] - pd.datetime(datesA[0].year - self.spin_up, datesA[0].month, datesA[0].day)).days

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

    def take5(self, pdict, dates, ebhex):
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

    def take5_missingdata(self, pdict, dates, ebhex):  # 5-24-2018 MC
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

    def nbrleapyears(self, start, end):  # MC 2018-07-10
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

    def inflow5(self, filename, dates, subId):
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
            dflow['T'] = np.nan #remove inflow temperature; issue for some lake with the ice covert formation.
        return dflow

    def performance_analysis(self, calibration=True,old=False):
        """
        Opens the comparison file created by make_comparison_file, and prints the results of analysis functions.
        :return: Score, a float representing the overall performance of the current simulation.
        """
        if calibration:
            outfolder = self.calibration_path
            if old:
                outfolder = self.old_calibration_path
        else:
            outfolder = self.outdir

        if os.path.exists("{}/Tztcompare.csv".format(outfolder)):
            with open("{}/Tztcompare.csv".format(outfolder), "r") as file:
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

            rmseT, rmsenT = root_mean_square(obs_list, sims_list)

            r_sqT = r_squared(obs_list, sims_list)
        else:
            sosT = np.nan
            rmseT, rmsenT = np.nan, np.nan
            r_sqT = np.nan

        if os.path.exists("{}/O2ztcompare.csv".format(outfolder)):
            with open("{}/O2ztcompare.csv".format(outfolder), "r") as file:
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

            rmseO, rmsenO = root_mean_square(obs_list, sims_list)

            r_sqO = r_squared(obs_list, sims_list)
        else:
            sosO = np.nan
            rmseO, rmsenO = np.nan, np.nan
            r_sqO = np.nan

        if os.path.exists("{}/Secchicompare.csv".format(outfolder)):
            with open("{}/Secchicompare.csv".format(outfolder), "r") as file:
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

            with open("{}/Attn_zt.csv".format(outfolder), "r") as file2:
                reader2 = list(csv.reader(file2))

            sosS = sums_of_squares(obs_list, sims_list)

            rmseS, rmsenS = root_mean_square(obs_list, sims_list)

            r_sqS = r_squared(obs_list, sims_list)

            secchiO_m = np.mean([float(i) for i in obs_list])
            secchiM_m = np.mean([float(i) for i in sims_list])
            secchiO_st = np.std([float(i) for i in obs_list])
            secchiM_st = np.std([float(i) for i in sims_list])
        else:
            sosS = np.nan
            rmseS, rmsenS = np.nan, np.nan
            r_sqS = np.nan
            secchiM_m, secchiO_m = np.nan, np.nan
            secchiM_st, secchiO_st = np.nan, np.nan

        print("Analysis of {}.".format(self.lake_name))
        print("Sums of squares : {}, {}, {}".format(sosT, sosO, sosS))
        print("RMSE : {}, {}, {}".format(rmseT, rmseO, rmseS))
        print("R squared : {}, {}, {}".format(r_sqT, r_sqO, r_sqS))

        return [rmseT, rmseO, rmseS], [rmsenT, rmsenO, rmsenS], [r_sqT, r_sqO, r_sqS], [secchiO_m, secchiM_m,
                                                                                        secchiO_st, secchiM_st]

    def outputfile(self, calibration=False,old=False, scenarioid=2, modelid=2):
        """
        Function calculating all variables asked by ISIMIP and formates them into a txt file.
        :param y1: Initial year of the simulation
        :param y2: Final year of the simulation
        :param outfolder: folder directory to where the raw data produce by the model are.
        :return: None
        """
        y1, y2 = self.start_year - 8, self.end_year

        if calibration:
            if old:
                outfolder=self.old_calibration_path
            else:
                outfolder = self.calibration_path
        else:
            outfolder = self.outdir

        # outfolder = self.calibration_path
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

    def comparison_obs_sims_new(self, thermocline, calibration=False,outputfolder=r'F:\output'):
            # Default values
            ice_modeled = False
            start = 2001
            end = 2010
            calibration_methods = ['calculated','old_calculated','estimated']
            levels = ['surface', 'deepwater']
            variables = ["Tzt" ,"O2zt"]

            if calibration:
                outfolder = self.calibration_path
            else:
                outfolder = self.outdir

            ### Creatin of the ice cover variable
            if os.path.exists((os.path.join(outfolder, "His.csv"))):
                ice_modeled = True
                icedata = pd.read_csv(os.path.join(outfolder, "His.csv"),header=None)
                dates = pd.date_range(start='1/1/%s' % start, end='12/31/%s' % end)
                print(len(dates), len(icedata))
                if len(dates) == len(icedata):
                    icedata = icedata.set_index(dates)
                    icedata['date'] = dates
                else:
                    dates = pd.date_range(start='1/1/%s' % self.start_year, end='12/31/%s' % self.end_year)
                    icedata= icedata.set_index(dates)
                    icedata['date'] = dates
                ice_cover = icedata.iloc[:,[6,8]]
            else:
                print("No ice cover modeled for lake %s" % self.lake_name)


            for variable in variables:
                # Data Treatement
                #self.variables_by_depth(2001, 2010, calibration=True,old=False)
                # self.runlakefinal(2,2, calibration=True,old=False)
                try:
                    ### Creation Obs and Sim variables
                    if os.path.exists(os.path.join(outfolder, "%scompare.csv" % variable)):
                        data = pd.read_csv(os.path.join(outfolder, "%scompare.csv" % variable), header=None,
                                           names=['Date', 'Depth', 'Observations', 'Modelisation'])

                        data['Dates'] = pd.to_datetime(data["Date"])
                        data = data.set_index(data['Dates'])
                        initial_date = data['Dates'].min()
                        final_date = data['Dates'].max()
                        depth_range = data["Depth"].unique()
                        ice_cover = ice_cover.loc["%s-01-01" % str(initial_date.year):"%s-12-31" % str(final_date.year)]
                        ### Division of the dataset into variable under and over thermocline

                        comparison_target = pd.DataFrame(index=data['Dates'], columns=calibration_methods)
                        comparison_target['Observations'] = data['Observations']
                        date = pd.date_range(start="%s-01-01" % str(initial_date.year),
                                             end="%s-12-31" % str(final_date.year))

                        frame = {"Dates": date}
                        modelfinalresult = pd.DataFrame(frame).set_index(date)
                        depth_layers = {}
                        for depthlevel in levels:

                            #division of the dataset by how they have beeen generated
                            for modelresult in calibration_methods:

                                if modelresult == 'calculated':
                                    modeldata = pd.read_csv(os.path.join(self.calibration_path, "%s.csv" % variable), header=None)
                                    try:
                                        data_target = pd.read_csv(
                                            os.path.join(self.calibration_path, "%scompare.csv" % variable), header=None,
                                            names=['Date', 'Depth', 'Observations', 'Modelisation']).set_index('Date')
                                        comparison_target[modelresult] = data_target['Modelisation']
                                    except:
                                        self.variables_by_depth(2001, 2010,True, False)
                                        self.runlakefinal(2, 2, calibration=True, old=False)
                                        print(self.calibration_path)

                                        # cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                                        #     self.calibration_path, 2001, 2010)
                                        # print(cmd)
                                        # os.system(cmd)
                                        data_target = pd.read_csv(
                                            os.path.join(self.calibration_path, "%scompare.csv" % variable), header=None,
                                            names=['Date', 'Depth', 'Observations', 'Modelisation']).set_index('Date')
                                        comparison_target[modelresult] = data_target['Modelisation']
                                elif modelresult == 'old_calculated':
                                    modeldata = pd.read_csv(os.path.join(self.old_calibration_path, "%s.csv" % variable), header=None)
                                    try:
                                        data_target = pd.read_csv(os.path.join(self.old_calibration_path, "%scompare.csv" % variable), header=None,
                                                              names=['Date', 'Depth', 'Observations', 'Modelisation']).set_index('Date')
                                        comparison_target[modelresult] = data_target['Modelisation']
                                    except:
                                        self.variables_by_depth(2001, 2010,True,True)
                                        self.runlakefinal(2, 2, calibration=True, old=True)
                                        print(self.old_calibration_path)
                                        # cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                                        #     self.old_calibration_path, 2001, 2010)
                                        # print(cmd)
                                        # os.system(cmd)
                                        data_target = pd.read_csv(
                                            os.path.join(self.old_calibration_path, "%scompare.csv" % variable), header=None,
                                            names=['Date', 'Depth', 'Observations', 'Modelisation']).set_index('Date')
                                        comparison_target[modelresult] = data_target['Modelisation']
                                else:
                                    modeldata = pd.read_csv(os.path.join(self.outdir, "%s.csv" % variable), header=None)

                                    try:
                                        data_target = pd.read_csv(
                                            os.path.join(self.outdir, "%scompare.csv" % variable), header=None,
                                            names=['Date', 'Depth', 'Observations', 'Modelisation']).set_index('Date')
                                        comparison_target[modelresult] = data_target['Modelisation']
                                    except:
                                        self.variables_by_depth(2001, 2010,False, False)
                                        self.runlakefinal(2, 2, calibration=False, old=False)
                                        print(self.outdir)
                                        # cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                                        #     self.outdir, 2001, 2010)
                                        # print(cmd)
                                        # os.system(cmd)
                                        data_target = pd.read_csv(
                                            os.path.join(self.outdir, "%scompare.csv" % variable), header=None,
                                            names=['Date', 'Depth', 'Observations', 'Modelisation']).set_index('Date')
                                        comparison_target[modelresult] = data_target['Modelisation']


                                dates = pd.date_range(start='1/1/%s' % start, end='12/31/%s' % end)
                                if len(dates) == len(modeldata):
                                    modeldata = modeldata.set_index(dates)
                                else:
                                    dates = pd.date_range(start='1/1/%s' % self.start_year, end='12/31/%s' % self.end_year)
                                    modeldata = modeldata.set_index(dates)

                                modeldata = modeldata.loc["%s-01-01" % str(initial_date.year):"%s-12-31" % str(final_date.year)]

                                if depthlevel == 'surface':
                                    subdatabydepth = [depth for depth in depth_range if depth < thermocline]
                                else:
                                    subdatabydepth = [depth for depth in depth_range if depth >= thermocline]


                                # Select depth representing the layers

                                numberbydepth = list(
                                    data[data['Depth'].isin(subdatabydepth)].groupby(['Depth']).count()['Observations'])

                                if len(numberbydepth) == 1:
                                    depthlayer = subdatabydepth[0]
                                elif len(numberbydepth) < 1:
                                    depthlayer = "no data"
                                else:
                                    max_position = [i for i, x in enumerate(numberbydepth) if x == max(numberbydepth)]
                                    max1 = max_position[0]
                                    if depthlevel == "deepwater":
                                        depthlayer = float(subdatabydepth[max_position[-1]])
                                    else:
                                        depthlayer = float(subdatabydepth[max_position[0]])
                                    # depthlayer = subdatabydepth[numberbydepth.index(max(numberbydepth)]

                                if depthlayer != "no data":

                                    if (floor(depthlayer) + 0.5) == depthlayer:
                                        modelatlayer = list(modeldata[int(floor(depthlayer))])
                                    elif (floor(depthlayer) + 0.5) > depthlayer:
                                        modelatlayer = [findYPoint((floor(depthlayer)) - 0.5, int(floor(depthlayer)) + 0.5,
                                                                   modeldata.iloc[y, int(floor(depthlayer)) - 1],
                                                                   modeldata.iloc[y, int(floor(depthlayer))], depthlayer) for y in
                                                        range(0, len(modeldata))]
                                    else:
                                        modelatlayer = [
                                            findYPoint(round(depthlayer), round(depthlayer) + 1,
                                                       modeldata.iloc[y, int(floor(depthlayer))],
                                                       modeldata.iloc[y, floor(depthlayer) + 1], depthlayer) for y in
                                            range(0, len(modeldata))]

                                    if variable == "O2zt":
                                        modelatlayer = [element * 0.001 for element in modelatlayer]

                                    modelfinalresult['%s_Model_%s' % (modelresult, depthlevel)] = modelatlayer

                                    depth_layers[depthlevel] = depthlayer



                        #### Creation of the figures

                        Graphics(self.lake_name,outputfolder).comparison_obs_sims_plot(variable, calibration_methods,modelfinalresult, data,depth_layers,comparison_target, ice_cover,icecovertarea=True)


                    else:
                        print("comparison file of %s doesn't exist"%variable)
                except:
                    print("tttt")

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
                     columns=['obs', 'sim'])

    d["obs"] = d["obs"].astype(float)
    d["sim"] = d["sim"].astype(float)

    try:
        results = mean_squared_error(d["obs"], d["sim"])
        results = sqrt(results)
        resultsnormalise = sqrt(mean_squared_error(d["obs"], d["sim"])) / (max(d["obs"]) - min(d["obs"]))
    except:
        results = np.nan
        resultsnormalise = np.nan
    return results, resultsnormalise

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
        rsquare = r2_score(x, y)
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
    :return: Temperature at the depth yc.graphique()
    """
    m = (float(ya) - float(yb)) / (float(xa) - float(xb))
    yc = (float(xc) - (float(xb) + 0.5)) * m + float(yb)
    return yc

def final_equation_parameters(longitude, latitude, depthmax, depthmean, CL, SCL, Turnover, area, volume, Area_sed):
    # swa_b1 = (4.02808 + 0.04542 * CL - 0.50796 * longitude - 0.10301 * SCL - 0.19259 * depthmean + 0.04685 * Turnover
    #           + 0.03645 * depthmax + 0.07105 * latitude)

    swa_b1 = 10**( 13.6349  -0.3636 *log10(Turnover) -7.8264*log10(latitude))
    # swa_b0 = (-0.3666 + 3.4702 * sqrt(SCL))
    swa_b0 = 10**(2.8320 -0.4710*log(volume) -0.1689*sqrt(SCL) +0.1366*sqrt(depthmax) + 0.7736*log10(area))
    # c_shelter = (2.1544 - 0.2194 * log10(area)) ** 2
    c_shelter = (-1.752484 -0.006331*(Turnover) +0.040554 *(latitude)+0.003181*(CL))

    # i_scv = ((87.7621 - 166.1369 * log10(area) + 8.3947 * log10(CL) + 0.9242 * log10(Turnover) + 68.6099 * log(volume)
    #           - 1.3728 * sqrt(SCL) + 20.5551 * log(longitude) - 130.8638 * log10(depthmean) - 65.8479 * log10(
    #             latitude)) ** 2)
    i_scv = 10**(-2.3419+1.6295*log10(Turnover)-1.0389*sqrt(SCL) -0.5237*sqrt(depthmax)
                 -11.5569*log10(depthmean)+1.7065*log(longitude)-15.9455*log10(area)+0.9293*log10(CL)+6.6762*log(volume))

    # i_sct = 1.577e+01 - 4.499e-01 * SCL - 1.717e-09 * area - 1.975e-01 * latitude - 1.421e-02 * CL
    i_sct = (46.3645-0.1034*sqrt(depthmax)+1.9745*log10(Turnover)+14.1969*log(volume)-710.7061*log10(Area_sed)+671.3723*log10(area)
     -1.1202*sqrt(SCL) +0.7519*log10(CL)-20.3059*log10(depthmean))
    # (46.41172 -0.09901*sqrt(depthmax)+2.00148*log10(Turnover)+ 14.16320*log(volume)-711.74348*log10(Area_sed)+
    #      672.47314*log10(area)-1.11753*sqrt(SCL) +0.79024*log10(CL) -20.24508*log10(depthmean))

    if i_sct <= 0:
        i_sct = 0.01
    # i_sct = 0
    # i_scv = 1.15

    #i_sco = -7.428e-01 - 1.737e-11 * volume + 1.049e-01 * longitude + 1.269e-02 * depthmean + 1.218e-03 * i_scv
    i_sco = 0.38630+0.02608 *CL+0.25853 *SCL
    if i_sco <= 0:
        i_sco = 0.01

    # k_sod = math.exp(
    #     103.17131 - 0.24447 * log10(CL) - 50.57089 * log10(latitude) - 0.05483 * swa_b0 + 0.73326 * log(swa_b1)
    #     - 2.48570 * log(longitude) - 0.05313 * sqrt(i_scv))
    k_sod = math.exp(3.6879 +1.8295*log10(CL) -0.9509*sqrt(SCL)+2.4462*log10(i_sct) )

    if k_sod <= 0:
        k_sod = 100
    # k_bod = 10 ** (-53.5941 + 18.3879 * log(longitude) + 1.002 * log(swa_b1) + 0.3627 * sqrt(depthmax) + 0.105 * sqrt(
    #     i_scv))
    k_bod = 10**(-47.4540+15.4350*log(longitude) -0.5858*log(i_scv)+ 5.6893*log10(i_sct)-0.3248*(swa_b0))
    if k_bod <= 0:
        k_bod = 0.001


    # i_scdoc = (374.8321 - 163.3363 * log10(latitude) - 29.1505 * log(longitude) - 0.3314 * sqrt(i_scv))
    # i_scdoc = (11.8440-0.4454*i_scv-1.7905*swa_b0-0.6655*sqrt(depthmax))
    i_scdoc = (-27.756-7.515*log10(CL)-2.684 *swa_b0 -7.410*log10(Turnover)+18.220*log(longitude))

    if i_scdoc <= 0:
        i_scdoc = 4.75
    kzn0, albice, albsnow = 0.000574155, 0.242398444, 0.647403043

    if swa_b1 <= 0.01:
        swa_b1 = 0.01

    if swa_b0 <= 0:
        swa_b0 = 0.01
    return swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow

def Area_base_i(i, surface_area, max_depth, mean_depth):
    """
        Function calculating the proportional base area of the cylindric layer at a certain depth.
        equation from Lester et al. (2004).
        :param i: the depth where the cylindric is calculated (from 0(surface) to max_depth)
        :param surface_area: surface area of the lake
        :param max_depth: maximm depth of the lake
        :param mean_depth: mean depth of the lake
        :return: base area of the cylindric layer at i depth
        """
    return surface_area * (1 - (i / max_depth) ** basin_shape(mean_depth, max_depth)) ** 2

def basin_shape(mean_depth, max_depth):
    """
    estimate shape of the lake for calculating the sediment area and water volume
    :param mean_depth: mean depth of the lake
    :param max_depth: mmaximun depth of the lake
    :return: shape coefficient. When equals  1, estimated lake has a cone-shaped,
                                when less than 1, estimated a saucer-shaped lake,
                                and when greater than 1, estimated bowl-shaped lake.
    """
    r = mean_depth / max_depth
    return (3 * r + (r ** 2 + 8 * r) ** 0.5) / (4 * (1 - r))

def ice_cover_comparison(outputfolder = r"F:\output" ,
                         lake_list = r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList_only_validation_12lakes.csv",
                         sjolista = r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\sjolista.xlsx"):


    exA, y1A, exB, y1B = scenarios[2]
    m1, m2 = models[2]
    y2A = y1A + 4
    y2B = y1B + 4

    data_lake = pd.read_csv(lake_list, encoding='latin')
    data_sjolista = pd.read_excel(sjolista)
    data_sjolista.drop("X", axis=1, inplace=True)
    data_sjolista.drop("Y", axis=1, inplace=True)
    data_sjolista.drop("adjustment  factor", axis=1, inplace=True)
    data_sjolista.drop("type of problem", axis=1, inplace=True)
    data_sjolista.drop("Comment", axis=1, inplace=True)
    data_sjolista.drop("endd", axis=1, inplace=True)
    data_sjolista.drop("startd", axis=1, inplace=True)
    data_sjo = data_sjolista.dropna()
    lake_id = data_lake["ebhex"].tolist()
    lake_id2 = data_sjo['ebhex'].tolist()

    test = data_sjo.loc[data_sjo['ebhex'].isin(lake_id)]
    test = test.sort_values(by=['ebhex'])
    test2 = data_lake.loc[data_lake['ebhex'].isin(lake_id2)]
    test2 = test2.sort_values(by=['ebhex'])

    list_selected_lake = test["ebhex"].tolist()
    listmeanice = []
    for eh in list_selected_lake:

        eh = eh[2:] if eh[:2] == '0x' else eh
        while len(eh) < 6:
            eh = '0' + eh
        d1, d2, d3 = eh[:2], eh[:4], eh[:6]
        outdir = os.path.join(outputfolder, d1, d2, d3,
                              'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (
                                  m1, exA, exB, m2, y1A, y2B), "calibration_result_old")

        outdir
        try:
            ice = pd.read_csv(os.path.join(outdir, 'His.csv'), names=['1', '2', '3', '4', '5', '6', '7', '8'])
            meanice = (ice['7'].sum()) / 10
            listmeanice.append(meanice)
            print(listmeanice)
        except:
            listmeanice.append("")

    test['meanice'] = listmeanice
    print(test2['area'])
    test['area'] = test2['area'].tolist()
    test['Mean'] = test2['Mean'].tolist()
    test['volume'] = test2['volume'].tolist()

    print(test)

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    test.to_csv(r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\icemodelvsdata1001old_%s.csv" % timestr)
    print("eee")

    list_selected_lake = lake_id
    test = data_lake
    listmeanice = []
    for eh in list_selected_lake:

        eh = eh[2:] if eh[:2] == '0x' else eh
        while len(eh) < 6:
            eh = '0' + eh
        d1, d2, d3 = eh[:2], eh[:4], eh[:6]
        outdir = os.path.join(outputfolder, d1, d2, d3,
                              'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (
                                  m1, exA, exB, m2, y1A, y2B), "calibration_result_old")

        try:
            ice = pd.read_csv(os.path.join(outdir, 'His.csv'), names=['1', '2', '3', '4', '5', '6', '7', '8'])
            meanice = (ice['7'].sum()) / 10
            listmeanice.append(meanice)
            print(listmeanice)
        except:
            listmeanice.append("")

    test['meanice'] = listmeanice

    print(test)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    test.to_csv(r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\icemodelvsdata100111old_%s.csv" % timestr)
    print("eee")



if __name__ == "__main__":
    lake_info = LakeInfo("test")

# def graphique(x, y, xerr, yerr, r_value, slope, intercept, calibration=False,old=False):
#
#     sns.set(font_scale=2)
#     sns.set_style("ticks")
#
#     plt.grid(False)
#
#     colorpalette = sns.color_palette("dark", 10)
#     lineStart = 0
#     lineEnd = 20
#     fig, ax = plt.subplots(figsize=(15.0, 14.5))
#     plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color=colorpalette[9], label="y= x", linewidth=4)
#     plt.xlim(0, 15)
#     plt.ylim(0, 20)
#     (_, caps, _) = plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=colorpalette[3], markersize=8, capsize=20,
#                                 linewidth=4, elinewidth=4)
#     for cap in caps:
#         cap.set_markeredgewidth(4)
#
#     fig.suptitle("")
#     fig.tight_layout(pad=2)
#
#     x = smodels.add_constant(x)  # constant intercept term
#     # Model: y ~ x + c
#     model = smodels.OLS(y, x)
#     fitted = model.fit()
#     x_pred = np.linspace(x.min(), x.max(), 50)
#     x_pred2 = smodels.add_constant(x_pred)
#     y_pred = fitted.predict(x_pred2)
#
#     ax.plot(x_pred, y_pred, '-', color='k', linewidth=4,
#             label="linear regression (y = %0.3f x + %0.3f) \n R\u00b2 : %0.3f " % (slope, intercept, r_value))
#
#     print(fitted.params)  # the estimated parameters for the regression line
#     print(fitted.summary())  # summary statistics for the regression
#
#     y_hat = fitted.predict(x)  # x is an array from line 12 above
#     y_err = y - y_hat
#     mean_x = x.T[1].mean()
#     n = len(x)
#     dof = n - fitted.df_model - 1
#
#     t = stats.t.ppf(1 - 0.025, df=dof)
#     s_err = np.sum(np.power(y_err, 2))
#     conf = t * np.sqrt((s_err / (n - 2)) * (1.0 / n + (
#             np.power((x_pred - mean_x), 2) / ((np.sum(np.power(x_pred, 2))) - n * (np.power(mean_x, 2))))))
#     upper = y_pred + abs(conf)
#     lower = y_pred - abs(conf)
#     ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.4, label="Confidence interval")
#
#     sdev, lower, upper = wls_prediction_std(fitted, exog=x_pred2, alpha=0.025)
#     ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.1, label="Prediction interval")
#     plt.xlabel("Average Observed Secchi_Depth (m)")
#     plt.ylabel("Average Modeled Secchi Depth (m)")
#     plt.ylim()
#     plt.xlim()
#
#     ax.legend(loc='lower right')
#
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     if calibration:
#         fig.savefig('Secchi_mean_comparison_calibration_old_%s_%s.png' % (old,timestr), dpi=125)
#     else:
#         fig.savefig('Secchi_mean_comparison_old_%s_%s.png' % (old,timestr), dpi=125)
#     plt.close()
#
# def graphiqueTO(x, y, z, symbol, r_value, slope, intercept, variable, calibration=False,old=False, lakeid="",
#                 outputfolder=r'F:\output'):
#     sns.set(font_scale=2)
#     sns.set_style("ticks")
#     print(len(x), len(y), len(z))
#     if lakeid == "":
#         rmse, nrmse = root_mean_square([item for sublist in x for item in sublist],
#                                        [item for sublist in y for item in sublist])
#     else:
#         rmse, nrmse = root_mean_square(x, y)
#     plt.grid(False)
#
#     colorpalette = sns.color_palette("dark", 10)
#     lineStart = -1
#     if variable == "Temperature (°C)":
#         lineEnd = 28
#     else:
#         lineEnd = 28
#     fig, ax = plt.subplots(figsize=(15.0, 12))
#
#     if variable == "Temperature (°C)":
#         plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color=colorpalette[0], label="y= x", linewidth=4)
#         plt.xlim(-1, 28)
#         plt.ylim(-1, 28)
#         ccmap = 'Blues'
#     else:
#         plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color=colorpalette[3], label="y= x", linewidth=4)
#         plt.xlim(-1, 28)
#         plt.ylim(-1, 28)
#         ccmap = 'Reds'
#     markers = ["o", "v", "^", "s", "P", "*", ">", "X", "D", "<", "p", "d"]
#     if lakeid == "":
#         for i, c in enumerate(np.unique(symbol)):
#             try:
#                 cs = plt.scatter(x[i], y[i], c=z[i], marker=markers[c], s=90, cmap=ccmap, linewidths=1, edgecolors='k',
#                                  alpha=0.8)
#             except:
#                 print("error in")
#     else:
#         try:
#             i = int(symbol[0])
#         except:
#             i = 1
#         print(i)
#         if i > 11:
#             print("here")
#         print(markers[i])
#         cs = plt.scatter(x, y, c=z, marker=markers[i], s=90, cmap=ccmap, linewidths=1, edgecolors='k',
#                          alpha=0.8)
#     cb = plt.colorbar(cs)
#     plt.clim(0.0, 1.0)
#     cb.ax.tick_params(labelsize=14)
#
#     cb.ax.invert_yaxis()
#
#     fig.suptitle("")
#     fig.tight_layout(pad=2)
#     if lakeid == "":
#         x, y = [item for sublist in x for item in sublist], [item for sublist in y for item in sublist]
#
#     x = smodels.add_constant(x)  # constant intercept term
#     # Model: y ~ x + c
#     model = smodels.OLS(y, x)
#     fitted = model.fit()
#     x_pred = np.linspace(x.min(), x.max(), 50)
#     x_pred2 = smodels.add_constant(x_pred)
#     y_pred = fitted.predict(x_pred2)
#
#     ax.plot(x_pred, y_pred, '-', color='k', linewidth=4,
#             label="linear regression (y = %0.3f x + %0.3f) \n R\u00b2 : %0.3f RMSE: %0.3f" % (
#                 slope, intercept, r_value, rmse))
#
#     print(fitted.params)  # the estimated parameters for the regression line
#     print(fitted.summary())  # summary statistics for the regression
#
#     y_hat = fitted.predict(x)  # x is an array from line 12 above
#     y_err = y - y_hat
#     mean_x = x.T[1].mean()
#     n = len(x)
#     dof = n - fitted.df_model - 1
#
#     t = stats.t.ppf(1 - 0.025, df=dof)
#     s_err = np.sum(np.power(y_err, 2))
#     conf = t * np.sqrt((s_err / (n - 2)) * (1.0 / n + (
#             np.power((x_pred - mean_x), 2) / ((np.sum(np.power(x_pred, 2))) - n * (np.power(mean_x, 2))))))
#     upper = y_pred + abs(conf)
#     lower = y_pred - abs(conf)
#     ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.4, label="Confidence interval")
#
#     sdev, lower, upper = wls_prediction_std(fitted, exog=x_pred2, alpha=0.025)
#     ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.1, label="Prediction interval")
#     plt.xlabel("Average Observed %s" % variable)
#     plt.ylabel("Average Modeled %s" % variable)
#     plt.ylim()
#     plt.xlim()
#
#     ax.legend(loc='best')  # 'upper left')
#     import time
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     if calibration:
#         if variable == "Temperature (°C)":
#             fig.savefig(os.path.join(outputfolder, 'Temperature_comparison_calibrated_old_%s_%s_%s.png' % (old,lakeid, timestr)), dpi=125)
#
#         else:
#             fig.savefig(os.path.join(outputfolder, 'Oxygen_comparison_calibrated_old_%s_%s_%s.png' % (old,lakeid, timestr)),
#                         dpi=125)
#     else:
#         if variable == "Temperature (°C)":
#             fig.savefig(os.path.join(outputfolder, 'Temperature_comparison_old_%s_%s_%s.png' % (old,lakeid, timestr)),
#                         dpi=125)
#         else:
#             fig.savefig(os.path.join(outputfolder, 'Oxygen_comparison_old_%s_%s_%s.png' % (old,lakeid, timestr)),
#                         dpi=125)
#     plt.close()
# def taylor_target_plot(all_data_from_model, all_data_from_observation, label_method, variable, information,
#                        label_taylor):
#     """
#
#     :param all_data_from_model:
#     :param all_data_from_observation:
#     :param label_method:
#     :param variable:
#     :return:
#     """
#     # take list of lists
#     # [[[]*3]*12]
#     colorpalette = ['#7A0014', '#E7FFD6', '#4A7FBF']  # sns.color_palette('Accent_r', 3)
#     markers = ["o", "v", "^", "s", "P", "*", ">", "X", "D", "<", "p", "d"]
#     sizes = [200, 150, 100]
#     biaslist, crmsdlist, rmsdlist = [None] * (len(all_data_from_model)), [None] * (len(all_data_from_model)), [None] * (len(all_data_from_model))
#     sdevlist, crmsdtaylist, ccoeflist = [None] * (len(all_data_from_model) ), [None] * (len(all_data_from_model) ), [None] * (len(all_data_from_model) )
#     fig1, ax1 = plt.subplots(figsize=(10, 10), dpi=100)
#     print(len(all_data_from_model))
#     for lake in range(0, len(all_data_from_model)):
#         observations = all_data_from_observation[lake]
#         models = all_data_from_model[lake]
#         print(len(models))
#         statistic_target_lake = [None] * (len(models))
#         statistic_taylor_lake = [None] * (len(models)+1)
#         bias, crmsd, rmsd =  [None] * (len(models)),[None] * (len(models)),[None] * (len(models))
#         sdev, crmsdtay, ccoef = [None] * (len(models)+1),[None] * (len(models)+1),[None] * (len(models)+1)
#         for method in range(0, len(models)):
#             stats_target = sm.target_statistics(models[method], observations[method],norm=True)
#             stats_taylor = sm.taylor_statistics(models[method], observations[method])
#
#             # Calculate statistics for target and taylor diagram
#             bias[method] = stats_target['bias']
#             crmsd[method] = stats_target['crmsd']
#             rmsd[method] = stats_target['rmsd']
#
#             if method == 0:
#                 crmsdtay[method] = stats_taylor['crmsd'][0]
#                 ccoef[method] = stats_taylor['ccoef'][0]
#                 sdev[method] = stats_taylor['sdev'][0]/np.mean(observations[method])
#
#
#             crmsdtay[method+1] = stats_taylor['crmsd'][1]
#             ccoef[method+1] = stats_taylor['ccoef'][1]
#             sdev[method + 1] = stats_taylor['sdev'][1] / np.mean(models[method])
#
#             statistic_target_lake[method] = stats_target
#             statistic_taylor_lake[method] = stats_taylor
#
#         biaslist[lake], crmsdlist[lake], rmsdlist[lake] = bias, crmsd, rmsd
#         sdevlist[lake], crmsdtaylist[lake], ccoeflist[lake] = sdev, crmsdtay, ccoef
#         # Specify labels for points in a list (M1 for model prediction # 1,
#         # etc.).
#
#         # fig2, ax2 = plt.subplots(figsize=(15, 10), dpi=100)
#         # colorss = ['darkred']
#         # label_methodt = ["Observations"] + label_method
#         # fig2 = sm.taylor_diagram(np.array(sdev), np.array(crmsdtay), np.array(ccoef),
#         #                          markerLabel=label_methodt,
#         #                          markerLabelColor='r',
#         #                          markerColor='darkred', markerLegend='on',
#         #
#         #                          colRMS='dimgrey', styleRMS=':',
#         #                          titleRMS='on',
#         #                          colSTD='grey', styleSTD='-.',
#         #                           titleSTD='on',
#         #                          colCOR='k', styleCOR='--',
#         #                          titleCOR='on', markerSize=10, alpha=0.0)
#         # # fig2 =  sm.taylor_diagram(np.array(sdev), np.array(crmsdtay), np.array(ccoef),
#         # #               markerLabel = label_methodt,
#         # #               markerLabelColor = 'r',
#         # #               markerColor = 'r', markerLegend = 'on',
#         # #               tickRMS = range(0,3,1),
#         # #               colRMS = 'dimgrey', styleRMS = ':', widthRMS = 2.0,
#         # #               titleRMS = 'on', titleRMSDangle = 4.0, tickSTD = range(0,4,1),
#         # #               axismax = 4.0, colSTD = 'grey', styleSTD = '-.',
#         # #               widthSTD = 1.0, titleSTD = 'on',
#         # #               colCOR = 'k', styleCOR = '--', widthCOR = 1.0,
#         # #               titleCOR = 'on', markerSize = 10, alpha = 0.0)
#         # # fig2 = sm.taylor_diagram(np.array(sdev), np.array(crmsdtay), np.array(ccoef),
#         # #                          markerLabel=label_methodt, markerLabelColor='k',
#         # #                                            markerLegend='on', markerColor='y',
#         # #                                            styleOBS='-', colOBS='grey', markerobs='o',
#         # #                                            markerSize=6, tickRMS=[0.0, 1.0, 2.0, 3.0,4.0,5.0],
#         # #                                            tickRMSangle=115, showlabelsRMS='on',
#         # #                                            titleRMS='on', titleOBS='Ref', checkstats='on')
#         #     #sm.taylor_diagram(np.array(sdev), np.array(crmsdtay), np.array(ccoef),
#         #                       #    checkStats='on', styleOBS = '-', markerLabel = label_methodt, colOBS = 'blue',
#         #                       # markerobs = 'o', markerLegend = 'on',stylerms ='-',colRMS='grey',
#         #                       # titleOBS = 'Observation',showlabelsRMS='on', titleRMS = 'off',
#         #                       # colCOR='dimgrey', alpha=0.7)
#         #
#         # plt.show()
#         # timestr = time.strftime("%Y%m%d-%H%M%S")
#         # plt.savefig(
#         #     os.path.join(outputfolder, "Comparative_taylor_%s_%s_%s" % (variable, information, timestr)))
#         # plt.close()
#         if information == "lake_698":
#             print('here')
#         #symbol = [markers[lake]]*len(models)
#         if not information == "all_lakes":
#             maxv = abs(round_decimals_up(max([max(bias),max(crmsd)])))
#             minv = abs(round_decimals_down(min([min(bias), min(crmsd)])))
#             if minv> maxv:
#                 inverse = maxv
#                 maxv = minv
#                 minv = inverse
#             ax1.add_patch(plt.Circle((0, 0), maxv, color='k', fill=False,ls='--',zorder=0))
#             ax1.add_patch(plt.Circle((0, 0), 1, color='k', fill=False, ls='-', zorder=0))
#             ax1.add_patch(plt.Circle((0, 0), minv, color='k', fill=False,ls='--',zorder=0))
#             if abs((maxv)-(minv))> 0.2:
#                 test = ((minv)+round_decimals_up(abs((maxv)-(minv))/2))
#                 ax1.add_patch(plt.Circle((0, 0), ((minv)+round_decimals_up(abs((maxv)-(minv))/2)), color='k', fill=False,ls='--',zorder=0))
#             for i in range(len(crmsd)):
#                 fig1=plt.scatter(y=bias[i], x=crmsd[i], color=colorpalette[i], marker=markers[lake],label=label_method[i],zorder=100,alpha=0.8,s=sizes[i],edgecolors='k')
#             if lake == 0:
#                 plt.legend()
#             if (minv)< (maxv):
#                 if (maxv) < 1:
#                     limit = 1.5
#                 else:
#                     limit = (maxv+0.5)
#             else:
#                 if (minv) < 1:
#                     limit = 1.5
#                 else:
#                     limit = (minv + 0.5)
#             plt.ylim(-1 * limit, limit)
#             plt.xlim(-1 * limit, limit)
#         else:
#             for i in range(len(crmsd)):
#                 fig1=plt.scatter(y=bias[i], x=crmsd[i], color=colorpalette[i], marker=markers[lake],label=label_method[i],zorder=100,alpha=0.8,s=sizes[i],edgecolors='k')
#             if lake == 0:
#                 plt.legend()
#
#     if information == "all_lakes":
#         t1 = [max(i) for i in biaslist]
#         t2 = max(t1)
#         t3 = max([t2, max([max(i) for i in crmsdlist])])
#         maxv = abs(round_decimals_up(max([max([max(i) for i in biaslist]), max([max(i) for i in crmsdlist])])))
#         minv = abs(round_decimals_down(min([min([min(i) for i in biaslist]), min([min(i) for i in crmsdlist])])))
#         if minv > maxv:
#             inverse = maxv
#             maxv = minv
#             minv = inverse
#         ax1.add_patch(plt.Circle((0, 0), maxv, color='k', fill=False, ls='--', zorder=0))
#         ax1.add_patch(plt.Circle((0, 0), 1, color='k', fill=False, ls='-', zorder=0))
#         ax1.add_patch(plt.Circle((0, 0), minv, color='k', fill=False, ls='--', zorder=0))
#         if abs(abs(maxv) - abs(minv)) > 0.2:
#             test = (abs(minv) + round_decimals_up(abs(abs(maxv) - abs(minv)) / 2))
#             ax1.add_patch(plt.Circle((0, 0), (abs(minv) + round_decimals_up(abs(abs(maxv) - abs(minv)) / 2)), color='k',
#                                      fill=False, ls='--', zorder=0))
#         for i in range(len(crmsd)):
#             fig1 = plt.scatter(y=bias[i], x=crmsd[i], color=colorpalette[i], marker=markers[lake],
#                                label=label_method[i], zorder=100, alpha=0.8, s=sizes[i], edgecolors='k')
#
#         if abs(minv) < abs(maxv):
#             if abs(maxv) < 1:
#                 limit = 1.5
#             else:
#                 limit = abs(maxv + 0.5)
#
#         else:
#             if abs(minv) < 1:
#                 limit = 1.1
#             else:
#                 limit = abs(minv - 0.5)
#         plt.ylim(-1 * limit, limit)
#         plt.xlim(-1 * limit, limit)
#
#
#     sns.despine()
#     ax1.spines['left'].set_position('center')
#     ax1.spines['bottom'].set_position('center')
#     plt.xlabel('Normalized\n cRMSE', horizontalalignment='right', x=1)
#     # #ax.set_xlabel('Normalized cRMSE', loc='right')
#     plt.ylabel('Normalized Bias', verticalalignment="top", y=1)
#     #ax.set_ylabel('Normalized Bias')
#     ax1.xaxis.set_minor_locator(MultipleLocator(5))
#     ax1.yaxis.set_minor_locator(MultipleLocator(5))
#     ax1.set_aspect(1.0)
#
#     #plt.show()
#
#
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     plt.savefig(
#         os.path.join(outputfolder, "Comparative_target_%s_%s_%s" % ( variable,information, timestr)))
#     plt.close()
#
#     # if information == "all_lakes":
#     #     fig2, ax2 = plt.subplots(figsize=(10,10), dpi=100)
#     #     limitrmsmax = round_decimals_up(max([max(i) for i in crmsdtaylist])+1,0)
#     #     if max([max(i) for i in crmsdtaylist]) < 0:
#     #         limitrmsmin = round_decimals_down(min([min(i) for i in crmsdtaylist]),0)
#     #     else:
#     #         limitrmsmin=0
#     #     limitsdmax = round_decimals_up(max([max(i) for i in sdevlist]) + 1, 0)
#     #     if min([min(i) for i in sdevlist]) < 0:
#     #         limitsdmin = round_decimals_down(min([min(i) for i in sdevlist]), 0)
#     #     else:
#     #         limitsdmin = 0
#     #     for i in range(len(label_method)):
#     #         lakesdev = [sdev[i] for sdev in sdevlist]
#     #         lakecrmsd = [crmsd[i] for crmsd in crmsdtaylist]
#     #         lakeccoef = [ccoef[i] for ccoef in ccoeflist]
#     #         if i == 0:
#     #             fig2 = sm.taylor_diagram(np.array(lakesdev), np.array(lakecrmsd), np.array(lakeccoef),markerLabel = label_taylor,
#     #                       markerLabelColor = 'r',
#     #                       markerColor = 'r', markerLegend = 'on',
#     #                       tickRMS = range(limitrmsmin,limitrmsmax,1),
#     #                       colRMS = 'm', styleRMS = ':', widthRMS = 2.0,
#     #                       titleRMS = 'on', titleRMSDangle = 40.0, tickSTD = range(limitsdmin,limitsdmax,1),
#     #                       axismax = limitsdmax, colSTD = 'b', styleSTD = '-.',
#     #                       widthSTD = 1.0, titleSTD = 'on',
#     #                       colCOR = 'k', styleCOR = '--', widthCOR = 1.0,
#     #                       titleCOR = 'on', markerSize = 10, alpha = 0.0)
#     #         else:
#     #             fig2 = sm.taylor_diagram(np.array(lakesdev), np.array(lakecrmsd), np.array(lakeccoef), overlay='on',
#     #                               markerLabel=label_taylor, markerLabelColor='b',
#     #                               markerColor='b')
#     #         plt.show()
#     #     timestr = time.strftime("%Y%m%d-%H%M%S")
#     #     plt.savefig(
#     #         os.path.join(outputfolder, "Comparative_taylor_%s_%s_%s" % (variable, information, timestr)))
#     #     plt.close()
    # def comparison_obs_simsold(self, thermocline):
    #     """
    #     Opens the comparison file created by make_comparison_file, and prints the results of analysis functions.
    #     :return: Score, a float representing the overall performance of the current simulation.
    #     """
    #     if 1 == 1:
    #
    #         # try:
    #         if 1 == 1:
    #             # if self.runlake(2,2) == False:
    #             if os.path.exists("{}/Tztcompare.csv".format(self.calibration_path)):
    #                 # sns.set(font_scale=3)
    #                 sns.set_style("ticks")
    #                 sns.set_context("ticks", rc={"font.size": 40, "axes.titlesize": 40, "axes.labelsize": 40})
    #
    #                 plt.grid(False)
    #                 plt.figure(figsize=(20, 10))
    #                 axe = plt.subplot(111)
    #                 for path_data in [self.calibration_path, self.outdir]:
    #                     variable = "Temperature"
    #                     data = pd.read_csv("{}/Tztcompare.csv".format(path_data), header=None)
    #                     data.columns = ['Dates', 'Depth', 'Observed %s (°C)' % variable, 'Modeled %s (°C)' % variable]
    #                     data['Dates'] = pd.to_datetime(data['Dates'])
    #                     data['Dates1'] = pd.to_datetime(data['Dates'])
    #                     data.set_index('Dates1', inplace=True)
    #                     exA, y1A, exB, y1B = scenarios[2]
    #                     all_data = pd.read_csv("{}/Tzt.csv".format(path_data), header=None)
    #                     all_data.columns = ["%s" % (i) for i in np.arange(0.5, len(all_data.columns), 1)]
    #
    #                     dataunder = data[data['Depth'] < thermocline]
    #                     dataover = data[data['Depth'] >= thermocline]
    #
    #                     Dates_range = pd.date_range(start='1/1/%s' % y1A, end='31/12/%s' % (int(y1B) + 4))
    #                     Alldataunder = pd.DataFrame()
    #                     Alldataunder['Dates'] = Dates_range
    #                     # alldataover = [Dates_range]
    #                     Alldataover = pd.DataFrame()
    #                     Alldataover['Dates'] = Dates_range
    #                     # alldataunder = [Dates_range]
    #
    #                     depthunder = ['Dates']
    #                     depthover = ['Dates']
    #                     for i in data['Depth'].unique():
    #                         if i in np.arange(0.5, len(all_data), 1):
    #                             dataa = all_data['%s' % i].tolist()
    #                         else:
    #                             if (round(i) - floor(i)) == 0:
    #                                 dataa = [findYPoint(int(round(i)) - 1, int(round(i)),
    #                                                     all_data.iloc[y, int(round(i)) - 1],
    #                                                     all_data.iloc[y, int(round(i))], i) for y in
    #                                          range(0, len(all_data))]
    #                             else:
    #                                 dataa = [
    #                                     findYPoint(int(round(i)), int(round(i)) + 1, all_data.iloc[y, int(round(i))],
    #                                                all_data.iloc[y, int(round(i)) + 1], i) for y in
    #                                     range(0, len(all_data))]
    #                         if i < thermocline:
    #                             depthunder.append(i)
    #                             Alldataunder['%s' % i] = dataa
    #                             # alldataunder.append(dataa)
    #                             print('under:', len(depthunder))
    #
    #                         else:
    #                             depthover.append(i)
    #                             Alldataover['%s' % i] = dataa
    #                             # alldataover.append(dataa)
    #                             print('over', len(depthover))
    #
    #                     try:
    #                         year = dataover.iloc[0, 0].year
    #                         Alldataover = Alldataover.loc[
    #                             Alldataover['Dates'] >= datetime.datetime.strptime("%s-01-01" % (year - 1), '%Y-%m-%d')]
    #                         year = dataunder.iloc[0, 0].year
    #                         Alldataunder = Alldataunder.loc[
    #                             Alldataunder['Dates'] >= datetime.datetime.strptime("%s-01-01" % (year - 1),
    #                                                                                 '%Y-%m-%d')]
    #                     except:
    #                         print("1")
    #                     colorpalette = sns.color_palette("colorblind")
    #
    #                     depthrange = list(map(float, depthover[1:]))
    #
    #                     for i in range(0, len(depthrange)):
    #                         # sns.lineplot(x="Dates", y='Observed %s (°C)' % variable, data=dataunder, color="black")
    #                         # d = colorpalette[int(i)]
    #                         # dd = "%s"%(depthrange[i])
    #                         if i in [0, (len(depthrange) - 1)]:
    #
    #                             if path_data == self.calibration_path:
    #                                 try:
    #                                     sns.lineplot(x="Dates", y=depthrange[i], data=Alldataunder,
    #                                                  color=colorpalette[int(i)], zorder=-1,
    #                                                  label='M%s' % (1))  # float(depthrange[0])))
    #                                 except:
    #                                     print('error layer')
    #                             else:
    #                                 try:
    #                                     sns.lineplot(x="Dates", y=depthrange[i], data=Alldataunder,
    #                                                  color=colorpalette[int(i)], zorder=-1, dashes=[(2, 2), (2, 2)],
    #                                                  label='M%s' % (1))  # float(depthrange[0])))
    #                                 except:
    #                                     print('error layer')
    #                         # sns.scatterplot(x="Dates", y="%s"%i, data=Alldataunder, markers="-o",color=colorpalette[floor(float(i))], label='Modeled %s(%s m)' % (variable, floor(float(i))))
    #                         # sns.scatterplot(x="Dates", y='Modeled %s (°C)' % variable, data=dataunder, markers="-o-",
    #                         #                color="black",
    #                         #                label='Observation %s (0-%s m)' % (variable, thermocline))
    #
    #                 sns.scatterplot(x="Dates", y='Observed %s (°C)' % variable, data=dataunder, markers="-o-", s=100,
    #                                 zorder=10,
    #                                 color="blue",
    #                                 label='O>%s' % (thermocline))
    #                 plt.xticks(rotation=15)
    #                 ax = plt.axes()
    #                 ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    #                 plt.ylabel("%s" % variable)
    #                 # Put a legend to the right of the current axis
    #                 axe.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #                 s3year = dataunder.iloc[0, 0].year
    #                 e3year = dataunder.iloc[-1, 0].year
    #                 s3y = datetime.datetime.strptime("%s-01-01" % (s3year), '%Y-%m-%d')
    #                 e3y = datetime.datetime.strptime("%s-01-01" % (e3year), '%Y-%m-%d')
    #                 plt.xlim(s3y, e3y)
    #                 plt.savefig("comparison_%s_%s_epi.png" % (self.lake_name, variable))
    #                 plt.close()
    #                 # sns.set(font_scale=5)
    #                 # sns.set_style("ticks")
    #                 sns.set_context("paper", font_scale=4)
    #                 plt.grid(False)
    #                 plt.figure(figsize=(20, 10))
    #                 axe = plt.subplot(111)
    #
    #                 for i in depthover[1:]:
    #                     # sns.lineplot(x="Dates", y='Observed %s (°C)' % variable, data=dataunder, color="black")
    #                     sns.lineplot(x="Dates", y="%s" % i, data=Alldataover, color='black', zorder=-1,
    #                                  label='M%s' % (floor(float(i))))
    #                     # sns.scatterplot(x="Dates", y="%s"%i, data=Alldataunder, markers="-o",color=colorpalette[floor(float(i))], label='Modeled %s(%s m)' % (variable, floor(float(i))))
    #                     # sns.scatterplot(x="Dates", y='Modeled %s (°C)' % variable, data=dataunder, markers="-o-",
    #                     #                color="black",
    #                     #                label='Observation %s (0-%s m)' % (variable, thermocline))
    #                 sns.scatterplot(x="Dates", y='Observed %s (°C)' % variable, data=dataover, markers="-o-", s=100,
    #                                 zorder=10,
    #                                 color="red",
    #                                 label='O%s-%s' % (thermocline, depthover[-1]))
    #                 plt.xticks(rotation=15)
    #                 ax = plt.axes()
    #                 ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    #                 plt.ylabel("%s" % variable)
    #                 axe.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #                 try:
    #                     syear = dataover.iloc[0, 0].year
    #                     eyear = dataover.iloc[-1, 0].year
    #                     sy = datetime.datetime.strptime("%s-01-01" % (syear), '%Y-%m-%d')
    #                     ey = datetime.datetime.strptime("%s-01-01" % (eyear), '%Y-%m-%d')
    #                     plt.xlim(sy, ey)
    #                 except:
    #                     print('1')
    #
    #                 plt.savefig("comparison_%s_%s_hypo.png" % (self.lake_name, variable))
    #                 plt.close()
    #
    #             if os.path.exists("{}/O2ztcompare.csv".format(self.calibration_path)):
    #
    #                 for path_data in [self.calibration_path, self.outdir]:
    #                     Alldataover, Alldataunder, dataunder, dataover = [], [], [], []
    #                     variable = "Oxygen"
    #                     data = pd.read_csv("{}/O2ztcompare.csv".format(path_data), header=None)
    #                     data.columns = ['Dates', 'Depth', 'Observed %s (°C)' % variable, 'Modeled %s (°C)' % variable]
    #                     data['Dates'] = pd.to_datetime(data['Dates'])
    #                     data['Dates1'] = pd.to_datetime(data['Dates'])
    #                     data.set_index('Dates1', inplace=True)
    #                     exA, y1A, exB, y1B = scenarios[2]
    #                     all_data = pd.read_csv("{}/O2zt.csv".format(path_data), header=None)
    #                     all_data.columns = ["%s" % (i) for i in np.arange(0.5, len(all_data.columns), 1)]
    #
    #                     dataunder = data[data['Depth'] < thermocline]
    #                     dataover = data[data['Depth'] >= thermocline]
    #
    #                     Dates_range = pd.date_range(start='1/1/%s' % y1A, end='31/12/%s' % (int(y1B) + 4))
    #                     Alldataunder = pd.DataFrame()
    #                     Alldataunder['Dates'] = Dates_range
    #                     # alldataover = [Dates_range]
    #                     Alldataover = pd.DataFrame()
    #                     Alldataover['Dates'] = Dates_range
    #                     # alldataunder = [Dates_range]
    #
    #                     depthunder = ['Dates']
    #                     depthover = ['Dates']
    #                     for i in data['Depth'].unique():
    #                         if i in np.arange(0.5, len(all_data), 1):
    #                             dataa = all_data['%s' % i].tolist()
    #
    #                         else:
    #                             if (round(i) - floor(i)) == 0:
    #                                 dataa = [findYPoint(int(round(i)) - 1, int(round(i)),
    #                                                     all_data.iloc[y, int(round(i)) - 1],
    #                                                     all_data.iloc[y, int(round(i))], i) for y in
    #                                          range(0, len(all_data))]
    #                             else:
    #                                 dataa = [
    #                                     findYPoint(int(round(i)), int(round(i)) + 1, all_data.iloc[y, int(round(i))],
    #                                                all_data.iloc[y, int(round(i)) + 1], i) for y in
    #                                     range(0, len(all_data))]
    #                             # dataa = [float(i)*0.001 for i in dataa]
    #                         dataa = [element * 0.001 for element in dataa]
    #                         if i < thermocline:
    #                             depthunder.append(i)
    #                             Alldataunder['%s' % i] = dataa
    #                             # alldataunder.append(dataa)
    #                             print('under:', len(depthunder))
    #
    #                         else:
    #                             depthover.append(i)
    #                             Alldataover['%s' % i] = dataa
    #                             # alldataover.append(dataa)
    #                             print('over', len(depthover))
    #
    #                     try:
    #                         year = dataover.iloc[0, 0].year
    #                         Alldataover = Alldataover.loc[
    #                             Alldataover['Dates'] >= datetime.datetime.strptime("%s-01-01" % (year - 1), '%Y-%m-%d')]
    #                         year = dataunder.iloc[0, 0].year
    #                         Alldataunder = Alldataunder.loc[
    #                             Alldataunder['Dates'] >= datetime.datetime.strptime("%s-01-01" % (year - 1),
    #                                                                                 '%Y-%m-%d')]
    #                     except:
    #                         print("1")
    #                     colorpalette = sns.color_palette("colorblind")
    #
    #                 sns.set(font_scale=2)
    #                 sns.set_style("ticks")
    #                 sns.set_context("paper", font_scale=4)
    #
    #                 plt.grid(False)
    #                 plt.figure(figsize=(20, 10))
    #                 axe = plt.subplot(111)
    #
    #                 for i in depthunder[1:]:
    #                     # sns.lineplot(x="Dates", y='Observed %s (°C)' % variable, data=dataunder, color="black")
    #                     sns.lineplot(x="Dates", y="%s" % i, data=Alldataunder, color="black", zorder=-1,
    #                                  label='M%s' % (floor(float(i))))
    #                     # sns.scatterplot(x="Dates", y="%s"%i, data=Alldataunder, markers="-o",color=colorpalette[floor(float(i))], label='Modeled %s(%s m)' % (variable, floor(float(i))))
    #                     # sns.scatterplot(x="Dates", y='Modeled %s (°C)' % variable, data=dataunder, markers="-o-",
    #                     #                color="black",
    #                     #                label='Observation %s (0-%s m)' % (variable, thermocline))
    #                 sns.scatterplot(x="Dates", y='Observed %s (°C)' % variable, data=dataunder, markers="-o-", s=100,
    #                                 zorder=10,
    #                                 color="blue",
    #                                 label='O>%s' % (thermocline))
    #                 plt.xticks(rotation=15)
    #                 ax = plt.axes()
    #                 ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    #                 plt.ylabel("%s" % variable)
    #                 # Put a legend to the right of the current axis
    #                 axe.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #                 s2year = dataunder.iloc[0, 0].year
    #                 e2year = dataunder.iloc[-1, 0].year
    #                 s2y = datetime.datetime.strptime("%s-01-01" % (s2year), '%Y-%m-%d')
    #                 e2y = datetime.datetime.strptime("%s-01-01" % (e2year), '%Y-%m-%d')
    #                 plt.xlim(s2y, e2y)
    #                 plt.savefig("comparison_%s_%s_epi.png" % (self.lake_name, variable))
    #                 plt.close()
    #                 sns.set(font_scale=2)
    #                 sns.set_style("ticks")
    #                 sns.set_context("paper", font_scale=4)
    #                 plt.grid(False)
    #                 plt.figure(figsize=(20, 10))
    #                 axe = plt.subplot(111)
    #
    #                 for i in depthover[1:]:
    #                     # sns.lineplot(x="Dates", y='Observed %s (°C)' % variable, data=dataunder, color="black")
    #                     sns.lineplot(x="Dates", y="%s" % i, data=Alldataover, color='black', zorder=-1,
    #                                  label='M%s' % (floor(float(i))))
    #                     # sns.scatterplot(x="Dates", y="%s"%i, data=Alldataunder, markers="-o",color=colorpalette[floor(float(i))], label='Modeled %s(%s m)' % (variable, floor(float(i))))
    #                     # sns.scatterplot(x="Dates", y='Modeled %s (°C)' % variable, data=dataunder, markers="-o-",
    #                     #                color="black",
    #                     #                label='Observation %s (0-%s m)' % (variable, thermocline))
    #                 sns.scatterplot(x="Dates", y='Observed %s (°C)' % variable, data=dataover, markers="-o-", s=100,
    #                                 zorder=10,
    #                                 color="red",
    #                                 label='O%s-%s' % (thermocline, depthover[-1]))
    #                 plt.xticks(rotation=15)
    #                 ax = plt.axes()
    #                 ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    #                 plt.ylabel("%s" % variable)
    #                 axe.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #                 try:
    #                     s1year = dataover.iloc[0, 0].year
    #                     e1year = dataover.iloc[-1, 0].year
    #                     s1y = datetime.datetime.strptime("%s-01-01" % (s1year), '%Y-%m-%d')
    #                     e1y = datetime.datetime.strptime("%s-01-01" % (e1year), '%Y-%m-%d')
    #                     plt.xlim(s1y, e1y)
    #                 except:
    #                     print("1")
    #                 plt.savefig("comparison_%s_%s_hypo.png" % (self.lake_name, variable))
    #                 plt.close()
    #         else:
    #             print("issues with the complete run")
    #         # except:
    #         print('error lake')

    # def comparison_obs_sims(self, thermocline, calibration=False,outputfolder=r'F:\output'):
    #
    #     if calibration:
    #         outfolder = self.calibration_path
    #     else:
    #         outfolder = self.outdir
    #     fig, ax = plt.subplots(1, 2, figsize=(30, 12))
    #     fig, ax = plt.subplots( figsize=(15, 6))
    #
    #     plotplacex = {"Tzt": 0, "O2zt": 1}
    #     if self.lake_name == 14939:
    #         print("here")
    #     for variable in ["O2zt"]: #"Tzt" ,
    #         # try:
    #         print(outfolder)
    #         if os.path.exists(os.path.join(outfolder, "%scompare.csv" % variable)):
    #             data = pd.read_csv(os.path.join(outfolder, "%scompare.csv" % variable), header=None,
    #                                names=['Date', 'Depth', 'Observations', 'Modelisation'])
    #
    #             data['Dates'] = pd.to_datetime(data["Date"])
    #             data = data.set_index(data['Dates'])
    #             initial_date = data['Dates'].min()
    #             final_date = data['Dates'].max()
    #             depth_range = data["Depth"].unique()
    #
    #             start = 2001
    #             end = 2010
    #             reds = sns.color_palette("rocket")
    #             blues = sns.color_palette("mako")
    #             characteristics = {"surface": [reds[2], reds[4]], "deepwater": [blues[1], blues[4]]}
    #             plotplacey = {"surface": 1, "deepwater": 0}
    #             markerstyle = {"surface": "o", "deepwater": "s"}
    #             markercolor = {"surface": "None", "deepwater": "black"}
    #             for modelresult in ['calculated']:  # ,'estimated']:
    #                 if modelresult == 'calculated':
    #                     modeldata = pd.read_csv(os.path.join(self.calibration_path, "%s.csv" % variable), header=None)
    #                 else:
    #                     modeldata = pd.read_csv(os.path.join(self.outdir, "%s.csv" % variable), header=None)
    #
    #                 dates = pd.date_range(start='1/1/%s' % start, end='12/31/%s' % end)
    #                 if len(dates) == len(modeldata):
    #                     modeldata = modeldata.set_index(dates)
    #                 else:
    #                     dates = pd.date_range(start='1/1/%s' % self.start_year, end='12/31/%s' % self.end_year)
    #                     modeldata = modeldata.set_index(dates)
    #
    #                 modeldata = modeldata.loc["%s-01-01" % str(initial_date.year):"%s-12-31" % str(final_date.year)]
    #
    #                 for depthlevel in ['surface', 'deepwater']:
    #                     if depthlevel == 'surface':
    #                         subdatabydepth = [depth for depth in depth_range if depth < thermocline]
    #                     else:
    #                         subdatabydepth = [depth for depth in depth_range if depth >= thermocline]
    #
    #                     # Select depth representing surface and deepwater
    #
    #                     numberbydepth = list(
    #                         data[data['Depth'].isin(subdatabydepth)].groupby(['Depth']).count()['Observations'])
    #
    #                     if len(numberbydepth) == 1:
    #                         depthlayer = subdatabydepth[0]
    #                     elif len(numberbydepth) < 1:
    #                         depthlayer = "no data"
    #                     else:
    #                         max_position = [i for i, x in enumerate(numberbydepth) if x == max(numberbydepth)]
    #                         max1 = max_position[0]
    #                         if depthlevel == "deepwater":
    #                             depthlayer = float(subdatabydepth[max_position[-1]])
    #                         else:
    #                             depthlayer = float(subdatabydepth[max_position[0]])
    #                         # depthlayer = subdatabydepth[numberbydepth.index(max(numberbydepth)]
    #
    #                     if depthlayer != "no data":
    #                         sns.set_style("ticks", {"xtick.major.size": 100, "ytick.major.size": 100})
    #                         if (floor(depthlayer) + 0.5) == depthlayer:
    #                             modelatlayer = list(modeldata[int(floor(depthlayer))])
    #                         elif (floor(depthlayer) + 0.5) > depthlayer:
    #                             modelatlayer = [findYPoint((floor(depthlayer)) - 0.5, int(floor(depthlayer)) + 0.5,
    #                                                        modeldata.iloc[y, int(floor(depthlayer)) - 1],
    #                                                        modeldata.iloc[y, int(floor(depthlayer))], depthlayer) for y in
    #                                             range(0, len(modeldata))]
    #                         else:
    #                             modelatlayer = [
    #                                 findYPoint(round(depthlayer), round(depthlayer) + 1,
    #                                            modeldata.iloc[y, int(floor(depthlayer))],
    #                                            modeldata.iloc[y, floor(depthlayer) + 1], depthlayer) for y in
    #                                 range(0, len(modeldata))]
    #
    #                         if variable == "O2zt":
    #                             modelatlayer = [element * 0.001 for element in modelatlayer]
    #
    #                         if modelresult == 'calculated':
    #
    #                             date = pd.date_range(start="%s-01-01" % str(initial_date.year),
    #                                                  end="%s-12-31" % str(final_date.year))
    #
    #                             modelfinalresult = pd.DataFrame(modelatlayer, index=date,
    #                                                             columns=['%s_Model_%s' % (modelresult, depthlevel)])
    #                             modelfinalresult['Dates'] = date
    #                         else:
    #                             modelfinalresult['%s_Model_%s' % (modelresult, depthlevel)] = modelatlayer
    #
    #                         if modelresult == "estimated":
    #
    #                             sns.lineplot(x="Dates", y="%s_Model_%s" % (modelresult, depthlevel), zorder=0,
    #                                          linewidth=1.5, color=characteristics[depthlevel][0], dashes=[(2, 2)],
    #                                          data=modelfinalresult)  # ,ax=ax[plotplacex[variable]])
    #                         else:
    #
    #                             sns.lineplot(x="Dates", y="%s_Model_%s" % (modelresult, depthlevel), zorder=0,
    #                                          linewidth=1.5,
    #                                          color=characteristics[depthlevel][1],
    #                                          data=modelfinalresult)  # ,ax=ax[plotplacex[variable]])
    #                             data2 = data.loc[data['Depth'] == depthlayer]
    #                             sns.scatterplot(x='Dates', y="Observations", data=data2, color='black', zorder=1,
    #                                             marker=markerstyle[depthlevel], s=50, facecolors=markercolor[depthlevel],
    #                                             edgecolor='k',
    #                                             linewidth=2,
    #                                             linestyle='-')  # ,ax=ax[plotplacex[variable]])
    #             # except:
    #             #     print("error with variable %s"%variable)
    #             # facecolors = markercolor[depthlevel], edgecolors = 'black',
    #
    #             plt.xticks(rotation=15)
    #
    #             # plt.title('seaborn-matplotlib example')
    #             timestr = time.strftime("%Y%m%d-%H%M%S")
    #             # plt.show()
    #             plt.savefig(os.path.join(outputfolder,"Comparative_timeline_%s_%s_%s" % (self.lake_name,variable, timestr)))
    #             plt.close()
    #     else:
    #         print("comparison file of %s doesn't exist"%variable)

    # def comparison_obs_sims_plot(self, variable_analized, calibration_methods,modeldata, obsdata,depthlayers,comparison_target,ice_cover):
    #
    #     fig,  axs = plt.subplots(2,1, sharex= True,constrained_layout = True, figsize=(15,8),gridspec_kw={'height_ratios': [1, 1]})
    #     reds = sns.color_palette("rocket")
    #     blues = sns.color_palette("mako")
    #     characteristics = {"surface": ["black","#B30000", "#FF9785"], "deepwater": ["black",'#14218F', "#0AEFFF"]}
    #     plotplacey = {"surface": 0, "deepwater": 1}
    #     markerstyle = {"surface": "o", "deepwater": "s"}
    #     markercolor = {"surface": "#B30000", "deepwater": '#14218F'}
    #     sns.set_style("ticks", {"xtick.major.size": 100, "ytick.major.size": 100})
    #
    #     plt.xticks(rotation=15)
    #     allmodels = []
    #     for depth_level in plotplacey:
    #         try:
    #             level = plotplacey[depth_level]
    #             depthlayer = depthlayers[depth_level]
    #             ax2 = axs[level].twinx()
    #             ax2.fill_between(ice_cover.iloc[:, 1].tolist(), ice_cover.iloc[:, 0].tolist(),
    #                              color="grey", alpha=0.2,zorder=-10)
    #
    #             ax2.set_ylim(0, 1)
    #             ax2.set_yticklabels([])
    #             ax2.yaxis.set_visible(False)
    #
    #             for method in calibration_methods:
    #                 allmodels.append(modeldata["%s_Model_%s" % (method, depth_level)].tolist())
    #                 if method == "estimated" :
    #
    #                     sns.lineplot(x="Dates", y="%s_Model_%s" % (method, depth_level),
    #                                  linewidth=1, color=characteristics[depth_level][2],
    #                                  data=modeldata, ax=axs[level],zorder=90)  # ,ax=ax[plotplacex[variable]])
    #                     axs[level].lines[2].set_linestyle("-.")
    #
    #
    #                 elif method == "old_calculated":
    #                     sns.lineplot(x="Dates", y="%s_Model_%s" % (method, depth_level),
    #                                  linewidth=1.5, color=characteristics[depth_level][1],
    #                                  data=modeldata, ax=axs[level],zorder=80)  # ,ax=ax[plotplacex[variable]])
    #
    #                     axs[level].lines[1].set_linestyle("--")
    #                     axs[level].set_ylim(-0.5, 30)
    #
    #                 else:
    #
    #                     observations = obsdata.loc[obsdata['Depth'] == depthlayer]
    #                     sns.scatterplot(x='Dates', y="Observations", data=observations, color='black',
    #                                     marker=markerstyle[depth_level], s=50, facecolors=markercolor[depth_level],
    #                                     edgecolor='k',
    #                                     linewidth=2,
    #                                     linestyle='-', ax=axs[level], zorder=100)
    #                     sns.lineplot(x="Dates", y="%s_Model_%s" % (method, depth_level),
    #                                  linewidth=2, color=characteristics[depth_level][0],
    #                                  data=modeldata, ax=axs[level],alpha=0.8,zorder = 60)
    #
    #
    #                     axs[level].set_ylim(-0.5, 30)
    #
    #
    #
    #             axs[level].set_ylim(-0.5, 30)
    #             axs[level].set_xlim(modeldata['Dates'].min(), modeldata['Dates'].max())
    #             axs[level].legend(["Second GA Calibration","First GA Calibration","Stewise Regression","Observations"])
    #         except:
    #             print("missing layer %s for lake %s"%(depth_level,self.lake_name))
    #     # plt.title('timeline_lake_%s_%s'%(self.lake_name,variable_analized)))
    #     timestr = time.strftime("%Y%m%d-%H%M%S")
    #     # plt.show()
    #     plt.savefig(os.path.join(outputfolder, "Comparative_timeline_%s_%s_%s" % (self.lake_name, variable_analized, timestr)))
    #     plt.close('all')
    #
    #
    #
    #     # Calculate statistics for target diagram
    #
    #     # if len(calibration_methods) >= 2:
    #     #
    #     #     bias = np.array([target_stats_esti['bias'], target_stats_old['bias'],
    #     #                      target_stats_cali['bias']])
    #     #     crmsd = np.array([target_stats_esti['crmsd'], target_stats_old['crmsd'],
    #     #                       target_stats_cali['crmsd']])
    #     #     rmsd = np.array([target_stats_esti['rmsd'], target_stats_old['rmsd'],
    #     #                      target_stats_cali['rmsd']])
    #     #
    #     #     # Specify labels for points in a list (M1 for model prediction # 1,
    #     #     # etc.).
    #     #     label = ['Estimation', 'old Calibration', 'New Calibration']
    #     #
    #     #     sm.target_diagram(bias, crmsd, rmsd, markerLabel=label)#,ticks=np.arange(-50, 60, 10))
    #     #
    #     #     # # maybe one taylor by type (one color by type) and all lake by taylor
    #     #     # from vacumm.misc.plot import taylor
    #     #     # taylor(allmodels, obsdata.tolist(), figsize=(8, 8), label_size='large', size=15,
    #     #     #        labels=calibration_methods, colors='cyan', title_size=18,
    #     #     #        savefigs=__file__, savefigs_pdf=True, show=False, close=True)
    #     #
    #     #     plt.savefig(
    #     #         os.path.join(outputfolder, "Comparative_target_%s_%s_%s" % (self.lake_name, variable_analized, timestr)))
    #     #
    #     #     plt.close()
    #     #
    #     #
    #     #     # # Set the figure properties (optional)
    #     #     # rcParams["figure.figsize"] = [8.0, 6.4]
    #     #     # rcParams['lines.linewidth'] = 1  # line width for plots
    #     #     # rcParams.update({'font.size': 12})  # font size of axes text
    #     #
    #     #     label = ["Obs",'Estimation', 'old Calibration', 'New Calibration']
    #     #
    #     #     # Store statistics in arrays
    #     #     sdev = np.array([ taylor_stats_cali['sdev'][0],taylor_stats_esti['sdev'][1], taylor_stats_old['sdev'][1],
    #     #                      taylor_stats_cali['sdev'][1]])
    #     #     crmsd = np.array([ taylor_stats_cali['crmsd'][0],taylor_stats_esti['crmsd'][1], taylor_stats_old['crmsd'][1],
    #     #                       taylor_stats_cali['crmsd'][1]])
    #     #     ccoef = np.array([ taylor_stats_cali['ccoef'][0],taylor_stats_esti['ccoef'][1], taylor_stats_old['ccoef'][1],
    #     #                       taylor_stats_cali['ccoef'][1]])
    #     #
    #     #     sm.taylor_diagram(sdev,crmsd,ccoef, checkStats='on', styleOBS = '-', markerLabel = label, colOBS = 'blue',
    #     #                       markerobs = 'o', markerLegend = 'on',stylerms ='-',colRMS='grey',
    #     #                       titleOBS = 'Observation',showlabelsRMS='on', titleRMS = 'off',
    #     #                       colCOR='dimgrey', alpha=0.7)
    #     #
    #     #     #sm.taylor_diagram(sdev, crmsd, ccoef,  markerLabel=label)
    #     #
    #     #     # Write plot to file
    #     #     plt.savefig( os.path.join(outputfolder, "Comparative_taylor_%s_%s_%s" % (self.lake_name, variable_analized, timestr)))
    #     #     plt.close()
    #     #     print("end")