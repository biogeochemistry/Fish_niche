#!/usr/bin/env python

""" Script for MyLake - ISIMIP
The main script launching the calibration and run of each lake with MyLake model.
"""

__author__ = "Julien Bellavance and Marianne Cote"

import multiprocessing
import os
import numpy as np
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import linregress

from lakes.Graphics import Graphics
from lakes.calibration_fish_niche import CalibrationInfo
from lakes.lake_information import LakeInfo, final_equation_parameters


num_cores = multiprocessing.cpu_count() - 6
matlab_directory = r"C:\Program Files\MATLAB\R2019b\bin\matlab"
output_path = r"Postproc"
input_folder = r"lakes/sweden_inflow_data/Validation_data_for_lookup.xlsx"
forcing_data_path = r"weather_cordex"


lakes = pd.read_csv("lakes/2017SwedenList.csv",
                    encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
lakes_data = lakes.set_index("lake_id").to_dict()
lakes_list = list(lakes_data.get("name").keys())
subid = list(lakes_data.get("subid").keys())

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

input_variables = ["hurs",
                   "pr",
                   "ps",
                   "rsds",
                   "sfcWind",
                   "tas"
                   ]
output_variables = ["Tzt.csv", "O2zt.csv", "lambdazt.csv", "DOCzt.csv", "Qst.csv", "Attn_zt.csv", "PARzt.csv",
                    "His.csv"]

output_unit = {"Tzt.csv": "deg C", "O2zt.csv": "mg m-2", "lambdazt.csv": "m-1", "DOCzt.csv": "mg m-3",
               "Qst.csv": "W m-2", "Attn_zt.csv": "-", "PARzt.csv": "umol m-2 s-1",
               "His.csv": {"Hi": "m", "Hs": "m", "Hsi": "m", "Tice": "deg C", "Tair": "deg C", "rho_snow": "kg m-3",
                           "IceIndicator": "-"}}

output_long = {"Tzt.csv": "Predicted temperature profile", "O2zt.csv": "Predicted dissolved oxygen profile",
               "lambdazt.csv": "Predicted average total light attenuation coefficient down to depth z",
               "DOCzt.csv": "Predicted dissolved organic carbon (DOC) profile",
               "Qst.csv": "Estimated surface heat fluxes",
               "Attn_zt.csv": "", "PARzt.csv": "Predicted irradiance down to depth z",
               "His.csv": {"Hi": "total ice thickness", "Hs": "snow thickness", "Hsi": "snow ice thickness",
                           "Tice": "ice surface temperature", "Tair": "surface temperature",
                           "rho_snow": "Initial snow density", "IceIndicator": "Indicator of ice cover"}}


def calibration_iteration(lakesList12 = "lakes/2017SwedenList_only_validation_12lakes.csv",
                          lakes_sample_for_temperature=False, lakes_sample_for_oxygen=False):
    """
    Simple function to call the calibration of all lakes.
    :param lakesList12: file name and directory (if not in same directory as this script) for the list of all the lakes
     with enough data to be calibrated
    :param lakes_sample_for_temperature: subsample of lakes from lakesList12(list of row's number) include in the
     temperature's calibration. if False, all lakes are used.
    :param lakes_sample_for_oxygen:  subsample of lakes from lakesList12(list of row's number) include in the oxygen's
    calibration. if False, all lakes are used.
    :return: 0 if function present error during the run. 1 if the function run correctly.
    """

    print(num_cores)
    try:
        lakes = pd.read_csv(lakesList12, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')

    except FileNotFoundError:
        print("Calibration cannot be done since the file containing the list of lakes do not exist. \n"
              "Verify file name and directory \n"
              "String given: %s" % lakesList12)
        return 0

    if ((isinstance(lakes_sample_for_temperature, list) or lakes_sample_for_temperature is False) and (
            isinstance(lakes_sample_for_oxygen, list) or lakes_sample_for_oxygen is False)):
        if lakes_sample_for_temperature is not False and not (all(isinstance(item, int) for item in lakes_sample_for_temperature)):
            print("Calibration cannot be done since at least one of the two lakes_sample variables are not a list of intergreters. \n List given: Temperature = %s, Oxygen = %s" % (lakes_sample_for_temperature, lakes_sample_for_oxygen))
            return 0
        if lakes_sample_for_oxygen is not False and not (
                all(isinstance(item, int) for item in lakes_sample_for_temperature)):
            print(
                "Calibration cannot be done since at least one of the two lakes_sample variables are not a list of intergreters. \n List given: Temperature = %s, Oxygen = %s" % (
                    lakes_sample_for_temperature, lakes_sample_for_oxygen))
            return 0
    else:
        print("Calibration cannot be done since at least one of the two lakes_sample variables are not a list or False.")
        return 0

    if not ['lake_id', 'name'] in lakes:
        print(
            "Calibration cannot be done since the file containing the list of lakes is missing at least one of the key variables: 'lake_id' or 'name'. \n"
            "Verify the column names given at the first row of the file \n"
            "Column names given: %s" % list(lakes.columns))
        return 0
    else:
        lakes_data = lakes.set_index("lake_id").to_dict()
        lakes_list = list(lakes_data.get("name").keys())
        for lake_number in range(0, len(lakes_list)):
            if lakes_sample_for_temperature is False:

                run_calibration(lake_number,lakesList12,lakes_list)
            else:
                if lake_number in lakes_sample_for_temperature:
                    run_calibration(lake_number,lakesList12,lakes_list)

            if lakes_sample_for_oxygen is False:
                run_calibrationsoxy(lake_number,lakesList12,lakes_list)
            else:
                if lake_number in lakes_sample_for_oxygen:
                    run_calibrationsoxy(lake_number,lakesList12,lakes_list)
        return 1


def run_calibration(lake_number, lakesList12="lakes/2017SwedenList_only_validation_12lakes.csv", lakes_list=lakes_list, old=False):
    """
    Simple function to call the calibration of a specific lake. It create the object Lake, prepare the necessary input
    files and call for the calibration done on matlab.
    :param lake_number: Type int. number of row where the lake is in the list of lakes.
    :param lakesList12: Type str. file name and directory (if not in same directory as this script) for the list of all
    the lakes with enough data to be calibrated.
    :param old:
    :return: 1 if correctly execute, 0 if error enconter.
    """

    # Create lake DataFrame
    try:
        lakes = pd.read_csv(lakesList12, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')

    except FileNotFoundError:
        print("Calibration cannot be done since the file containing the list of lakes do not exist. \n"
              "Verify file name and directory \n"
              "String given: %s" % lakesList12)
        return 0

    # lakes_data = lakes.set_index("lake_id").to_dict()
    # lakes_list = list(lakes_data.get("name").keys())
    lake_name = lakes_list[lake_number]
    print(lake_name)

    lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                           list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                           list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                           list(lakes["Mean"])[lake_number],
                           list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                           list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number], scenarioid=2,
                           calibration=True, old=old)

    test = False
    print("%s/Calibration_Completescore.csv" % lake.calibration_path)
    if not os.path.exists("%s/Calibration_Complete.csv" % lake.calibration_path):
        print("Calibration for {} is already complete.\n".format(lake_name))
        lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                               list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                               list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                               list(lakes["Mean"])[lake_number],
                               list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                               list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number], scenarioid=2,
                               calibration=True, old=old)
        lake.lake_input(2, 2, True)
        return None
    else:
        lake.lake_input(2, 2, True)
        lake.variables_by_depth(lake.start_year, lake.end_year)
        lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                               list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                               list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                               list(lakes["Mean"])[lake_number],
                               list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                               list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number], scenarioid=2,
                               calibration=True, old=old)
        lake.lake_input(2, 2, True)
        # test = lake.runlake(2, 2)

    if test == False:
        if not os.path.exists(lake.input_file):
            print("Calibration can't {} be completed. input file doesn't exist.\n".format(lake_name))
            return None
        elif not os.path.exists(lake.input_file) and \
                not os.path.exists(lake.init_file) and \
                not os.path.exists(lake.par_file):
            print("not all initial files existing for %s" % lake_name)
        elif os.path.exists(os.path.join(lake.calibration_path, "Calibration_problem.txt")):
            print("Unable to calibration {}.\n".format(lake_name))
            return None

        else:
            icedays = 110
            lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                   list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                   list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                   list(lakes["Mean"])[lake_number],
                                   list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                   list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                   scenarioid=2, calibration=True, old=old)
            cmd = "matlab -wait -r -nosplash -nodesktop MyLake_optimizer_final(%d,%d,%d,'%s','%s','%s','%s',%f,%f,%f,'%s');quit" % \
                  (lake.start_year, lake.end_year, lake.spin_up, lake.par_file,lake.input_file,lake.init_file,
                   lake.calibration_path, lake.latitude, lake.longitude, icedays, "temperature")

            print(cmd)

            os.system(cmd)
            lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                   list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                   list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                   list(lakes["Mean"])[lake_number],
                                   list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                   list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                   scenarioid=2, calibration=True, old=old)
            lake.lake_input(2, 2, True)

        if os.path.exists("%s/Calibration_Complete.csv" % lake.calibration_path):
            print("Calibration for {} is now complete.\n".format(lake_name))

            lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                   list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                   list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                   list(lakes["Mean"])[lake_number],
                                   list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                   list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                   scenarioid=2, calibration=True, old=old)
            lake.lake_input(2, 2, True)

        else:
            lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                   list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                   list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                   list(lakes["Mean"])[lake_number],
                                   list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                   list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                   scenarioid=2, calibration=True, old=old)
            cmd = "matlab -wait -r -nosplash -nodesktop MyLake_optimizer(%d,%d,%d,'%s','%s','%s','%s',%f,%f,%f);" \
                  "quit" % (
                      lake.start_year, lake.end_year, lake.spin_up, lake.par_file,lake.input_file,lake.init_file,
                      lake.calibration_path, lake.latitude, lake.longitude, icedays)

            print(cmd)

            os.system(cmd)
            lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                   list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                   list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                   list(lakes["Mean"])[lake_number],
                                   list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                   list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                   scenarioid=2, calibration=True, old=old)
            lake.lake_input(2, 2, True)

    return 1



def run_calibrationsoxy(lake_number, lakesList12="lakes/2017SwedenList_only_validation_12lakes.csv"):
    """
        Intermediary function to call Nelder-mead optimization function for a single lake.

        :param lake_number: Type int. The subid of the lake to calibrate.
        :return: None
    """

    global df
    lakes = pd.read_csv(lakesList12, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    lake_name = lakes_list[lake_number]
    print(lake_name)

    lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                           list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                           list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                           list(lakes["Mean"])[lake_number],
                           list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                           list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number], scenarioid=2,
                           calibration=True)

    test = False
    print("%s/Calibration_CompleteOXY.csv" % lake.calibration_path)
    if not os.path.exists("%s/Calibration_CompleteOXY.csv" % lake.calibration_path):
        print("Calibration for {} is already complete.\n".format(lake_name))
        lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                               list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                               list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                               list(lakes["Mean"])[lake_number],
                               list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                               list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number], scenarioid=2,
                               calibration=True)
        lake.lake_input(2, 2, True)
        return None
    else:
        lake.lake_input(2, 2, True)
        lake.variables_by_depth(lake.start_year, lake.end_year)
        lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                               list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                               list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                               list(lakes["Mean"])[lake_number],
                               list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                               list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number], scenarioid=2,
                               calibration=True)
        lake.lake_input(2, 2, True)
        # test = lake.runlake(2, 2)

    if test == False:
        if os.path.exists("{}/Observed_Oxygen.csv".format(lake.calibration_path)):
            df = pd.read_csv("{}/Observed_Oxygen.csv".format(lake.calibration_path))
        if not os.path.exists(lake.input_file):
            print("Calibration can't {} be completed. input file doesn't exist.\n".format(lake_name))
            return None
        elif not os.path.exists(lake.input_file) and \
                not os.path.exists(lake.init_file) and \
                not os.path.exists(lake.par_file):
            print("not all initial files existing for %s" % lake_name)
        elif os.path.exists(os.path.join(lake.calibration_path, "Calibration_problem.txt")):
            print("Unable to calibration {}.\n".format(lake_name))
            return None
        elif df.empty:
            os.remove("{}/Observed_Oxygen.csv".format(lake.calibration_path))
            print("Calibration can't {} be completed. observation file doesn't exist.\n".format(lake_name))
            return None
        elif not os.path.exists("{}/Observed_Oxygen.csv".format(lake.calibration_path)):
            print("Calibration can't {} be completed. observation file doesn't exist.\n".format(lake_name))
            return None

        else:
            icedays = 120
            lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                   list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                   list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                   list(lakes["Mean"])[lake_number],
                                   list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                   list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                   scenarioid=2, calibration=True)

            # eng = matlab.engine.start_matlab()
            # eng.MyLake_optimizer_final(lake.start_year, lake.end_year, lake.spin_up, lake.par_file,lake.input_file,lake.init_file,
            #        lake.calibration_path, lake.latitude, lake.longitude, icedays, "oxygen", nargout=0)
            ## stop matlab.engine
            # eng.quit()
            cmd = r'%s -wait -r -nosplash -nodesktop MyLake_optimizer_final(%d,%d,%d,%s,%s,%s,%s,%f,%f,%f,%s);quit' % (
                '"%s"' % matlab_directory,
                lake.start_year, lake.end_year, lake.spin_up, "%s" % lake.par_file,
                "%s" % lake.input_file,
                "%s" % lake.par_file, "%s" % lake.calibration_path, lake.latitude,
                lake.longitude, icedays, "%s" % "oxygen")

            print(cmd)

            # os.system(cmd)
            lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                   list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                   list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                   list(lakes["Mean"])[lake_number],
                                   list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                   list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                   scenarioid=2, calibration=True)
            lake.lake_input(2, 2, True)

        if os.path.exists("%s/Calibration_CompleteOXY.csv" % lake.calibration_path):
            print("Calibration for {} is now complete.\n".format(lake_name))
            # cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\');quit' % (
            #     os.path.join(lake.calibration_path, "2020init"), os.path.join(lake.calibration_path, "2020par"),
            #     os.path.join(lake.calibration_path, "2020input"), lake.start_year, lake.end_year, lake.calibration_path)
            #
            #
            # print(cmd)
            #
            # os.system(cmd)
            lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                   list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                   list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                   list(lakes["Mean"])[lake_number],
                                   list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                   list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                   scenarioid=2, calibration=True)
            lake.lake_input(2, 2, True)

        # else:
        #     lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
        #                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
        #                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
        #                            list(lakes["Mean"])[lake_number],
        #                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
        #                            list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
        #                            scenarioid=2, calibration=True)
        #     cmd = "matlab -wait -r -nosplash -nodesktop MyLake_optimizer(%d,%d,%d,'%s','%s','%s','%s',%f,%f,%f);quit" % \
        #           (lake.start_year, lake.end_year, lake.spin_up, os.path.join(lake.calibration_path, "2020par"),
        #            os.path.join(lake.calibration_path, "2020input"), os.path.join(lake.calibration_path, "2020init"),
        #            lake.calibration_path, lake.latitude, lake.longitude, icedays)
        #
        #     print(cmd)
        #
        #     os.system(cmd)
        #     lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
        #                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
        #                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
        #                            list(lakes["Mean"])[lake_number],
        #                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
        #                            list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
        #                            scenarioid=2, calibration=True)
        #     lake.lake_input(2, 2, True)

    #     if os.path.exists("%s/Calibration_Complete.csv" % lake.calibration_path):
    #         print("Calibration for {} is now complete.\n".format(lake_name))
    #         cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\');quit' % (
    #             os.path.join(lake.calibration_path, "2020init"), os.path.join(lake.calibration_path, "2020par"),
    #             os.path.join(lake.calibration_path, "2020input"), lake.start_year, lake.end_year, lake.calibration_path)
    #
    #
    #         print(cmd)
    #
    #         os.system(cmd)
    #         lake = CalibrationInfo(lake_name, list(lakes["lake_id"])[lake_number], list(lakes["subid"])[lake_number],
    #                                list(lakes["ebhex"])[lake_number], list(lakes["area"])[lake_number],
    #                                list(lakes["depth"])[lake_number], list(lakes["longitude"])[lake_number],
    #                                list(lakes["latitude"])[lake_number], list(lakes["volume"])[lake_number],list(lakes["Turnover"])[lake_number], 2)
    #         lake.lake_input(2, 2)
    #
    # else:
    #     cmd = "matlab -wait -r -nosplash -nodesktop MyLake_optimizer_oxygen(%d,%d,%d,'%s','%s','%s','%s',%f,%f);quit" % \
    #           (lake.start_year, lake.end_year,lake.spin_up, os.path.join(lake.calibration_path, "2020par"),
    #            os.path.join(lake.calibration_path, "2020input"), os.path.join(lake.calibration_path, "2020init"),
    #            lake.calibration_path, lake.latitude, lake.longitude)
    #
    #     print(cmd)


def calibration_iteration5():
    """
    Simple function to call the calibration of all lakes.
    :return: None
    """
    print(num_cores)
    for lake_number in range(0, len(lakes_list) - 3):
        run_calibrations5(lake_number)


def run_calibrations5(lake_number, lakesList12="lakes/2017SwedenList_only_validation_12lakes.csv"):
    """
        Intermediary function to call Nelder-mead optimization function for a single lake.

        :param lake_number: Type int. The subid of the lake to calibrate.
        :return: None
    """

    lakes = pd.read_csv(lakesList12, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    lake_name = lakes_list[lake_number]
    print(lake_name)

    lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                           list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                           list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                           list(lakes["Mean"])[lake_number],
                           list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                           list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number], 2)

    lake.lake_input(2, 2)
    lake.variables_by_depth(lake.start_year, lake.end_year)
    lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                           list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                           list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                           list(lakes["Mean"])[lake_number],
                           list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                           list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number], 2)
    # test = True
    print("%s/Calibration_Completescore.csv" % lake.calibration_path)
    if os.path.exists("%s/Calibration_Complete.csv" % lake.calibration_path):
        print("Calibration for {} is already complete.\n".format(lake_name))
        lake = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                               list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                               list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                               list(lakes["Mean"])[lake_number],
                               list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                               list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number], 2)

        lake.lake_input(2, 2)
        # lake.variables_by_depth()
        test = lake.runlake(2, 2)
        return None
    else:
        lake.lake_input(2, 2)
        lake.variables_by_depth(lake.start_year, lake.end_year)
        test = lake.runlake(2, 2)

    # if test == False:
    #     if not os.path.exists("{}/2020input".format(lake.calibration_path)):
    #         print("Calibration can't {} be completed. input file doesn't exist.\n".format(lake_name))
    #         return None
    #     elif not os.path.exists(os.path.join(lake.calibration_path, "2020input")) and \
    #             not os.path.exists(os.path.join(lake.calibration_path, "2020init")) and \
    #             not os.path.exists(os.path.join(lake.calibration_path, "2020par")):
    #         print("not all initial files existing for %s" % lake_name)
    #     elif os.path.exists(os.path.join(lake.calibration_path, "Calibration_problem.txt")):
    #         print("Unable to calibration {}.\n".format(lake_name))
    #         return None
    #
    #     else:
    #         cmd = "matlab -wait -r -nosplash -nodesktop MyLake_optimizer(%d,%d,%d,'%s','%s','%s','%s',%f,%f);quit" % \
    #               (lake.start_year, lake.end_year,lake.spin_up, os.path.join(lake.calibration_path, "2020par"),
    #                os.path.join(lake.calibration_path, "2020input"), os.path.join(lake.calibration_path, "2020init"),
    #                lake.calibration_path, lake.latitude, lake.longitude)
    #
    #         print(cmd)
    #
    #         os.system(cmd)
    #         lake = CalibrationInfo(lake_name, list(lakes["lake_id"])[lake_number], list(lakes["subid"])[lake_number],
    #                                list(lakes["ebhex"])[lake_number], list(lakes["area"])[lake_number],
    #                                list(lakes["depth"])[lake_number], list(lakes["longitude"])[lake_number],
    #                                list(lakes["latitude"])[lake_number], list(lakes["volume"])[lake_number],list(lakes["Turnover"])[lake_number], 2)
    #         lake.lake_input(2, 2)
    #
    #     if os.path.exists("%s/Calibration_Completescore.csv" % lake.calibration_path):
    #         print("Calibration for {} is now complete.\n".format(lake_name))
    #         # cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\');quit' % (
    #         #     os.path.join(lake.calibration_path, "2020init"), os.path.join(lake.calibration_path, "2020par"),
    #         #     os.path.join(lake.calibration_path, "2020input"), lake.start_year, lake.end_year, lake.calibration_path)
    #         #
    #         #
    #         # print(cmd)
    #         #
    #         # os.system(cmd)
    #         lake = CalibrationInfo(lake_name, list(lakes["lake_id"])[lake_number], list(lakes["subid"])[lake_number],
    #                                list(lakes["ebhex"])[lake_number], list(lakes["area"])[lake_number],
    #                                list(lakes["depth"])[lake_number], list(lakes["longitude"])[lake_number],
    #                                list(lakes["latitude"])[lake_number], list(lakes["volume"])[lake_number],list(lakes["Turnover"])[lake_number], 2)
    #         lake.lake_input(2, 2)
    #
    # else:
    #     cmd = "matlab -wait -r -nosplash -nodesktop MyLake_optimizer(%d,%d,%d,'%s','%s','%s','%s',%f,%f);quit" % \
    #           (lake.start_year, lake.end_year,lake.spin_up, os.path.join(lake.calibration_path, "2020par"),
    #            os.path.join(lake.calibration_path, "2020input"), os.path.join(lake.calibration_path, "2020init"),
    #            lake.calibration_path, lake.latitude, lake.longitude)
    #
    #     print(cmd)



def summary_characteristics_lake_parallel(modelid=2, scenarioid=2, lakes_listcsv="2017SwedenList.csv",
                                          calibration=False, old=False, outputfolder="Postproc", new=False):
    """
    Simple function to call a parallel calibration of all lakes.
    :return: None
    """
    # print(num_cores)
    # lakes_list = list(lakes_data.get("name").values())
    lakes = pd.read_csv(lakes_listcsv,
                        encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    subid = list(lakes_data.get("subid").keys())
    print(len(lakes_list))

    i = 0
    # for lake_number in range(1,8):# len(lakes_list)):
    #     summary_characteristics_lake1(lake_number,lakes_listcsv,modelid,scenarioid,calibration,old)

    Parallel(n_jobs=num_cores, verbose=10)(
        delayed(summary_characteristics_lake1)(lake_number, lakes_listcsv, modelid, scenarioid) for lake_number in range(0, len(lakes_list)))

    # for lake_number in range(0, len(lakes_list)):
    #     summary_characteristics_lake1(lake_number, lakes_listcsv, modelid, scenarioid, calibration, old, outputfolder, new)


def summary_characteristics_lake1(lake_number, lakes_listcsv="2017SwedenList.csv", modelid=2, scenarioid=2,
                                  calibration=False, old=False, outputfolder="Postproc", new=False):
    lakes = pd.read_csv(lakes_listcsv, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    subid = list(lakes_data.get("subid").keys())
    try:
        if str(list(lakes["Mean"])[lake_number]) != "nan":
            if 1 == 1:  # try:
                lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                    list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                    list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                    list(lakes["Mean"])[lake_number],
                                    list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                    list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                    scenarioid=scenarioid, modelid=modelid, calibration=True, old=old,
                                    outputfolder=outputfolder, new=new)
                # lakeinfo.lake_input(modelid,scenarioid)

                # test = lakeinfo.runlakefinal(modelid,scenarioid)
                if not os.path.exists(os.path.join(lakeinfo.outdir, '20211210REDOCOMPLETE')):

                    if lakeinfo.lake_id in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 698, 16765, 67035]:
                        if calibration:
                            if 1 == 1:  # if list(lakes["lake_id"])[lake_number] in [32276,310,14939,30704,31895,6950,99045,33590,33494,698,16765,67035]:
                                swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                                    final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                              list(lakes["latitude"])[lake_number],
                                                              list(lakes["depth"])[lake_number],
                                                              list(lakes["Mean"])[lake_number],
                                                              list(lakes["C.L"])[lake_number],
                                                              list(lakes["SC.L"])[lake_number],
                                                              list(lakes["Turnover"])[lake_number],
                                                              list(lakes["area"])[lake_number],
                                                              list(lakes["volume"])[lake_number],
                                                              list(lakes["sedimentArea"])[lake_number])

                                lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                    list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                                    list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                    list(lakes["Mean"])[lake_number],
                                                    list(lakes["longitude"])[lake_number],
                                                    list(lakes["latitude"])[lake_number],
                                                    list(lakes["volume"])[lake_number],
                                                    list(lakes["Turnover"])[lake_number],
                                                    scenarioid=scenarioid, modelid=modelid, calibration=calibration,
                                                    old=old, outputfolder=outputfolder, new=new)
                                lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                        list(lakes["subid"])[lake_number],
                                                        list(lakes["ebhex"])[lake_number],
                                                        list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                        list(lakes["Mean"])[lake_number],
                                                        list(lakes["longitude"])[lake_number],
                                                        list(lakes["latitude"])[lake_number],
                                                        list(lakes["volume"])[lake_number],
                                                        list(lakes["Turnover"])[lake_number], 2, calibration=calibration,
                                                        old=old, outputfolder=outputfolder, new=new)

                                lakec.variables_by_depth(2001, 2010)
                            lakeinfo.lake_input(modelid, scenarioid, calibration=calibration)
                        else:
                            lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                    list(lakes["subid"])[lake_number],
                                                    list(lakes["ebhex"])[lake_number],
                                                    list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                    list(lakes["Mean"])[lake_number],
                                                    list(lakes["longitude"])[lake_number],
                                                    list(lakes["latitude"])[lake_number],
                                                    list(lakes["volume"])[lake_number],
                                                    list(lakes["Turnover"])[lake_number], 2, calibration=calibration,
                                                    old=old, outputfolder=outputfolder, new=new)
                            lakec.variables_by_depth(2001, 2010)
                            swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                                final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                          list(lakes["latitude"])[lake_number],
                                                          list(lakes["depth"])[lake_number],
                                                          list(lakes["Mean"])[lake_number],
                                                          list(lakes["C.L"])[lake_number], list(lakes["SC.L"])[lake_number],
                                                          list(lakes["Turnover"])[lake_number],
                                                          list(lakes["area"])[lake_number],
                                                          list(lakes["volume"])[lake_number],
                                                          list(lakes["sedimentArea"])[lake_number])

                            lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                                list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                list(lakes["Mean"])[lake_number],
                                                list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                                list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                                swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0,
                                                albice,
                                                albsnow, scenarioid=scenarioid, modelid=modelid, calibration=calibration,
                                                old=old, outputfolder=outputfolder, new=new)
                            lakeinfo.lake_input(modelid, scenarioid)

                        lakeinfo.runlakefinal(modelid, scenarioid, calibration=calibration, old=old)
                    else:
                        if not calibration:
                            swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                                final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                          list(lakes["latitude"])[lake_number],
                                                          list(lakes["depth"])[lake_number],
                                                          list(lakes["Mean"])[lake_number],
                                                          list(lakes["C.L"])[lake_number], list(lakes["SC.L"])[lake_number],
                                                          list(lakes["Turnover"])[lake_number],
                                                          list(lakes["area"])[lake_number],
                                                          list(lakes["volume"])[lake_number],
                                                          list(lakes["sedimentArea"])[lake_number])

                            lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                                list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                list(lakes["Mean"])[lake_number],
                                                list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                                list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                                swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0,
                                                albice,
                                                albsnow, scenarioid=scenarioid, modelid=modelid, calibration=calibration,
                                                old=old, outputfolder=outputfolder, new=new)
                            lakeinfo.lake_input(modelid, scenarioid)
                            outdir = lakeinfo.outdir
                            lakeinfo.runlakefinal(modelid, scenarioid, calibration)
                            if not os.path.exists(os.path.join(outdir, '20211210REDOCOMPLETE')):

                                if os.path.exists(os.path.join(outdir, '20211210REDOCOMPLETE')):
                                    lakeinfo.outputfile(calibration=calibration)

                                if os.path.exists(os.path.join(outdir, "strat.csv")) and os.path.exists(
                                        os.path.join(outdir, "watertemp.csv")) \
                                        and os.path.exists(os.path.join(outdir, "thermodepth.csv")):
                                    thermodepth = pd.read_csv(os.path.join(outdir, "thermodepth.csv"), header=None)
                                    thermodepth = thermodepth[thermodepth[1] != 0]

                                    thermocline = thermodepth.mean()[1]

                                    # print(lakeinfo.lake_id)

                                    allmodelfinal = lakeinfo.comparison_obs_sims_newv2(calibration=calibration,
                                                                                       thermocline=thermocline)

            # except:
            #     print("error iteration")
        else:
            print("lake is missing necessary variable(s) to run the model")
    except:
        print("problem with run the model")


def summary_characteristics_lake(modelid=2, scenarioid=2, lakes_listcsv="2017SwedenList.csv", calibration=False,
                                 withfig=False, old=False, outputfolder="Postproc", new=False):
    # lakes_list = list(lakes_data.get("name").values())
    lakes = pd.read_csv(lakes_listcsv, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    subid = list(lakes_data.get("subid").keys())

    summary = pd.DataFrame(index=range(len(lakes_list)),
                           columns=["lake_id", "subid", "lake_name", "ebhex",
                                    "kz_n0", "c_shelter", "alb_ice", "alb_snow", "i_scv", "i_sct", "swa_b0", "swa_b1",
                                    "i_scdoc", "k_bod", "k_sod", "i_scO",
                                    "area", "depth", "longitude", "latitude", "volume"])
    i = 0

    for lake_number in range(0, len(lakes_list)):
        if 1 == 1:  # lake_number != 6:#try:
            if list(lakes["lake_id"])[lake_number] in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 698,
                                                       16765, 67035]:
                if calibration:
                    if 1 == 1:  # if list(lakes["lake_id"])[lake_number] in [32276,310,14939,30704,31895,6950,99045,33590,33494,698,16765,67035]:
                        swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                            final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                      list(lakes["latitude"])[lake_number],
                                                      list(lakes["depth"])[lake_number],
                                                      list(lakes["Mean"])[lake_number],
                                                      list(lakes["C.L"])[lake_number], list(lakes["SC.L"])[lake_number],
                                                      list(lakes["Turnover"])[lake_number],
                                                      list(lakes["area"])[lake_number],
                                                      list(lakes["volume"])[lake_number],
                                                      list(lakes["sedimentArea"])[lake_number])

                        lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                            list(lakes["Mean"])[lake_number],
                                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                            list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                            scenarioid=scenarioid, modelid=modelid, calibration=calibration, old=old,
                                            outputfolder=outputfolder, new=new)
                        outdir = lakeinfo.calibration_path
                        # lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                        #                         list(lakes["subid"])[lake_number],
                        #                         list(lakes["ebhex"])[lake_number],
                        #                         list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                        #                         list(lakes["Mean"])[lake_number],
                        #                         list(lakes["longitude"])[lake_number],
                        #                         list(lakes["latitude"])[lake_number],
                        #                         list(lakes["volume"])[lake_number],
                        #                         list(lakes["Turnover"])[lake_number], 2,calibration=calibration,old=old)
                        # lakec.variables_by_depth(2001, 2010)
                else:
                    swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                        final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                  list(lakes["latitude"])[lake_number],
                                                  list(lakes["depth"])[lake_number], list(lakes["Mean"])[lake_number],
                                                  list(lakes["C.L"])[lake_number], list(lakes["SC.L"])[lake_number],
                                                  list(lakes["Turnover"])[lake_number],
                                                  list(lakes["area"])[lake_number],
                                                  list(lakes["volume"])[lake_number],
                                                  list(lakes["sedimentArea"])[lake_number])

                    lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                        list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                        list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                        list(lakes["Mean"])[lake_number],
                                        list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                        list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                        swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0,
                                        albice,
                                        albsnow, scenarioid=scenarioid, modelid=modelid, old=old,
                                        outputfolder=outputfolder, new=new)
                    # lakeinfo.lake_input(modelid, scenarioid, calibration=calibration)
                    outdir = lakeinfo.outdir
                # print(outdir)
                if os.path.exists(os.path.join(outdir, '20211210REDOCOMPLETE')):
                    summary.loc[i, "run_completed"] = 1
                    lakeinfo.outputfile(calibration=calibration)
                else:
                    summary.loc[i, "run_completed"] = 0
                # print(os.path.join(outdir, "strat.csv"))
                # print(os.path.exists(os.path.join(outdir, "strat.csv")),os.path.exists(os.path.join(outdir, "watertemp.csv")), os.path.exists(os.path.join(outdir, "thermodepth.csv")))
                if os.path.exists(os.path.join(outdir, "strat.csv")) and os.path.exists(
                        os.path.join(outdir, "watertemp.csv")) \
                        and os.path.exists(os.path.join(outdir, "thermodepth.csv")):
                    if list(lakes["lake_id"])[lake_number] in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590,
                                                               33494,
                                                               698, 16765, 67035]:
                        thermodepth = pd.read_csv(os.path.join(outdir, "thermodepth.csv"), header=None)
                        thermodepth = thermodepth[thermodepth[1] != 0]

                        thermocline = thermodepth.mean()[1]
                        if calibration:  # try:
                            # print("old")
                            # lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                            #                         list(lakes["subid"])[lake_number],
                            #                         list(lakes["ebhex"])[lake_number],
                            #                         list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                            #                         list(lakes["Mean"])[lake_number],
                            #                         list(lakes["longitude"])[lake_number],
                            #                         list(lakes["latitude"])[lake_number],
                            #                         list(lakes["volume"])[lake_number],
                            #                         list(lakes["Turnover"])[lake_number], 2)
                            # lakec.variables_by_depth(2001,2010)
                            # cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                            #     outdir, 2001, 2010)
                            # print(cmd)
                            # os.system(cmd)
                            # print(lakeinfo.lake_id)

                            allmodelfinal = lakeinfo.comparison_obs_sims_newv2(calibration=True,
                                                                               thermocline=thermocline)
                            # datasummary = allmodelfinal
                            # datasummary['lake_id'] = lake_number
                            # datasummary.to_csv("summary_data_lake_%s_%s_%s_old_%s_%s.csv" % (lake_number,modelid, scenarioid, old, timestr), index=False)

            else:
                if not calibration:
                    swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                        final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                  list(lakes["latitude"])[lake_number],
                                                  list(lakes["depth"])[lake_number], list(lakes["Mean"])[lake_number],
                                                  list(lakes["C.L"])[lake_number], list(lakes["SC.L"])[lake_number],
                                                  list(lakes["Turnover"])[lake_number],
                                                  list(lakes["area"])[lake_number],
                                                  list(lakes["volume"])[lake_number],
                                                  list(lakes["sedimentArea"])[lake_number])

                    lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                        list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                        list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                        list(lakes["Mean"])[lake_number],
                                        list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                        list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                        swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0,
                                        albice,
                                        albsnow, scenarioid=scenarioid, modelid=modelid, old=old,
                                        outputfolder=outputfolder, new=new)
                    # lakeinfo.lake_input(modelid, scenarioid, calibration=calibration)
                    outdir = lakeinfo.outdir
                    if os.path.exists(os.path.join(outdir, '20211210REDOCOMPLETE')):
                        summary.loc[i, "run_completed"] = 1
                        lakeinfo.outputfile(calibration=calibration)
                    else:
                        summary.loc[i, "run_completed"] = 0
                    if os.path.exists(os.path.join(outdir, "strat.csv")) and os.path.exists(
                            os.path.join(outdir, "watertemp.csv")) \
                            and os.path.exists(os.path.join(outdir, "thermodepth.csv")):
                        thermodepth = pd.read_csv(os.path.join(outdir, "thermodepth.csv"), header=None)
                        thermodepth = thermodepth[thermodepth[1] != 0]

                        thermocline = thermodepth.mean()[1]

                        # print(lakeinfo.lake_id)

                        # allmodelfinal = lakeinfo.comparison_obs_sims_newv2(calibration=calibration,
                        #                                                    thermocline=thermocline)
                        # except:
                        #     print("issues")
            if list(lakes["lake_id"])[lake_number] in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 698,
                                                       16765, 67035] or not calibration:
                # test = lakeinfo.runlakefinal(2, 2)

                # caracteristics
                summary.loc[i, 'lake_name'] = lakeinfo.lake_name
                summary.loc[i, "subid"] = lakeinfo.subid
                summary.loc[i, "lake_id"] = lakeinfo.lake_id
                summary.loc[i, "ebhex"] = lakeinfo.ebhex
                summary.loc[i, "longitude"] = lakeinfo.longitude
                summary.loc[i, "latitude"] = lakeinfo.latitude
                summary.loc[i, "depth"] = lakeinfo.depth
                summary.loc[i, "area"] = lakeinfo.area
                summary.loc[i, "volume"] = lakeinfo.volume

                # paramters
                summary.loc[i, "i_scdoc"] = lakeinfo.i_sc_doc
                summary.loc[i, "swa_b1"] = lakeinfo.swa_b1
                summary.loc[i, "swa_b0"] = lakeinfo.swa_b0
                summary.loc[i, "k_bod"] = lakeinfo.k_bod
                summary.loc[i, "k_sod"] = lakeinfo.k_sod
                summary.loc[i, "kz_n0"] = lakeinfo.kz_n0
                summary.loc[i, "c_shelter"] = lakeinfo.c_shelter
                summary.loc[i, "alb_snow"] = lakeinfo.alb_melt_snow
                summary.loc[i, "alb_ice"] = lakeinfo.alb_melt_ice
                summary.loc[i, "i_scv"] = lakeinfo.i_scv
                summary.loc[i, "i_sct"] = lakeinfo.i_sct
                summary.loc[i, "i_scO"] = lakeinfo.i_sco

                if list(lakes["lake_id"])[lake_number] in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590,
                                                           33494,
                                                           698, 16765, 67035]:
                    rmse, rmsen, r_sq, secchi_m = lakeinfo.performance_analysis(calibration=calibration, old=old)

                    # scores
                    summary.loc[i, "Score"] = sum(rmsen)
                    summary.loc[i, "nrmseT"] = rmsen[0]
                    summary.loc[i, "nrmseO"] = rmsen[1]
                    summary.loc[i, "nrmseS"] = rmsen[2]

                    # rmse for comparison
                    summary.loc[i, "rmseT"] = rmse[0]
                    summary.loc[i, "rmseO"] = rmse[1]
                    summary.loc[i, "rmseS"] = rmse[2]
                    summary.loc[i, "rT"] = r_sq[0]
                    summary.loc[i, "rO"] = r_sq[1]
                    summary.loc[i, "rS"] = r_sq[2]

                    summary.loc[i, "Mean_secchi_obs"] = secchi_m[0]
                    summary.loc[i, "st_secchi_obs"] = secchi_m[2]
                    summary.loc[i, "Mean_secchi_sim"] = secchi_m[1]
                    summary.loc[i, "st_secchi_sim"] = secchi_m[3]

                # except:
                #     print("error iteration")

                i += 1
        # except:
        #     print("errorwith file!")
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if calibration:
        summary.to_csv("summary_info_lake_lakes_cali_%s_%s_old_%s_%s.csv" % (modelid, scenarioid, old, timestr),
                       index=False)
    else:
        summary.to_csv("summary_info_lake_lakes_%s_%s_old_%s_%s.csv" % (modelid, scenarioid, old, timestr), index=False)

    if withfig:
        Graphics(outputfolder=outputfolder).density_plot(summary, calibration, old)


def summary_obsvation_data(lakes_listcsv="2017SwedenList.csv"):
    lakes = pd.read_csv(lakes_listcsv, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    subid = list(lakes_data.get("subid").keys())
    stats = []

    date, depth, obsT, obsO2, lakeid, lakeebhex, max, mean = [], [], [], [], [], [], [], []
    for lake_number in range(0, len(lakes_list)):
        if list(lakes["lake_id"])[lake_number] in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 698,
                                                   16765,
                                                   67035]:
            lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                list(lakes["Mean"])[lake_number],
                                list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number])

            try:
                for variable in ["Tzt", "O2zt"]:
                    data = pd.read_csv(os.path.join(lakeinfo.calibration_path, "%scompare.csv" % variable), header=None,
                                       names=["dates", "depths", "obs", "sim"])
                    date.extend(data["dates"].tolist())
                    depth.extend(data["depths"].tolist())
                    if variable == "Tzt":
                        obsT.extend(data["obs"].tolist())
                        obsO2.extend([np.nan] * len(data))

                    else:
                        obsT.extend([np.nan] * len(data))
                        obsO2.extend(data["obs"].tolist())

                    max1 = data.iloc[data['obs'].argmax(), :-1].values.tolist()
                    max1.extend(["max", variable, lakeinfo.lake_id])
                    stats.append(max1)
                    min1 = data.iloc[data['obs'].argmin(), :-1].values.tolist()
                    min1.extend(["min", variable, lakeinfo.lake_id])

                    stats.append(min1)
                    meanobs = list(data.mean())[0:-1]
                    meanobs.extend(["mean", variable, lakeinfo.lake_id])
                    mean1 = [""]
                    mean1.extend(meanobs)

                    stats.append(mean1)

                    lakeid.extend([lakeinfo.lake_id] * len(data))
                    lakeebhex.extend([lakeinfo.ebhex] * len(data))
                    max.extend([lakeinfo.depth] * len(data))
                    mean.extend([lakeinfo.mean] * len(data))

            except:
                print("%s not disponible for %s" % (variable, lakeinfo.ebhex))

    finallist = pd.DataFrame([date, depth, obsT, obsO2, lakeid, lakeebhex, max, mean])
    finalliststats = pd.DataFrame(stats, columns=["dates", "depths", "obs", "stat", "variabe", "lake_id"])

    T_final = finallist.transpose()
    T_final.columns = ["dates", "depths", "obsT", "obsO", "lakeid", "lakepath", "max", "mean"]
    # final = pd.DataFrame(data=finallist,columns=["dates","depths","obsT","obsO","lakeid","lakepath","max","mean"])
    # T_final.to_csv("summary_obs_lakes.csv", index=False)
    finalliststats.to_csv("summary_obs_lakes_stats.csv", index=False)
    T_final['dates'] = pd.to_datetime(T_final['dates'])
    T_final['lakeid'] = T_final['lakeid'].apply(str)
    finalT = T_final.dropna(subset=['obsT'])

    Graphics(output_path).observation_plot(finalT, T_final)


def summary_characteristics_lake_summary(models, scenarios, lakes_listcsv="2017SwedenList.csv", calibration=False,
                                         old=False):
    # lakes_list = list(lakes_data.get("name").values())
    global lakeinfo
    lakes = pd.read_csv(lakes_listcsv, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    subid = list(lakes_data.get("subid").keys())

    summary = pd.DataFrame(index=range(len(lakes_list)),
                           columns=["lake_id", "subid", "lake_name", "ebhex", "i_scdoc", "swa_b1", "swa_b0", "k_bod",
                                    "k_sod", "kz_n0",
                                    "c_shelter", "alb_snow", "alb_ice", "i_scv", "i_sct", "i_scO", "area", "depth",
                                    "longitude", "latitude", "volume"])
    i = 0
    for modelid in models:
        for scenarioid in scenarios:
            for lake_number in range(0, len(lakes_list)):
                if 1 == 1:  # try:
                    if calibration:
                        if list(lakes["lake_id"])[lake_number] in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590,
                                                                   33494, 698, 16765, 67035]:
                            swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                                final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                          list(lakes["latitude"])[lake_number],
                                                          list(lakes["depth"])[lake_number],
                                                          list(lakes["Mean"])[lake_number],
                                                          list(lakes["C.L"])[lake_number],
                                                          list(lakes["SC.L"])[lake_number],
                                                          list(lakes["Turnover"])[lake_number],
                                                          list(lakes["area"])[lake_number],
                                                          list(lakes["volume"])[lake_number],
                                                          list(lakes["sedimentArea"])[lake_number])

                            lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                                list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                list(lakes["Mean"])[lake_number],
                                                list(lakes["longitude"])[lake_number],
                                                list(lakes["latitude"])[lake_number],
                                                list(lakes["volume"])[lake_number],
                                                list(lakes["Turnover"])[lake_number], scenarioid=scenarioid,
                                                modelid=modelid, old=old)
                    else:
                        swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                            final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                      list(lakes["latitude"])[lake_number],
                                                      list(lakes["depth"])[lake_number],
                                                      list(lakes["Mean"])[lake_number],
                                                      list(lakes["C.L"])[lake_number], list(lakes["SC.L"])[lake_number],
                                                      list(lakes["Turnover"])[lake_number],
                                                      list(lakes["area"])[lake_number],
                                                      list(lakes["volume"])[lake_number],
                                                      list(lakes["sedimentArea"])[lake_number])

                        lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                            list(lakes["Mean"])[lake_number],
                                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                            list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                            swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0,
                                            albice,
                                            albsnow, scenarioid=scenarioid, modelid=modelid, old=old)
                        lakeinfo.lake_input(modelid, scenarioid)

                    test = lakeinfo.runlakefinal(modelid=modelid,scenarioid=scenarioid)

                    # caracteristics
                    summary.loc[i, 'model'] = modelid
                    summary.loc[i, 'scenario'] = scenarioid
                    summary.loc[i, 'lake_name'] = lakeinfo.lake_name
                    summary.loc[i, "subid"] = lakeinfo.subid
                    summary.loc[i, "lake_id"] = lakeinfo.lake_id
                    summary.loc[i, "ebhex"] = lakeinfo.ebhex
                    summary.loc[i, "longitude"] = lakeinfo.longitude
                    summary.loc[i, "latitude"] = lakeinfo.latitude
                    summary.loc[i, "depth"] = lakeinfo.depth
                    summary.loc[i, "area"] = lakeinfo.area
                    summary.loc[i, "volume"] = lakeinfo.volume

                    # paramters
                    summary.loc[i, "i_scdoc"] = lakeinfo.i_sc_doc
                    summary.loc[i, "swa_b1"] = lakeinfo.swa_b1
                    summary.loc[i, "swa_b0"] = lakeinfo.swa_b0
                    summary.loc[i, "k_bod"] = lakeinfo.k_bod
                    summary.loc[i, "k_sod"] = lakeinfo.k_sod
                    summary.loc[i, "kz_n0"] = lakeinfo.kz_n0
                    summary.loc[i, "c_shelter"] = lakeinfo.c_shelter
                    summary.loc[i, "alb_snow"] = lakeinfo.alb_melt_snow
                    summary.loc[i, "alb_ice"] = lakeinfo.alb_melt_ice
                    summary.loc[i, "i_scv"] = lakeinfo.i_scv
                    summary.loc[i, "i_sct"] = lakeinfo.i_sct
                    summary.loc[i, "i_scO"] = lakeinfo.i_sco

                    rmse, rmsen, r_sq, secchi_m = lakeinfo.performance_analysis(calibration=calibration, old=old)

                    # scores
                    summary.loc[i, "Score"] = sum(rmsen)
                    summary.loc[i, "nrmseT"] = rmsen[0]
                    summary.loc[i, "nrmseO"] = rmsen[1]
                    summary.loc[i, "nrmseS"] = rmsen[2]

                    # rmse for comparison
                    summary.loc[i, "rmseT"] = rmse[0]
                    summary.loc[i, "rmseO"] = rmse[1]
                    summary.loc[i, "rmseS"] = rmse[2]
                    summary.loc[i, "rT"] = r_sq[0]
                    summary.loc[i, "rO"] = r_sq[1]
                    summary.loc[i, "rS"] = r_sq[2]

                    summary.loc[i, "Mean_secchi_obs"] = secchi_m[0]
                    summary.loc[i, "st_secchi_obs"] = secchi_m[2]
                    summary.loc[i, "Mean_secchi_sim"] = secchi_m[1]
                    summary.loc[i, "st_secchi_sim"] = secchi_m[3]
                    # print(lakeinfo.outdir)
                    lakeinfo.outputfile(calibration=calibration, old=old)
                    if os.path.exists(os.path.join(lakeinfo.outdir, '20211210REDOCOMPLETE')):
                        summary.loc[i, "run_completed"] = 1
                        # lakeinfo.outputfile(calibration=True)
                    else:
                        summary.loc[i, "run_completed"] = 0
                    if calibration and not old:
                        if os.path.exists(os.path.join(lakeinfo.outdir, "strat.csv")) and os.path.exists(
                                os.path.join(lakeinfo.outdir, "watertemp.csv")) \
                                and os.path.exists(os.path.join(lakeinfo.outdir, "thermodepth.csv")):
                            thermodepth = pd.read_csv(os.path.join(lakeinfo.outdir, "thermodepth.csv"), header=None)
                            thermodepth = thermodepth[thermodepth[1] != 0]

                            thermocline = thermodepth.mean()[1]
                            lakeinfo.comparison_obs_sims_new(thermocline, calibration=calibration)
                # except:
                #     print("error iteration")

                i += 1

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    summary.to_csv("summary_info_lake_lakes_summary_%s1.csv" % (timestr), index=False)
    # Graphics(output_path).density_plot(summary, calibration, old)


def FishNiche_mean_secchi_graph(lakes_listcsv="2017SwedenList.csv", calibration=False, old=False):
    # lakes_list = list(lakes_data.get("name").values())
    lakes = pd.read_csv(lakes_listcsv, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    subid = list(lakes_data.get("subid").keys())
    y1A, y2B = 2001 - 5, 2010

    all_mean_secchi_model, all_mean_secchi_data = [], []
    all_std_secchi_model, all_std_secchi_data = [], []
    all_lake = []
    # lakes_list = list(lakes_data.get("name").values())

    x = 0
    for lake_number in range(0, len(lakes_list)):
        modelid, scenarioid = 2, 2
        if list(lakes["lake_id"])[lake_number] in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 698,
                                                   16765, 67035]:

            if calibration:

                swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                    final_equation_parameters(list(lakes["longitude"])[lake_number],
                                              list(lakes["latitude"])[lake_number],
                                              list(lakes["depth"])[lake_number], list(lakes["Mean"])[lake_number],
                                              list(lakes["C.L"])[lake_number], list(lakes["SC.L"])[lake_number],
                                              list(lakes["Turnover"])[lake_number], list(lakes["area"])[lake_number],
                                              list(lakes["volume"])[lake_number],
                                              list(lakes["sedimentArea"])[lake_number])

                lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                    list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                    list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                    list(lakes["Mean"])[lake_number],
                                    list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                    list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                    scenarioid=scenarioid, modelid=modelid, calibration=calibration, old=old, new=True)
                outdir = lakeinfo.calibration_path
            else:
                swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                    final_equation_parameters(list(lakes["longitude"])[lake_number],
                                              list(lakes["latitude"])[lake_number],
                                              list(lakes["depth"])[lake_number], list(lakes["Mean"])[lake_number],
                                              list(lakes["C.L"])[lake_number], list(lakes["SC.L"])[lake_number],
                                              list(lakes["Turnover"])[lake_number], list(lakes["area"])[lake_number],
                                              list(lakes["volume"])[lake_number],
                                              list(lakes["sedimentArea"])[lake_number])

                lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                    list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                    list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                    list(lakes["Mean"])[lake_number],
                                    list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                    list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                    swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice,
                                    albsnow, scenarioid=scenarioid, modelid=modelid, calibration=calibration, old=old,
                                    new=True)
                lakeinfo.lake_input(modelid, scenarioid)
                # lakeinfo.runlakefinal(modelid, scenarioid, calibration)
                outdir = lakeinfo.outdir
            if 1 == 1:  # os.path.exists(os.path.join(lakeinfo.calibration_path, "Calibration_Completescore.csv")):
                if os.path.exists("{}/Secchicompare.csv".format(outdir)):
                    try:
                        # if x == 6:
                        #     print("here")
                        secchi = pd.read_csv("{}/Secchicompare.csv".format(outdir), header=None)
                        all_mean_secchi_data.append(secchi.mean()[2])
                        all_mean_secchi_model.append(secchi.mean()[3])
                        all_std_secchi_data.append(secchi.std()[2])
                        all_std_secchi_model.append(secchi.std()[3])
                        all_lake.append(list(lakes["lake_id"])[lake_number])
                        x += 1
                    except:
                        print("Secchicompare.csv lake %s does not exist" % lakeinfo.lake_name)

    mean_model_data = all_mean_secchi_model
    pd.DataFrame([all_lake, all_mean_secchi_data, mean_model_data, all_std_secchi_model, all_std_secchi_data]).to_csv(
        "test12_%s.csv" % calibration)
    std_model_data = all_std_secchi_model
    slope, intercept, r_value, p_value, std_err = linregress(all_mean_secchi_data, mean_model_data)

    # results = graphique(all_mean_secchi_data, mean_model_data, all_std_secchi_data, std_model_data, r_value, slope,
    #                     intercept, calibration, old=old)
    resultds = Graphics(output_path, width=3, height=3, font_family="Arial", size=11.1).graphique_secchi_v2(
        all_mean_secchi_data, mean_model_data, all_std_secchi_data, std_model_data, calibration=calibration, old=old)
    # resultds = Graphics(output_path, width=3, height=3, font_family="Times New Roman", size=12).graphique_secchi_v2(
    #     all_mean_secchi_data, mean_model_data, all_std_secchi_data, std_model_data, calibration=calibration, old=old)


def comparison_plot(lakes_listcsv="2017SwedenList.csv", modelid=2, scenarioid=2, outputfolder="Postproc"):
    # lakes_list = list(lakes_data.get("name").values())
    lakes = pd.read_csv(lakes_listcsv, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    subid = list(lakes_data.get("subid").keys())
    y1A, y2B = 2001 - 5, 2010

    all_temp_model, all_temp_data, all_temp_depth, all_lake_temp = [], [], [], []
    all_oxygen_model, all_oxygen_data, all_oxygen_depth, all_lake_oxygen = [], [], [], []
    label_method = ["GA1", "GA2", "SR"]
    # lakes_list = list(lakes_data.get("name").values())
    position = -1
    lakeslist = []
    for lake_number in range(0, len(lakes_list)):
        outputfolder1 = outputfolder
        lake_temp_model, lake_temp_data, lake_temp_depth, lake_lake_temp = [], [], [], []
        lake_oxygen_model, lake_oxygen_data, lake_oxygen_depth, lake_lake_oxygen = [], [], [], []
        target = True
        for calibration in [True, False]:
            for old in [True, False]:
                if calibration or (not calibration and not old):
                    if list(lakes["lake_id"])[lake_number] in [6950, 32276, 310, 14939, 30704, 31895, 99045, 33590,
                                                               33494, 698, 16765, 67035]:
                        lakeslist.append(list(lakes["lake_id"])[lake_number])
                        position += 1

                        if calibration:
                            lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                                list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                list(lakes["Mean"])[lake_number],
                                                list(lakes["longitude"])[lake_number],
                                                list(lakes["latitude"])[lake_number],
                                                list(lakes["volume"])[lake_number],
                                                list(lakes["Turnover"])[lake_number],
                                                scenarioid=scenarioid, modelid=modelid, calibration=calibration,
                                                outputfolder=outputfolder, old=old)
                            if old:
                                outdir = lakeinfo.old_calibration_path
                                outputfolder1 = os.path.join(outputfolder, "result")
                            else:
                                outdir = lakeinfo.calibration_path
                                outputfolder1 = os.path.join(outputfolder, "result")

                            # lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                            #                         list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                            #                         list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                            #                         list(lakes["Mean"])[lake_number],
                            #                         list(lakes["longitude"])[lake_number],
                            #                         list(lakes["latitude"])[lake_number],
                            #                         list(lakes["volume"])[lake_number],
                            #                         list(lakes["Turnover"])[lake_number], scenarioid=scenarioid, outputfolder=outputfolder,old=old, calibration=calibration)
                            # lakec.variables_by_depth(lakec.start_year, lakec.end_year)
                        else:
                            swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                                final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                          list(lakes["latitude"])[lake_number],
                                                          list(lakes["depth"])[lake_number],
                                                          list(lakes["Mean"])[lake_number],
                                                          list(lakes["C.L"])[lake_number],
                                                          list(lakes["SC.L"])[lake_number],
                                                          list(lakes["Turnover"])[lake_number],
                                                          list(lakes["area"])[lake_number],
                                                          list(lakes["volume"])[lake_number],
                                                          list(lakes["sedimentArea"])[lake_number])

                            lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                                list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                list(lakes["Mean"])[lake_number],
                                                list(lakes["longitude"])[lake_number],
                                                list(lakes["latitude"])[lake_number],
                                                list(lakes["volume"])[lake_number],
                                                list(lakes["Turnover"])[lake_number],
                                                swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod,
                                                kzn0,
                                                albice,
                                                albsnow, scenarioid=scenarioid, modelid=modelid, calibration=False,
                                                outputfolder=outputfolder, old=old)

                            outdir = lakeinfo.outdir
                            # lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                            #                         list(lakes["subid"])[lake_number],
                            #                         list(lakes["ebhex"])[lake_number],
                            #                         list(lakes["area"])[lake_number],
                            #                         list(lakes["depth"])[lake_number],
                            #                         list(lakes["Mean"])[lake_number],
                            #                         list(lakes["longitude"])[lake_number],
                            #                         list(lakes["latitude"])[lake_number],
                            #                         list(lakes["volume"])[lake_number],
                            #                         list(lakes["Turnover"])[lake_number], scenarioid=scenarioid, outputfolder=outputfolder,
                            #                         calibration=calibration, old=old)
                            # lakec.variables_by_depth(lakec.start_year, lakec.end_year)

                            outputfolder1 = os.path.join(outputfolder, "result")
                        if lakeinfo.lake_id in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 698, 16765,
                                                67035]:
                            cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                                outdir, 2001, 2010)
                            print(cmd)
                            # os.system(cmd)

                        if 1 == 1:  # os.path.exists(os.path.join(lakeinfo.calibration_path,
                            if os.path.exists("{}/Tztcompare.csv".format(outdir)):
                                if 1 == 1:  # try:
                                    temp = pd.read_csv("{}/Tztcompare.csv".format(outdir), header=None)
                                    # temp['lake'] = lake_number
                                    temp['lake'] = position
                                    lake_temp_data.append(temp[2].tolist())
                                    lake_temp_model.append(temp[3].tolist())
                                    temp['depths'] = temp[1] / lakeinfo.depth
                                    lake_temp_depth.append(temp['depths'].tolist())
                                    lake_lake_temp.append(temp['lake'].tolist())
                                    # lake_temp_data = temp[2].tolist()
                                    # lake_temp_model = temp[3].tolist()
                                    # lake_temp_depth = temp['depths'].tolist()
                                    # lake_lake_temp = temp['lake'].tolist()

                            else:
                                lake_temp_data.append([])
                                lake_temp_model.append([])
                                lake_temp_depth.append([])
                                lake_lake_temp.append([position])

                            if os.path.exists("{}/O2ztcompare.csv".format(outdir)):
                                if 1 == 1:  # try:
                                    secchi = pd.read_csv("{}/O2ztcompare.csv".format(outdir), header=None)
                                    # secchi['lake'] = lake_number
                                    secchi['lake'] = position
                                    lake_oxygen_data.append(secchi[2].tolist())
                                    lake_oxygen_model.append(secchi[3].tolist())
                                    secchi['depths'] = secchi[1] / lakeinfo.depth
                                    lake_oxygen_depth.append(secchi['depths'].tolist())
                                    lake_lake_oxygen.append(secchi['lake'].tolist())
                                    # target = True
                                    # lake_oxygen_data = secchi[2].tolist()
                                    # lake_oxygen_model = secchi[3].tolist()
                                    # lake_oxygen_depth = secchi['depths'].tolist()
                                    # lake_lake_oxygen = secchi['lake'].tolist()


                            else:
                                lake_oxygen_data.append([])
                                lake_oxygen_model.append([])
                                lake_oxygen_depth.append([])
                                lake_lake_oxygen.append([position])
                                target = False

        if lakeinfo.lake_id in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 698, 16765, 67035]:

            Graphics(outputfolder).taylor_target_plot([lake_temp_model], [lake_temp_data], label_method, "Temperature",
                                                      "lake_%s" % lakes_list[lake_number],
                                                      [list(lakes["lake_id"])[lake_number]])
            if target:
                try:
                    Graphics(outputfolder).taylor_target_plot([lake_oxygen_model], [lake_oxygen_data], label_method,
                                                              "Oxygen", "lake_%s" % lakes_list[lake_number],
                                                              [list(lakes["lake_id"])[lake_number]])
                except:
                    print('error')
            print('tet')

        all_temp_data.append(lake_temp_data)
        all_temp_model.append(lake_temp_model)
        all_temp_depth.append(lake_temp_depth)
        all_lake_temp.append(lake_lake_temp)

        all_oxygen_data.append(lake_oxygen_data)
        all_oxygen_model.append(lake_oxygen_model)
        all_oxygen_depth.append(lake_oxygen_depth)
        all_lake_oxygen.append(lake_lake_oxygen)

        # except:
        # print("eror")

    xtemp = [item for sublist in all_temp_data for item in sublist]
    ytemp = [item for sublist in all_temp_model for item in sublist]
    ztemp = [item for sublist in all_temp_depth for item in sublist]
    symboltemp = [item for sublist in all_lake_temp for item in sublist]
    xoxy = [item for sublist in all_oxygen_data for item in sublist]
    yoxy = [item for sublist in all_oxygen_model for item in sublist]
    zoxy = [item for sublist in all_oxygen_depth for item in sublist]
    symboloxy = [item for sublist in all_lake_oxygen for item in sublist]
    print(len(all_temp_data[0]), len(all_temp_data))
    if 1 == 1:  # try:
        Graphics(outputfolder).taylor_target_plot(all_temp_model, all_temp_data, label_method, "Temperature",
                                                  "all_lakes", all_lake_temp)
        Graphics(outputfolder).taylor_target_plot(all_oxygen_model, all_oxygen_data, label_method, "Oxygen",
                                                  "all_lakes", all_lake_oxygen)
    # except:
    #     print('error')


def comparison_plot_v2(lakes_listcsv="2017SwedenList.csv", modelid=2, scenarioid=2, outputfolder="Postproc"):
    # lakes_list = list(lakes_data.get("name").values())
    lakes = pd.read_csv(lakes_listcsv, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    subid = list(lakes_data.get("subid").keys())
    y1A, y2B = 2001 - 5, 2010

    all_temp_model, all_temp_data, all_temp_depth, all_lake_temp = [], [], [], []
    all_oxygen_model, all_oxygen_data, all_oxygen_depth, all_lake_oxygen = [], [], [], []
    label_method = ["GA", "SR"]
    # lakes_list = list(lakes_data.get("name").values())
    position = -1
    lakeslist = []
    for lake_number in range(0, len(lakes_list)):
        outputfolder1 = outputfolder
        lake_temp_model, lake_temp_data, lake_temp_depth, lake_lake_temp = [], [], [], []
        lake_oxygen_model, lake_oxygen_data, lake_oxygen_depth, lake_lake_oxygen = [], [], [], []
        target = True
        for calibration in [True, False]:
            for old in [False]:
                if calibration or (not calibration and not old):
                    if list(lakes["lake_id"])[lake_number] in [6950, 32276, 310, 14939, 30704, 31895, 99045, 33590,
                                                               33494, 698, 16765, 67035]:
                        lakeslist.append(list(lakes["lake_id"])[lake_number])
                        position += 1

                        if calibration:
                            lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                                list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                list(lakes["Mean"])[lake_number],
                                                list(lakes["longitude"])[lake_number],
                                                list(lakes["latitude"])[lake_number],
                                                list(lakes["volume"])[lake_number],
                                                list(lakes["Turnover"])[lake_number],
                                                scenarioid=scenarioid, modelid=modelid, calibration=calibration,
                                                outputfolder=outputfolder, old=old, new=True)
                            if old:
                                outdir = lakeinfo.old_calibration_path
                                outputfolder1 = os.path.join(outputfolder, "result")
                            else:
                                outdir = lakeinfo.calibration_path
                                outputfolder1 = os.path.join(outputfolder, "result")

                            # lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                            #                         list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                            #                         list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                            #                         list(lakes["Mean"])[lake_number],
                            #                         list(lakes["longitude"])[lake_number],
                            #                         list(lakes["latitude"])[lake_number],
                            #                         list(lakes["volume"])[lake_number],
                            #                         list(lakes["Turnover"])[lake_number], scenarioid=scenarioid, outputfolder=outputfolder,old=old, calibration=calibration)
                            # lakec.variables_by_depth(lakec.start_year, lakec.end_year)
                        else:
                            swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                                final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                          list(lakes["latitude"])[lake_number],
                                                          list(lakes["depth"])[lake_number],
                                                          list(lakes["Mean"])[lake_number],
                                                          list(lakes["C.L"])[lake_number],
                                                          list(lakes["SC.L"])[lake_number],
                                                          list(lakes["Turnover"])[lake_number],
                                                          list(lakes["area"])[lake_number],
                                                          list(lakes["volume"])[lake_number],
                                                          list(lakes["sedimentArea"])[lake_number])

                            lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                                list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                list(lakes["Mean"])[lake_number],
                                                list(lakes["longitude"])[lake_number],
                                                list(lakes["latitude"])[lake_number],
                                                list(lakes["volume"])[lake_number],
                                                list(lakes["Turnover"])[lake_number],
                                                swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod,
                                                kzn0,
                                                albice,
                                                albsnow, scenarioid=scenarioid, modelid=modelid, calibration=False,
                                                outputfolder=outputfolder, old=old, new=True)

                            outdir = lakeinfo.outdir
                            # lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                            #                         list(lakes["subid"])[lake_number],
                            #                         list(lakes["ebhex"])[lake_number],
                            #                         list(lakes["area"])[lake_number],
                            #                         list(lakes["depth"])[lake_number],
                            #                         list(lakes["Mean"])[lake_number],
                            #                         list(lakes["longitude"])[lake_number],
                            #                         list(lakes["latitude"])[lake_number],
                            #                         list(lakes["volume"])[lake_number],
                            #                         list(lakes["Turnover"])[lake_number], scenarioid=scenarioid, outputfolder=outputfolder,
                            #                         calibration=calibration, old=old)
                            # lakec.variables_by_depth(lakec.start_year, lakec.end_year)

                            outputfolder1 = os.path.join(outputfolder, "result")
                        if lakeinfo.lake_id in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 698, 16765,
                                                67035]:
                            cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                                outdir, 2001, 2010)
                            print(cmd)
                            # os.system(cmd)

                        if 1 == 1:  # os.path.exists(os.path.join(lakeinfo.calibration_path,
                            if os.path.exists("{}/Tztcompare.csv".format(outdir)):
                                if 1 == 1:  # try:
                                    temp = pd.read_csv("{}/Tztcompare.csv".format(outdir), header=None)
                                    # temp['lake'] = lake_number
                                    temp['lake'] = position
                                    lake_temp_data.append(temp[2].tolist())
                                    lake_temp_model.append(temp[3].tolist())
                                    temp['depths'] = temp[1] / lakeinfo.depth
                                    lake_temp_depth.append(temp['depths'].tolist())
                                    lake_lake_temp.append(temp['lake'].tolist())
                                    # lake_temp_data = temp[2].tolist()
                                    # lake_temp_model = temp[3].tolist()
                                    # lake_temp_depth = temp['depths'].tolist()
                                    # lake_lake_temp = temp['lake'].tolist()

                            else:
                                lake_temp_data.append([])
                                lake_temp_model.append([])
                                lake_temp_depth.append([])
                                lake_lake_temp.append([position])

                            if os.path.exists("{}/O2ztcompare.csv".format(outdir)):
                                if 1 == 1:  # try:
                                    secchi = pd.read_csv("{}/O2ztcompare.csv".format(outdir), header=None)
                                    # secchi['lake'] = lake_number
                                    secchi['lake'] = position
                                    lake_oxygen_data.append(secchi[2].tolist())
                                    lake_oxygen_model.append(secchi[3].tolist())
                                    secchi['depths'] = secchi[1] / lakeinfo.depth
                                    lake_oxygen_depth.append(secchi['depths'].tolist())
                                    lake_lake_oxygen.append(secchi['lake'].tolist())
                                    # target = True
                                    # lake_oxygen_data = secchi[2].tolist()
                                    # lake_oxygen_model = secchi[3].tolist()
                                    # lake_oxygen_depth = secchi['depths'].tolist()
                                    # lake_lake_oxygen = secchi['lake'].tolist()


                            else:
                                lake_oxygen_data.append([])
                                lake_oxygen_model.append([])
                                lake_oxygen_depth.append([])
                                lake_lake_oxygen.append([position])
                                target = False

            # if lakeinfo.lake_id in [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 698, 16765, 67035]:

            # Graphics(outputfolder).taylor_target_plot_v2([lake_temp_model], [lake_temp_data], label_method, "Temperature",
            #                                           "lake_%s" % lakes_list[lake_number],
            #                                           [list(lakes["lake_id"])[lake_number]])
            # if target:
            #     try:
            #         Graphics(outputfolder).taylor_target_plot_v2([lake_oxygen_model], [lake_oxygen_data], label_method,
            #                                                   "Oxygen", "lake_%s" % lakes_list[lake_number],
            #                                                   [list(lakes["lake_id"])[lake_number]])
            #     except:
            #         print('error')
            print('tet')

        all_temp_data.append(lake_temp_data)
        all_temp_model.append(lake_temp_model)
        all_temp_depth.append(lake_temp_depth)
        all_lake_temp.append(lake_lake_temp)

        all_oxygen_data.append(lake_oxygen_data)
        all_oxygen_model.append(lake_oxygen_model)
        all_oxygen_depth.append(lake_oxygen_depth)
        all_lake_oxygen.append(lake_lake_oxygen)

        # except:
        # print("eror")

    xtemp = [item for sublist in all_temp_data for item in sublist]
    ytemp = [item for sublist in all_temp_model for item in sublist]
    ztemp = [item for sublist in all_temp_depth for item in sublist]
    symboltemp = [item for sublist in all_lake_temp for item in sublist]
    xoxy = [item for sublist in all_oxygen_data for item in sublist]
    yoxy = [item for sublist in all_oxygen_model for item in sublist]
    zoxy = [item for sublist in all_oxygen_depth for item in sublist]
    symboloxy = [item for sublist in all_lake_oxygen for item in sublist]
    # print(len(all_temp_data[0]), len(all_temp_data))
    if 1 == 1:  # try:
        # Graphics(outputfolder, width=3.25, height=3.25, font_family="Times New Roman", size=12).taylor_target_plot_v2(
        #     all_temp_model, all_temp_data, label_method, "Temperature",
        #     "all_lakes", all_lake_temp,speciallakes=[10])
        # Graphics(outputfolder, width=3.25, height=3.25, font_family="Times New Roman", size=12).taylor_target_plot_v2(
        #     all_oxygen_model, all_oxygen_data, label_method, "Oxygen",
        #     "all_lakes", all_lake_oxygen,speciallakes=[10])
        Graphics(outputfolder, width=3.25, height=3.25, font_family="Arial", size=11.5).taylor_target_plot_v2(
            all_temp_model, all_temp_data, label_method, "Temperature",
            "all_lakes", all_lake_temp,speciallakes=[10])
        Graphics(outputfolder, width=3.25, height=3.25, font_family="Arial", size=11.5).taylor_target_plot_v2(
            all_oxygen_model, all_oxygen_data, label_method, "Oxygen",
            "all_lakes", all_lake_oxygen,speciallakes=[10])
    # except:
    #     print('error')


def FishNiche_TO_graph(lakes_listcsv="2017SwedenList.csv", calibration=False, modelid=2, scenarioid=2,
                       outputfolder="Postproc", old=False):
    # lakes_list = list(lakes_data.get("name").values())
    lakes = pd.read_csv(lakes_listcsv, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    subid = list(lakes_data.get("subid").keys())
    y1A, y2B = 2001 - 5, 2010

    all_temp_model, all_temp_data, all_temp_depth, all_lake_temp = [], [], [], []
    all_oxygen_model, all_oxygen_data, all_oxygen_depth, all_lake_oxygen = [], [], [], []

    # lakes_list = list(lakes_data.get("name").values())
    position = -1
    for lake_number in range(0, len(lakes_list)):
        outputfolder1 = outputfolder
        if 1 == 1:  # try:
            if list(lakes["lake_id"])[lake_number] in [6950, 32276, 310, 14939, 30704, 31895, 99045, 33590, 33494, 698,
                                                       16765, 67035]:
                position += 1
                print(position, list(lakes["lake_id"])[lake_number])
                # if position > 11:
                #     print("here")
                if calibration:

                    lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                        list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                        list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                        list(lakes["Mean"])[lake_number],
                                        list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                        list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                        scenarioid=scenarioid, modelid=modelid, calibration=calibration,
                                        outputfolder=outputfolder, old=old)
                    outdir = lakeinfo.calibration_path
                    outputfolder1 = os.path.join(outputfolder, "result_old_calibration")
                    lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                            list(lakes["Mean"])[lake_number],
                                            list(lakes["longitude"])[lake_number],
                                            list(lakes["latitude"])[lake_number],
                                            list(lakes["volume"])[lake_number],
                                            list(lakes["Turnover"])[lake_number], 2, outputfolder=outputfolder,
                                            old=old, calibration=calibration)
                    lakec.variables_by_depth(lakec.start_year, lakec.end_year)
                    # lakeinfo.runlakefinal(2, 2, calibration=False)
                    print(outdir)
                    # cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                    #     outdir, 2001, 2010)
                    # print(cmd)
                    # os.system(cmd)
                    outputfolder1 = outputfolder

                else:
                    swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                        final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                  list(lakes["latitude"])[lake_number],
                                                  list(lakes["depth"])[lake_number], list(lakes["Mean"])[lake_number],
                                                  list(lakes["C.L"])[lake_number], list(lakes["SC.L"])[lake_number],
                                                  list(lakes["Turnover"])[lake_number],
                                                  list(lakes["area"])[lake_number],
                                                  list(lakes["volume"])[lake_number],
                                                  list(lakes["sedimentArea"])[lake_number])
                    if old:
                        lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                            list(lakes["Mean"])[lake_number],
                                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                            list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                            scenarioid=scenarioid, modelid=modelid, calibration=calibration,
                                            outputfolder="Postproc", old=old)
                        outdircali = lakeinfo.calibration_path

                        lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                            list(lakes["Mean"])[lake_number],
                                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                            list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                            scenarioid=scenarioid, modelid=modelid, calibration=True,
                                            outputfolder=outputfolder, old=old)
                        lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                                list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                list(lakes["Mean"])[lake_number],
                                                list(lakes["longitude"])[lake_number],
                                                list(lakes["latitude"])[lake_number],
                                                list(lakes["volume"])[lake_number],
                                                list(lakes["Turnover"])[lake_number], 2, outputfolder=outputfolder,
                                                old=old, calibration=calibration)
                        lakec.variables_by_depth(lakec.start_year, lakec.end_year)
                        outdir = lakeinfo.outdir
                        lakeinfo.runlakefinal(2, 2, calibration=False)
                        print(outdircali)
                        # cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                        #     outdir, 2001, 2010)
                        # print(cmd)
                        # os.system(cmd)
                        outputfolder1 = outputfolder
                    else:
                        lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                            list(lakes["Mean"])[lake_number],
                                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                            list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                            swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0,
                                            albice,
                                            albsnow, scenarioid=scenarioid, modelid=modelid, calibration=False,
                                            outputfolder=outputfolder, old=old)
                        # lakeinfo.lake_input(2,2)
                        outdir = lakeinfo.outdir
                        # cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                        #     outdir, 2001, 2010)
                        # print(cmd)
                        # os.system(cmd)
                if os.path.exists(os.path.join(lakeinfo.calibration_path,
                                               "Calibration_Complete.csv")):
                    if os.path.exists("{}/Tztcompare.csv".format(outdir)):
                        if 1 == 1:  # try:
                            secchi = pd.read_csv("{}/Tztcompare.csv".format(outdir), header=None)
                            # secchi['lake'] = lake_number
                            secchi['lake'] = position
                            all_temp_data.append(secchi[2].tolist())
                            all_temp_model.append(secchi[3].tolist())
                            secchi['depths'] = secchi[1] / lakeinfo.depth
                            all_temp_depth.append(secchi['depths'].tolist())
                            all_lake_temp.append(secchi['lake'].tolist())
                            lake_temp_data = secchi[2].tolist()
                            lake_temp_model = secchi[3].tolist()
                            lake_temp_depth = secchi['depths'].tolist()
                            lake_temp_lake = secchi['lake'].tolist()

                            slope, intercept, r_value, p_value, std_err = linregress(lake_temp_data, lake_temp_model)
                            results = Graphics(output_path).graphiqueTO(x=lake_temp_data, y=lake_temp_model,
                                                                        z=lake_temp_depth,
                                                                        symbol=lake_temp_lake,
                                                                        variable="Temperature (°C)",
                                                                        calibration=calibration, old=old,
                                                                        lakeid="_%s" % (lakeinfo.lake_id),
                                                                        outputfolder=outputfolder1)

                        # except:
                        #     print("Tztcompare.csv lake %s does not exist" % lakeinfo.lake_name)
                    else:
                        all_temp_data.append([])
                        all_temp_model.append([])
                        all_temp_depth.append([])
                        all_lake_temp.append([position])

                    if os.path.exists("{}/O2ztcompare.csv".format(outdir)):
                        if 1 == 1:  # try:
                            secchi = pd.read_csv("{}/O2ztcompare.csv".format(outdir), header=None)
                            # secchi['lake'] = lake_number
                            secchi['lake'] = position
                            all_oxygen_data.append(secchi[2].tolist())
                            all_oxygen_model.append(secchi[3].tolist())
                            secchi['depths'] = secchi[1] / lakeinfo.depth
                            all_oxygen_depth.append(secchi['depths'].tolist())
                            all_lake_oxygen.append(secchi['lake'].tolist())
                            lake_oxy_data = secchi[2].tolist()
                            lake_oxy_model = secchi[3].tolist()
                            lake_oxy_depth = secchi['depths'].tolist()
                            lake_oxy_lake = secchi['lake'].tolist()

                            slope, intercept, r_value, p_value, std_err = linregress(lake_oxy_data, lake_oxy_model)
                            results = Graphics(output_path).graphiqueTO(x=lake_oxy_data, y=lake_oxy_model,
                                                                        z=lake_oxy_depth,
                                                                        symbol=lake_oxy_lake,

                                                                        variable="Oxygen (mg/L)",
                                                                        calibration=calibration, old=old,
                                                                        lakeid="_%s" % (lakeinfo.lake_id),
                                                                        outputfolder=outputfolder1)

                        # except:
                        #     print("O2ztcompare.csv lake %s does not exist" % lakeinfo.lake_name)
                    else:
                        all_oxygen_data.append([])
                        all_oxygen_model.append([])
                        all_oxygen_depth.append([])
                        all_lake_oxygen.append([position])

        # except:
        # print("eror")

    xtemp = [item for sublist in all_temp_data for item in sublist]
    ytemp = [item for sublist in all_temp_model for item in sublist]
    ztemp = [item for sublist in all_temp_depth for item in sublist]
    symboltemp = [item for sublist in all_lake_temp for item in sublist]
    xoxy = [item for sublist in all_oxygen_data for item in sublist]
    yoxy = [item for sublist in all_oxygen_model for item in sublist]
    zoxy = [item for sublist in all_oxygen_depth for item in sublist]
    symboloxy = [item for sublist in all_lake_oxygen for item in sublist]

    # try:
    slope, intercept, r_value, p_value, std_err = linregress(xtemp, ytemp)
    results = Graphics(output_path).graphiqueTO(x=all_temp_data, y=all_temp_model, z=all_temp_depth, symbol=symboltemp,
                                                variable="Temperature (°C)", calibration=calibration,
                                                old=old)

    # except:
    #     print("here")

    # try:
    slope, intercept, r_value, p_value, std_err = linregress(xoxy, yoxy)
    results = Graphics(output_path).graphiqueTO(x=all_oxygen_data, y=all_oxygen_model, z=all_oxygen_depth,
                                                symbol=symboloxy, variable="Oxygen (mg/L)", calibration=calibration,
                                                old=old)
    # except:
    #     print("here")


def FishNiche_TO_graph_v2(lakes_listcsv="2017SwedenList.csv", calibration=False, modelid=2, scenarioid=2,
                          outputfolder="Postproc", old=False):
    # lakes_list = list(lakes_data.get("name").values())
    lakes = pd.read_csv(lakes_listcsv, encoding='ISO-8859-1')  # _only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    subid = list(lakes_data.get("subid").keys())
    y1A, y2B = 2001 - 5, 2010

    all_temp_model, all_temp_data, all_temp_depth, all_lake_temp = [], [], [], []
    all_oxygen_model, all_oxygen_data, all_oxygen_depth, all_lake_oxygen = [], [], [], []

    # lakes_list = list(lakes_data.get("name").values())
    position = -1
    for lake_number in range(0, len(lakes_list)):
        outputfolder1 = outputfolder
        if 1 == 1:  # try:
            if list(lakes["lake_id"])[lake_number] in [6950, 32276, 310, 14939, 30704, 31895, 99045, 33590, 33494, 698,
                                                       16765, 67035]:
                position += 1
                print(position, list(lakes["lake_id"])[lake_number])
                # if position > 11:
                #     print("here")
                if calibration:

                    lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                        list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                        list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                        list(lakes["Mean"])[lake_number],
                                        list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                        list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                        scenarioid=scenarioid, modelid=modelid, calibration=calibration,
                                        outputfolder=outputfolder, old=old)
                    outdir = lakeinfo.calibration_path
                    outputfolder1 = os.path.join(outputfolder, "result_old_calibration")
                    lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                            list(lakes["Mean"])[lake_number],
                                            list(lakes["longitude"])[lake_number],
                                            list(lakes["latitude"])[lake_number],
                                            list(lakes["volume"])[lake_number],
                                            list(lakes["Turnover"])[lake_number], 2, outputfolder=outputfolder,
                                            old=old, calibration=calibration)
                    lakec.variables_by_depth(lakec.start_year, lakec.end_year)
                    # lakeinfo.runlakefinal(2, 2, calibration=False)
                    print(outdir)
                    # cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                    #     outdir, 2001, 2010)
                    # print(cmd)
                    # os.system(cmd)
                    outputfolder1 = outputfolder

                else:
                    swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0, albice, albsnow = \
                        final_equation_parameters(list(lakes["longitude"])[lake_number],
                                                  list(lakes["latitude"])[lake_number],
                                                  list(lakes["depth"])[lake_number], list(lakes["Mean"])[lake_number],
                                                  list(lakes["C.L"])[lake_number], list(lakes["SC.L"])[lake_number],
                                                  list(lakes["Turnover"])[lake_number],
                                                  list(lakes["area"])[lake_number],
                                                  list(lakes["volume"])[lake_number],
                                                  list(lakes["sedimentArea"])[lake_number])
                    if old:
                        lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                            list(lakes["Mean"])[lake_number],
                                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                            list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                            scenarioid=scenarioid, modelid=modelid, calibration=calibration,
                                            outputfolder="Postproc", old=old)
                        outdircali = lakeinfo.calibration_path

                        lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                            list(lakes["Mean"])[lake_number],
                                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                            list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                            scenarioid=scenarioid, modelid=modelid, calibration=True,
                                            outputfolder=outputfolder, old=old)
                        lakec = CalibrationInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                                list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                list(lakes["Mean"])[lake_number],
                                                list(lakes["longitude"])[lake_number],
                                                list(lakes["latitude"])[lake_number],
                                                list(lakes["volume"])[lake_number],
                                                list(lakes["Turnover"])[lake_number], 2, outputfolder=outputfolder,
                                                old=old, calibration=calibration)
                        lakec.variables_by_depth(lakec.start_year, lakec.end_year)
                        outdir = lakeinfo.outdir
                        lakeinfo.runlakefinal(2, 2, calibration=False)
                        # print(outdircali)
                        # cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                        #     outdir, 2001, 2010)
                        # print(cmd)
                        # os.system(cmd)
                        outputfolder1 = outputfolder
                    else:
                        lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                            list(lakes["Mean"])[lake_number],
                                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                            list(lakes["volume"])[lake_number], list(lakes["Turnover"])[lake_number],
                                            swa_b1, swa_b0, c_shelter, i_sct, i_scv, i_sco, i_scdoc, k_sod, k_bod, kzn0,
                                            albice,
                                            albsnow, scenarioid=scenarioid, modelid=modelid, calibration=False,
                                            outputfolder=outputfolder, old=old)
                        # lakeinfo.lake_input(2,2)
                        outdir = lakeinfo.outdir
                        # cmd = 'matlab -wait -r -nosplash -nodesktop compare_model_result_old_data(\'%s\',%d,%d);quit' % (
                        #     outdir, 2001, 2010)
                        # print(cmd)
                        # os.system(cmd)
                if os.path.exists(os.path.join(lakeinfo.outdir_origin_cali,
                                               "Calibration_Complete.csv")):
                    if os.path.exists("{}/Tztcompare.csv".format(outdir)):
                        if 1 == 1:  # try:
                            secchi = pd.read_csv("{}/Tztcompare.csv".format(outdir), header=None)
                            # secchi['lake'] = lake_number
                            secchi['lake'] = position
                            all_temp_data.append(secchi[2].tolist())
                            all_temp_model.append(secchi[3].tolist())
                            secchi['depths'] = secchi[1] / lakeinfo.depth
                            all_temp_depth.append(secchi['depths'].tolist())
                            all_lake_temp.append(secchi['lake'].tolist())
                            lake_temp_data = secchi[2].tolist()
                            lake_temp_model = secchi[3].tolist()
                            lake_temp_depth = secchi['depths'].tolist()
                            lake_temp_lake = secchi['lake'].tolist()

                            slope, intercept, r_value, p_value, std_err = linregress(lake_temp_data, lake_temp_model)
                            # results = Graphics(output_path).graphiqueTO_v2(x=lake_temp_data, y=lake_temp_model, z=lake_temp_depth,
                            #                       symbol=lake_temp_lake,
                            #                       variable="Temperature (°C)", calibration=calibration, old=old,
                            #                       lakeid="_%s" % (lakeinfo.lake_id), outputfolder=outputfolder1)

                        # except:
                        #     print("Tztcompare.csv lake %s does not exist" % lakeinfo.lake_name)
                    else:
                        all_temp_data.append([])
                        all_temp_model.append([])
                        all_temp_depth.append([])
                        all_lake_temp.append([position])

                    if os.path.exists("{}/O2ztcompare.csv".format(outdir)):
                        if 1 == 1:  # try:
                            secchi = pd.read_csv("{}/O2ztcompare.csv".format(outdir), header=None)
                            # secchi['lake'] = lake_number
                            secchi['lake'] = position
                            all_oxygen_data.append(secchi[2].tolist())
                            all_oxygen_model.append(secchi[3].tolist())
                            secchi['depths'] = secchi[1] / lakeinfo.depth
                            all_oxygen_depth.append(secchi['depths'].tolist())
                            all_lake_oxygen.append(secchi['lake'].tolist())
                            lake_oxy_data = secchi[2].tolist()
                            lake_oxy_model = secchi[3].tolist()
                            lake_oxy_depth = secchi['depths'].tolist()
                            lake_oxy_lake = secchi['lake'].tolist()

                            # slope, intercept, r_value, p_value, std_err = linregress(lake_oxy_data, lake_oxy_model)
                            # results = Graphics(output_path).graphiqueTO_v2(x=lake_oxy_data, y=lake_oxy_model, z=lake_oxy_depth,
                            #                       symbol=lake_oxy_lake,
                            #
                            #                       variable="Oxygen (mg/L)", calibration=calibration, old=old,
                            #                       lakeid="_%s" % (lakeinfo.lake_id), outputfolder=outputfolder1)

                        # except:
                        #     print("O2ztcompare.csv lake %s does not exist" % lakeinfo.lake_name)
                    else:
                        all_oxygen_data.append([])
                        all_oxygen_model.append([])
                        all_oxygen_depth.append([])
                        all_lake_oxygen.append([position])

        # except:
        # print("eror")

    xtemp = [item for sublist in all_temp_data for item in sublist]
    ytemp = [item for sublist in all_temp_model for item in sublist]
    ztemp = [item for sublist in all_temp_depth for item in sublist]
    symboltemp = [item for sublist in all_lake_temp for item in sublist]
    xoxy = [item for sublist in all_oxygen_data for item in sublist]
    yoxy = [item for sublist in all_oxygen_model for item in sublist]
    zoxy = [item for sublist in all_oxygen_depth for item in sublist]
    symboloxy = [item for sublist in all_lake_oxygen for item in sublist]

    # try:
    slope, intercept, r_value, p_value, std_err = linregress(xtemp, ytemp)
    results = Graphics(output_path, width=3, height=3, font_family="Arial", size=11.1).graphiqueTO_v2(x=all_temp_data,
                                                                                                      y=all_temp_model,
                                                                                                      z=all_temp_depth,
                                                                                                      symbol=symboltemp,
                                                                                                      variable="Temperature (°C)",
                                                                                                      calibration=calibration,
                                                                                                      old=old)
    # results = Graphics(output_path, width=3, height=3, font_family="Times New Roman", size=12).graphiqueTO_v2(
    #     x=all_temp_data, y=all_temp_model, z=all_temp_depth, symbol=symboltemp, variable="Temperature (°C)",
    #     calibration=calibration, old=old)

    # except:
    #     print("here")

    # try:
    slope, intercept, r_value, p_value, std_err = linregress(xoxy, yoxy)
    results = Graphics(output_path, width=3, height=3, font_family="Arial", size=11.1).graphiqueTO_v2(x=all_oxygen_data,
                                                                                                      y=all_oxygen_model,
                                                                                                      z=all_oxygen_depth,
                                                                                                      symbol=symboloxy,
                                                                                                      variable="Oxygen (mg/L)",
                                                                                                      calibration=calibration,
                                                                                                      old=old)
    # results = Graphics(output_path, width=3, height=3, font_family="Times New Roman", size=12).graphiqueTO_v2(
    #     x=all_oxygen_data, y=all_oxygen_model, z=all_oxygen_depth, symbol=symboloxy, variable="Oxygen (mg/L)",
    #     calibration=calibration, old=old)

    # except:
    #     print("here")


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


def data_descriptif():
    xlxs_datat = r"lakes\sweden_inflow_data\Validation_data_for_lookup12.xlsx"
    lakes12 = pd.read_csv("2017SwedenList_only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_id = list(lakes12.set_index("subid").to_dict().get("lake_id").values())
    data_lakes = pd.DataFrame()
    for lake_id in lakes_id:
        obs_file = pd.read_excel(xlxs_datat, sheet_name="%s" % lake_id)
        obs_file["date"] = pd.to_datetime(obs_file["date"])
        for variable in ["O2(mg/l)", "Water temp (°C)", "Siktdjup (m)"]:
            y = np.array(obs_file["depth(max)"])
            depths = np.unique(y)
            x = np.array(obs_file["date"])
            dates = np.unique(x)
            obs_file1 = obs_file.set_index("date")
            data_final = pd.DataFrame(index=dates, columns=depths)
            for depth in depths:
                selected = obs_file1.loc[obs_file1['depth(max)'] == depth]
                selected = selected[variable]
                test = selected.index
                for date in list(test):
                    try:

                        data_final.loc[date, depth] = selected[date].mean()

                    except:
                        print("nothing")

                # result = pd.concat([data_final, selected], axis=1).reindex(data_final.index)
                # data_final.loc[variable] = selected

            # data = obs_file.pivot(index="date", columns="depth(max)", values="%s"%variable)
            numbers = data_final.count()
            max = depths[-1]
            aa = numbers[:max / 2]
            epi = numbers[:max / 2].idxmax()
            if numbers.loc[depths[0]] > (epi - 5):
                epi = depths[0]
            bb = numbers[max / 2:]
            hypo = numbers[max / 2:].idxmax()
            if numbers.loc[depths[-1]] > (hypo - 5):
                hypo = depths[-1]
            if variable == "Siktdjup (m)":
                data_final1 = data_final[[epi]]
            else:
                data_final1 = data_final[[epi, hypo]]
            ax = data_final1.plot(lw=2, colormap='jet', style='o--', title='lake %s' % lake_id, figsize=(20, 5))
            ax.set_xlabel("Date")
            if variable == "O2(mg/l)":
                ax.set_ylabel("Oxygen Concentration (mg/l)")
                vari = "O2"
            elif variable == "Water temp (°C)":
                ax.set_ylabel("Water Temperature (°C)")
                vari = "Temp"
            else:
                ax.set_ylabel("Secchi Depth (m)")
                vari = "Secchi"

            ax.legend(["%s m" % epi, "%s m" % hypo]);
            fig = ax.get_figure()
            fig.savefig("data_descriptif_%s_%s" % (lake_id, vari))


def violin_plot45(lakes_list1="2017SwedenList.csv", output_path="Postproc"):
    from datetime import timedelta, date
    # lake = lakes_list[lake_number]
    lakes = pd.read_csv(lakes_list1, encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())

    variables = [["Tzt.csv", "Change in Surface Temperature"],
                 ["O2zt.csv", "Change in Bottom Oxygen Concentration"],
                 ["His.csv", "Change in Ice Cover Duration"]]
    model_data = [["model", "lake", "volume", "depth", "scenario", variables[0][1], variables[1][1], variables[2][1]]]
    lakesss_data = pd.DataFrame(columns=["lake", "model", "volume", "depth",
                                         "dateM", "dateD", "historicalT", "rcp45T", "rcp85T", "diff45T", "diff85T",
                                         "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                                         "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
    # kernel = [["model", "lake", "scenario", variables[0][1], variables[1][1], variables[2][1]]]

    if os.path.exists("annually_average_T_Ice_cover_Oxygen.csv"):
        final_data = pd.read_csv("annually_average_T_Ice_cover_Oxygen.csv")
    else:
        aaaa = 0
        for modelid in [2]:
            m1, m2 = models[modelid]
            # if 1==1:
            if 1 == 1:  ##try:

                n = 1
                if aaaa == 0:
                    lakess_data = pd.DataFrame(
                        columns=["lake", "model", "volume", "depth",
                                 "dateM", "dateD", "historicalT", "rcp45T", "rcp85T",
                                 "diff45T", "diff85T",
                                 "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                                 "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
                else:
                    lakesss_data = lakesss_data.append(lakess_data, ignore_index=True)
                aaaa = 0
                lake_number = -1
                # if 1==1:
                for lake in lakes_list:
                    if lake != 72891:
                        lake_number += 1
                        lake_data = pd.DataFrame(
                            columns=["lake", "model", "volume", "depth",
                                     "dateM", "dateD", "historicalT", "rcp45T", "rcp85T",
                                     "diff45T", "diff85T",
                                     "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                                     "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
                        print(m2, lake, n, lake_number)
                        n += 1

                        volume = lakes_data.get('volume').get(lake)
                        depth = lakes_data.get('depth').get(lake)
                        eh = lakes_data.get('ebhex').get(lake)
                        eh = eh[2:] if eh[:2] == '0x' else eh
                        while len(eh) < 6:
                            eh = '0' + eh
                        d1, d2, d3 = eh[:2], eh[:4], eh[:6]

                        for scenarioid in [1, 5, 8]:
                            exA, y1A, exB, y1B = scenarios[scenarioid]
                            # y2A = y1A + 4
                            y2B = y1B + 4
                            outdir = os.path.join(output_path, d1, d2, d3,
                                                  'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (m1, exA, exB, m2, y1A, y2B))

                            lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                                list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                                list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                                list(lakes["Mean"])[lake_number],
                                                list(lakes["longitude"])[lake_number],
                                                list(lakes["latitude"])[lake_number],
                                                list(lakes["volume"])[lake_number], scenarioid, scenarioid=scenarioid,
                                                modelid=modelid)

                            # lake.variables_by_depth()
                            # lakeinfo.runlake(modelid,scenarioid)

                            if scenarioid == 1:
                                dstart = date(y1A, 1, 1)
                                dend = date(y2B, 12, 31)

                                # this will give you a list containing all of the dates
                                dd = [dstart + timedelta(days=x) for x in range((dend - dstart).days + 1)]

                                for year in range(y1A, y2B + 1):
                                    try:
                                        leapyear = date(year, 2, 29)
                                        print("%s is a leap year" % year)
                                        dd.remove(leapyear)
                                        if leapyear in dd:
                                            print("error")
                                    except:
                                        print("%s is not a leap year" % year)

                                date_stringsY = [d.strftime('%y') for d in dd]
                                date_stringsM = [d.strftime('%m') for d in dd]
                                date_stringsD = [d.strftime('%d') for d in dd]
                                lake_data["dateY"] = date_stringsY
                                lake_data["dateM"] = date_stringsM
                                lake_data["dateD"] = date_stringsD
                                lake_data["lake"] = lake
                                lake_data["model"] = modelid
                                lake_data["volume"] = volume
                                lake_data["depth"] = depth

                            for variable in [0, 1, 2]:
                                data = pd.read_csv(os.path.join(outdir, variables[variable][0]), header=None)
                                data['Date'] = pd.date_range(start=date(y1A, 1, 1), periods=len(data), freq='D')

                                data = data[~((data.Date.dt.month == 2) & (data.Date.dt.day == 29))]
                                data = data.drop(columns=['Date'])
                                if variable == 0:
                                    lake_data["%sT" % exA] = data[0]
                                elif variable == 1:
                                    lake_data["%sO" % exA] = data.iloc[:, -1]
                                else:
                                    icecoverduration = (data[6].sum()) / 10
                                    lake_data["%sI" % exA] = icecoverduration

                        # data_summary = lake_data.mean()
                        data_summary1 = lake_data

                        data_sum = lake_data
                        for rcp in ["45", "85"]:
                            for letter in ["T", "O", "I"]:

                                if letter == "T" or letter == "I":
                                    data_summary1["diff%s%s" % (rcp, letter)] = data_summary1[
                                                                                    "rcp%s%s" % (rcp, letter)].astype(
                                        float) - data_summary1["historical%s" % letter].astype(float)
                                elif letter == "O":
                                    data_summary1["diff%s%s" % (rcp, letter)] = (data_summary1[
                                                                                     "rcp%s%s" % (rcp, letter)].astype(
                                        float) - data_summary1["historical%s" % letter].astype(float)) * 0.001
                                    data_sum["diff%s%s" % (rcp, letter)] = (
                                            data_summary1["rcp%s%s" % (rcp, letter)].astype(float) -
                                            data_summary1["historical%s" % letter].astype(float))
                                    data_sum["diff%s%s" % (rcp, letter)] = data_sum["diff%s%s" % (rcp, letter)].astype(
                                        float) * 0.001

                                if letter == "I":
                                    data_summary1["ice%s%s" % (rcp, letter)] = data_summary1[
                                        "rcp%s%s" % (rcp, letter)].astype(float)
                                    data_summary1["icehisto%s" % (letter)] = data_summary1[
                                        "historical%s" % letter].astype(float)

                        # book = load_workbook("annually_average_T_Ice_cover_Oxygen_test.xlsx")
                        # writer = pd.ExcelWriter("annually_average_T_Ice_cover_Oxygen_test.xlsx", engine='openpyxl')
                        # writer.book = book
                        # data_summary1.to_excel(writer, sheet_name='lake_%s' % lake)
                        # writer.save()
                        # writer.close()

                        data_sum1 = data_summary1
                        data_summary1 = data_summary1.groupby('dateY').mean()
                        data_summary = data_summary1.mean()
                        if aaaa == 0:
                            lakess_data = lake_data
                            aaaa += 1
                        else:
                            lakess_data = lakess_data.append(lake_data, ignore_index=True)

                        for rcp in ["45", "85"]:
                            model_code = {1: 'KNM',
                                          2: 'DMI',
                                          3: 'MPI',
                                          4: 'MOH',
                                          5: 'IPS',
                                          6: 'CNR'}

                            try:
                                model_data.append(
                                    [model_code.get(modelid), lake, volume, depth, "rcp%s" % rcp,
                                     data_summary["diff%sT" % rcp],
                                     data_summary["diff%sO" % rcp], data_summary["diff%sI" % rcp]])
                            except:
                                print("")
            #
            # except:
            #     print("model %s doesnt exist" % (m1 + m2))
        headers = model_data.pop(0)
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        final_data = pd.DataFrame(model_data, columns=headers)
        lakesss_data.to_csv("annually_average_T_Ice_cover_Oxygen_originall.csv", index=False)
        final_data.to_csv("annually_average_T_Ice_cover_Oxygen.csv", index=False)

    final_data['Change in Ice Cover Duration'] = final_data['Change in Ice Cover Duration']/12
    print(final_data[final_data['scenario']=="rcp85"]['Change in Ice Cover Duration'].mean())
    print(final_data[final_data['scenario'] == "rcp85"]['Change in Ice Cover Duration'].std())
    print(final_data[final_data['scenario'] == "rcp45"]['Change in Ice Cover Duration'].mean())
    # Graphics("Postproc", height=3 * 3, font_family="Times New Roman", size=11.5).density_plot2(final_data,
    #                                                                                             ["temperature",
    #                                                                                              "oxygen", "ice"])
    Graphics("Postproc", height=3 * 3, font_family="Arial", size=11).density_plot2(final_data,
                                                                                    ["temperature",
                                                                                     "oxygen", "ice"])
    # Graphics("Postproc", width=6.5, height=3 * 3, font_family="Times New Roman",
    #          size=11.5).timeseries_habitat_by_model(partype="01")
    Graphics("Postproc", width=6.5, height=6.5, font_family="Arial", size=11).timeseries_habitat_by_model(partype="01")
    Graphics("Postproc", width=6.5, height=6.5, font_family="Arial", size=11).timeseries_habitat_by_model(partype="01",
                                                                                                           uninhabitable=False)
    # Graphics("Postproc", width=6.5, height=3 * 3, font_family="Times New Roman", size=12).timeseries_habitat_by_model(
    #     partype="01", uninhabitable=False)

def generate_timeseries_by_model(listmodels, listscenarios, lakelistfile, datafolder, partype="01"):
    i = 0
    complete_data = pd.DataFrame()
    for model in listmodels:
        for scenario in listscenarios:
            exA, y1A, exB, y1B = scenarios[scenario]
            y2B = y1B + 4
            m1, m2 = models[model]
            print(os.path.join(datafolder, 'fish_niche_export%s_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' % (partype, m1, exA, exB,  m2, y1A, y2B)))
            if not os.path.exists( os.path.join(datafolder, 'fish_niche_export%s_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' % (partype, m1, exA, exB,  m2, y1A, y2B))):
                # if os.os.path.exists(os.os.path.join(datafolder, 'fish_niche_export_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(m1, exA, exB, m2, y1A, y2B))):

                # cmd = 'generateVolumeTimeseries(\'%s\',\'%s\',\'%s\',\'%s\',%d,\'%s\',%d,\'%s\',\'%s\');' % (
                #     r'Clakes\2017SwedenList1.csv', m1, m2, exA, y1A, exB, y2B, datafolder,"part1")

                # Parallel(n_jobs=num_cores, verbose=10)(
                #     delayed(run_generateVolumeTimeseries_parallel)(lake_number, lakes_listcsv, modelid, scenarioid, calibration,
                #                                            old,
                #                                            outputfolder, new) for
                #     lake_number in range(0, len(lakes_list)))
                cmd = r'%s -wait -r -nosplash -nodesktop generateVolumeTimeseries(%s,%s,%s,%s,%d,%s,%d,%s,%s);quit' % (
                    "'%s'" % matlab_directory,
                    "'%s'" % r'lakes\2017SwedenList.csv',
                    "'%s'" % m1, "'%s'" % m2, "'%s'" % exA, y1A, "'%s'" % exB,
                    y2B, "'%s'" % datafolder, "'%s'" % "part1")
                print(cmd)
                os.system(cmd)

            # if os.path.exists(os.path.join(datafolder, 'fish_niche_export%s_EUR-11_%s_%s-%s_%s_%s0101-%s1231_part12.csv' % (partype,
            #                                                                                                    m1, exA,
            #                                                                                                    exB, m2,
            #                                                                                                    y1A,
            #                                                                                                    y2B))) and os.path.exists(os.path.join(datafolder, 'fish_niche_export%s_EUR-11_%s_%s-%s_%s_%s0101-%s1231_part2.csv' % (partype,
            #                                                                                                    m1, exA,
            #                                                                                                    exB, m2,
            #                                                                                                    y1A,
            #                                                                                                    y2B))):
            #     a = pd.read_csv(os.path.join(datafolder, 'fish_niche_export%s_EUR-11_%s_%s-%s_%s_%s0101-%s1231_part1.csv' % (partype,
            #                                                                                                    m1, exA,
            #                                                                                                    exB, m2,
            #                                                                                                    y1A,
            #                                                                                                    y2B)))
            #     b = pd.read_csv(os.path.join(datafolder, 'fish_niche_export%s_EUR-11_%s_%s-%s_%s_%s0101-%s1231_part2.csv' % (partype,
            #                                                                                                    m1, exA,
            #                                                                                                    exB, m2,
            #                                                                                                    y1A,
            #                                                                                                    y2B)))
            #     b = b.dropna(axis=1)
            #     merged = a.merge(b, on='lakeid')
            #     merged.to_csv(os.path.join(datafolder, 'fish_niche_export%s_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' % (partype,
            #                                                                                                    m1, exA,
            #                                                                                                    exB, m2,
            #                                                                                                    y1A,
            #                                                                                                    y2B)), index=False)
            if os.path.exists(os.path.join(datafolder, 'fish_niche_export%s_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' % (partype,
                                                                                                               m1, exA,
                                                                                                               exB, m2,
                                                                                                               y1A,
                                                                                                               y2B))):

                datasheet = os.path.join(datafolder, 'fish_niche_export%s_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' % (partype,
                                                                                                                m1, exA,
                                                                                                                exB, m2,
                                                                                                                y1A,
                                                                                                                y2B))
                # print(datasheet)
                timeseries = pd.read_csv(datasheet)
                timeseries['Date'] = pd.to_datetime(timeseries['Date'], format="%d.%m.%Y")
                timeseries_select = pd.DataFrame(
                    columns=['Date', 'Model', 'Scenario', 'Lake_group', 'Lake_id', 'pT', 'pO2', 'pPAR', 'Total Volume',
                             'phabitable'])

                timeseries_select['Date'] = timeseries['Date']
                timeseries_select['Scenario'] = exA
                timeseries_select['Lake_group'] = 2
                timeseries_select['Total Volume'] = timeseries['Total Volume']
                timeseries_select['Lake_id'] = timeseries['lakeid']
                timeseries_select.loc[timeseries['Total Volume'] < 1.0e7, 'Lake_group'] = 1
                timeseries_select.loc[timeseries['Total Volume'] > 5.0e9, 'Lake_group'] = 3
                timeseries_select['pT'] = timeseries['Pourcentage Volume with T < 15 C']
                timeseries_select['pO2'] = timeseries['Pourcentage Volume with O2 > 3000']
                timeseries_select['pPAR'] = timeseries['Pourcentage Volume with PPFD > %s' % int(partype)]
                timeseries_select['phabitable'] = timeseries['Pourcentage Volume satisfying all three previous']
                # timeseries_select.loc[timeseries_select['pO2'] <= timeseries_select['pPAR'], 'phabitable'] = \
                # timeseries_select['pT'] - (1 - timeseries_select['pO2'])
                # timeseries_select.loc[timeseries_select['pO2'] > timeseries_select['pPAR'], 'phabitable'] = \
                # timeseries_select['pT'] - (1 - timeseries_select['pPAR'])
                # timeseries_select.loc[timeseries_select['phabitable'] < 0, 'phabitable'] = 0
                timeseries_select['Model'] = model
                print('completed')
                if i == 0:
                    complete_data = timeseries_select
                    print('first')
                    i += 1
                else:
                    complete_data = complete_data.append(timeseries_select, ignore_index=True)
                    print('added')


    complete_data.loc[complete_data['Lake_group'] == 1].to_csv(os.path.join(datafolder, 'complete_data_1%s.csv' % partype),
                                                               index=False)
    print('1_save')
    complete_data.loc[complete_data['Lake_group'] == 2].to_csv(os.path.join(datafolder, 'complete_data_2%s.csv' % partype),
                                                               index=False)
    print('2_save')
    complete_data.loc[complete_data['Lake_group'] == 3].to_csv(os.path.join(datafolder, 'complete_data_3%s.csv' % partype),
                                                               index=False)
    print('end')




if __name__ == "__main__":
    # calibration_iteration()

    # lakes_list = ["Langtjern"]
    #
    lakes_list1 = "2017SwedenList_only_validation_12lakes.csv"
    #
    comparison_plot_v2(lakes_listcsv=lakes_list1, modelid=2, scenarioid=2, outputfolder="Postproc")
    FishNiche_mean_secchi_graph(calibration=True,old=False)
    FishNiche_mean_secchi_graph(calibration=False, old=False)
    FishNiche_TO_graph_v2(lakes_listcsv=lakes_list1, calibration=False, modelid=2, scenarioid=2,
                          outputfolder="Postproc", old=False)
    FishNiche_TO_graph_v2(lakes_listcsv=lakes_list1, calibration=True, modelid=2, scenarioid=2,
                          outputfolder="Postproc", old=False)
    violin_plot45(lakes_list1="2017SwedenList.csv", output_path="Postproc")
    print("part 2!")
    # summary_characteristics_lake(lakes_listcsv=lakes_list1, calibration=True, withfig=False, old=False,new=False, outputfolder="Postproc")
    print("part 3!")

    for model in [1,2,3,4,5,6]:
        for scenario in [1,2,3,4,5,6,7,8]:
            print("model :",model, "scenario :",scenario)
            summary_characteristics_lake_parallel(modelid=model, scenarioid=scenario)

    # summary_characteristics_lake_summary([1,2,3,4,5,6],[1,2,3,4,5,6,7,8], lakes_listcsv="2017SwedenList.csv", calibration=False,
    #                                      old=False)

    # lakes_list1 = "2017SwedenList.csv"
    # summary_characteristics_lake_parallel(modelid=2, scenarioid=1, lakes_listcsv="2017SwedenList.csv",
    #                                       calibration=False, old=False, outputfolder="Postproc", new=False)
    # summary_characteristics_lake1(1, lakes_listcsv="2017SwedenList.csv", modelid=2, scenarioid=1,
    #                               calibration=False, old=False, outputfolder="Postproc", new=False)

    # for model in [2]:
    #     for scenario in [2]:
    #         summary_characteristics_lake_parallel(modelid=model, scenarioid=scenario,lakes_listcsv=lakes_list1,
    #                                               calibration=False, old=False, outputfolder="Postproc",new=False)

    #         # summary_characteristics_lake1(4,modelid=model, scenarioid=scenario, lakes_listcsv=lakes_list1, calibration=False, old=False, outputfolder=r'Postproc',new=True)
    #         summary_characteristics_lake_parallel(modelid=model, scenarioid=scenario,
    #                                               calibration=True, old=False, outputfolder="Postproc",new=False)
    #
    # print("part 5!")
    #
    # summary_characteristics_lake(lakes_listcsv="2017SwedenList_only_validation_12lakes.csv", calibration=True,
    #                               withfig=True, old=False,new=False, outputfolder="Postproc")
    # #
    # # summary_characteristics_lake(lakes_listcsv=lakes_list1, calibration=True,
    # #                              withfig=True, old=False, outputfolder="Postproc",new=False)

    # ice_cover_comparison(lake_list="2017SwedenList.csv")
    # ice_cover_comparison(lake_list="2017SwedenList.csv", calibration=True)
    # #

    # for model in [2]:
    #     for scenario in [2,1,5,8]:
    #         summary_characteristics_lake_parallel(modelid=model, scenarioid=scenario, lakes_listcsv="2017SwedenList.csv",
    #                                               calibration=False, old=False, outputfolder="Postproc",new=False)

    generate_timeseries_by_model([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8],
                                 r'C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList.csv',
                                 "Postproc")  # ,"one_Pourcent")

    # fishout.generate_timeseries_by_model([2], [2, 1, 5, 8],
    #                              r'C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList.csv', "Postproc")  # ,"one_Pourcent")

    # summary_characteristics_lake1(0, lakes_listcsv="2017SwedenList_only_validation_12lakes.csv",modelid=4,scenarioid=1)
    # violin_plot45(lakes_list1="2017SwedenList.csv", output_path="Postproc")

    # comparison_plot_v2(lakes_listcsv=lakes_list1, modelid=2, scenarioid=2, outputfolder="Postproc")
    # # FishNiche_mean_secchi_graph(calibration=True,old=False)
    # # FishNiche_mean_secchi_graph(calibration=False, old=False)
    # # FishNiche_TO_graph_v2(lakes_listcsv=lakes_list1, calibration=False, modelid=2, scenarioid=2,
    # #                       outputfolder="Postproc", old=False)
    # # FishNiche_TO_graph_v2(lakes_listcsv=lakes_list1, calibration=True, modelid=2, scenarioid=2,
    # #                       outputfolder="Postproc", old=False)
    # # #
    # # # # for model in [2]:
    # # # #     for scenario in [2]:
    # # # #         summary_characteristics_lake_parallel(modelid=model, scenarioid=scenario,
    # # # #                                               calibration=False, old=False, outputfolder="Postproc",new=False)
    # # comparison_plot_v2(lakes_listcsv=lakes_list1, modelid=2, scenarioid=2, outputfolder="Postproc")
    # # ice_cover_comparison(lake_list="2017SwedenList.csv")
    # # ice_cover_comparison(lake_list="2017SwedenList.csv",calibration=True)
    # # violin_plot45(lakes_list1="2017SwedenList.csv", output_path="Postproc")
    # # lakes_list1 = "2017SwedenList_only_validation_12lakes.csv"
    # # summary_characteristics_lake(lakes_listcsv=lakes_list1, calibration=True,
    # #                              withfig=True, old=False, outputfolder="Postproc", new=False)
    # # Graphics("Postproc", height=3 * 3, font_family="Arial",size=12).density_plot2(final_data,["temperature","oxygen","ice"])
    #
    # #
    # # violin_plot("45", "2017SwedenList.csv", r"C:\Users\macot620\Documents\GitHub\Fish_niche\output")
    # violin_plot45("2017SwedenList.csv", r"C:\Users\macot620\Documents\GitHub\Fish_niche\output")
    # # comparison_plot(lakes_list1)
    # # summary_characteristics_lake(2,2,lakes_list1, calibration=True)
    # # summary_characteristics_lake_parallel(2,2,lakes_list1)
    # # FishNiche_mean_secchi_graph(lakes_list1)
    # # calibration_iteration()
    # # summary_characteristics_lake(calibration=True,lakes_listcsv="2017SwedenList_DATA.csv")
    # # summary_characteristics_lake_summary([2],[2],lakes_listcsv="2017SwedenList_DATA.csv",calibration=True)
    # # FishNiche_mean_secchi_graph(lakes_list1,True,False)
    # # # # FishNiche_mean_secchi_graph(lakes_list1,True,True)
    # # FishNiche_mean_secchi_graph(lakes_list1, False, False)
    # # FishNiche_TO_graph_v2(lakes_listcsv=lakes_list1,calibration=False,old=False,outputfolder=r'Postproc')
    # # FishNiche_TO_graph_v2(lakes_listcsv=lakes_list1, calibration=True, old=False, outputfolder=r'Postproc')
    # #
    # # lakes_list1 = "2017SwedenList.csv"
    # print(models.keys())
    # for model in models.keys():
    # for model in [1]:
    #     for scenario in scenarios.keys():
    #         if model == 2:
    #             if not scenario in [1,2,5,8]:
    #                 summary_characteristics_lake_parallel(modelid=model, scenarioid=scenario, lakes_listcsv=lakes_list1,
    #                                                       calibration=False, old=False, outputfolder="Postproc",new=True)
    #         else:
    #             try:
    #                 summary_characteristics_lake_parallel(modelid=model, scenarioid=scenario, lakes_listcsv=lakes_list1,
    #                                                   calibration=False, old=False, outputfolder="Postproc",new=True)
    #             except:
    #                 print("error with: model %s scenario %s"%(model,scenario))
    #         # summary_characteristics_lake1(4,modelid=model, scenarioid=scenario, lakes_listcsv=lakes_list1, calibration=False, old=False, outputfolder=r'Postproc',new=True)
    # #         # summary_characteristics_lake_parallel(modelid=model, scenarioid=scenario, lakes_listcsv=lakes_list1,
    # #         #                                       calibration=True, old=False, outputfolder=r'Postproc',
    # #         #                                       new=True)
    # # # FishNiche_TO_graph_v2(lakes_listcsv=lakes_list1, calibration=True,old=False)
    # # FishNiche_TO_graph(lakes_listcsv=lakes_list1, calibration=True,old=False)
    # # exploratory()
    # # summary_characteristics_lake_parallel(2, 2, lakes_list1, calibration=True)
    # # summary_characteristics_lake_parallel(2, 2, lakes_list1, calibration=False)
    # # FishNiche_mean_secchi_graph(calibration=True, old=True)
    # # Graphics("Postproc").violin_plot("85", "2017SwedenList.csv")
    # # violin_plot45("2017SwedenList.csv", "Postproc")
    # # FishNiche_TO_graph(calibration=True, old=True)
    # # calibration_iteration()
    # # summary_characteristics_lake_parallel(2,2, lakes_list1, calibration=True)
    # # summary_characteristics_lake_parallel(2, 2, lakes_list1, calibration=False)
    # # calibration_iteration()
    # # FishNiche_mean_secchi_graph(calibration=True, old=False)
    # # FishNiche_mean_secchi_graph(calibration=False,old=False)
    # # FishNiche_mean_secchi_graph(calibration=True,old=True)
    # # violin_plot45()
    # # summary_characteristics_lake(calibration=True)
    # # summary_obsvation_data()
    # # summary_characteristics_lake(2,2)
    # # FishNiche_TO_graph(lakes_list1)
    # # #violin_parallel()
    # # summary_characteristics_lake(2,2, "2017SwedenList_only_validation_12lakes.csv", calibration=True,
    # #                              withfig=True, old=False)
    # # summary_characteristics_lake(2, 2, "2017SwedenList_only_validation_12lakes.csv", calibration=False,
    # #                              withfig=True, old=False)
    # # summary_characteristics_lake(2, 2, "2017SwedenList_only_validation_12lakes.csv", calibration=True,
    # #                              withfig=True, old=True)
    # # summary_characteristics_lake_summary([2],[2], lakes_listcsv="2017SwedenList_only_validation_12lakes.csv", calibration=True,
    # #                                      old=False)
    # # summary_characteristics_lake_summary([2],[2], lakes_listcsv="2017SwedenList_only_validation_12lakes.csv", calibration=True,
    # #                                      old=True)

