#!/usr/bin/env python

""" Script for MyLake - ISIMIP
The main script launching the calibration and run of each lake with MyLake model.
"""

__author__ = "Julien Bellavance and Marianne Cote"

import os

import pandas as pd
from joblib import Parallel, delayed
import seaborn as sns
from lake_information import LakeInfo, graphique, graphiqueTO
from calibration_fish_niche import CalibrationInfo
from datetime import date, timedelta
import datetime
import multiprocessing
import matplotlib as plt
import numpy as np

from scipy.stats import pearsonr,linregress

num_cores = multiprocessing.cpu_count ()-2
output_path = r"C:\Users\macot620\Documents\GitHub\Fish_niche\output"
input_folder = r"C:\Users\macot620\Documents\GitHub\Fish_niche\sweden_inflow_data\Validation_data_for_lookup.csv"
forcing_data_path = r"F:\cordex"
# output_path = r"F:\output-21-08-2018"

lakes = pd.read_csv("2017SwedenList_only_validation_12lakes.csv", encoding='ISO-8859-1')
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

def calibration_iterationoxy():
    """
    Simple function to call the calibration of all lakes.
    :return: None
    """
    print(num_cores)
    for lake_number in range(0,5):#len(lakes_list)):

        run_calibrationsoxy(lake_number)


def run_calibrationsoxy(lake_number):
    """
        Intermediary function to call Nelder-mead optimization function for a single lake.

        :param lake_number: Type int. The subid of the lake to calibrate.
        :return: None
    """

    lake_name = lakes_list[lake_number]
    print(lake_name)

    lake = CalibrationInfo(lake_name, list(lakes["lake_id"])[lake_number], list(lakes["subid"])[lake_number],
                           list(lakes["ebhex"])[lake_number], list(lakes["area"])[lake_number],
                           list(lakes["depth"])[lake_number], list(lakes["longitude"])[lake_number],
                           list(lakes["latitude"])[lake_number], list(lakes["volume"])[lake_number], 2)


    test = True
    print( "%s/Calibration_Completescore.csv"%lake.calibration_path)
    if not os.path.exists("%s/Calibration_CompleteOXY.csv" % lake.calibration_path):
        if not os.path.exists( "%s/Calibration_Complete.csv"%lake.calibration_path):
            print("Calibration for {} is NOT already complete.\n".format(lake_name))
            lake = CalibrationInfo(lake_name, list(lakes["lake_id"])[lake_number], list(lakes["subid"])[lake_number],
                                   list(lakes["ebhex"])[lake_number], list(lakes["area"])[lake_number],
                                   list(lakes["depth"])[lake_number], list(lakes["longitude"])[lake_number],
                                   list(lakes["latitude"])[lake_number], list(lakes["volume"])[lake_number], 2)
            lake.lake_input(2, 2)
            return None
        else:
            lake.lake_input(2,2)
            lake.variables_by_depth()
            test = lake.runlake(2, 2)


    if test == False:
        if not os.path.exists("{}/2020input".format(lake.calibration_path)):
            print("Calibration can't {} be completed. input file doesn't exist.\n".format(lake_name))
            return None
        elif not os.path.exists(os.path.join(lake.calibration_path, "2020input")) and \
                not os.path.exists(os.path.join(lake.calibration_path, "2020init")) and \
                not os.path.exists(os.path.join(lake.calibration_path, "2020par")):
            print("not all initial files existing for %s" % lake_name)
        elif os.path.exists(os.path.join(lake.calibration_path, "Calibration_problem.txt")):
            print("Unable to calibration {}.\n".format(lake_name))
            return None

        else:
            cmd = "matlab -wait -r -nosplash -nodesktop MyLake_optimizer_oxygen(%d,%d,'%s','%s','%s','%s',%f,%f);quit" % \
                  (lake.start_year, lake.end_year, os.path.join(lake.calibration_path, "2020par"),
                   os.path.join(lake.calibration_path, "2020input"), os.path.join(lake.calibration_path, "2020init"),
                   lake.calibration_path, lake.latitude, lake.longitude)

            print(cmd)

            os.system(cmd)
            lake = CalibrationInfo(lake_name, list(lakes["lake_id"])[lake_number], list(lakes["subid"])[lake_number],
                                   list(lakes["ebhex"])[lake_number], list(lakes["area"])[lake_number],
                                   list(lakes["depth"])[lake_number], list(lakes["longitude"])[lake_number],
                                   list(lakes["latitude"])[lake_number], list(lakes["volume"])[lake_number], 2)
            lake.lake_input(2, 2)

        if os.path.exists("%s/Calibration_Completescore.csv" % lake.calibration_path):
            print("Calibration for {} is now complete.\n".format(lake_name))
            # cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\');quit' % (
            #     os.path.join(lake.calibration_path, "2020init"), os.path.join(lake.calibration_path, "2020par"),
            #     os.path.join(lake.calibration_path, "2020input"), lake.start_year, lake.end_year, lake.calibration_path)
            #
            #
            # print(cmd)
            #
            # os.system(cmd)
            lake = CalibrationInfo(lake_name, list(lakes["lake_id"])[lake_number], list(lakes["subid"])[lake_number],
                                   list(lakes["ebhex"])[lake_number], list(lakes["area"])[lake_number],
                                   list(lakes["depth"])[lake_number], list(lakes["longitude"])[lake_number],
                                   list(lakes["latitude"])[lake_number], list(lakes["volume"])[lake_number], 2)
            lake.lake_input(2, 2)

    else:
        cmd = "matlab -wait -r -nosplash -nodesktop MyLake_optimizer_oxygen(%d,%d,'%s','%s','%s','%s',%f,%f);quit" % \
              (lake.start_year, lake.end_year, os.path.join(lake.calibration_path, "2020par"),
               os.path.join(lake.calibration_path, "2020input"), os.path.join(lake.calibration_path, "2020init"),
               lake.calibration_path, lake.latitude, lake.longitude)

        print(cmd)

def calibration_iteration():
    """
    Simple function to call the calibration of all lakes.
    :return: None
    """
    print(num_cores)
    for lake_number in range(0,5):#len(lakes_list)):

        run_calibrations(lake_number)


def run_calibrations(lake_number):
    """
        Intermediary function to call Nelder-mead optimization function for a single lake.

        :param lake_number: Type int. The subid of the lake to calibrate.
        :return: None
    """

    lake_name = lakes_list[lake_number]
    print(lake_name)

    lake = CalibrationInfo(lake_name, list(lakes["lake_id"])[lake_number], list(lakes["subid"])[lake_number],
                           list(lakes["ebhex"])[lake_number], list(lakes["area"])[lake_number],
                           list(lakes["depth"])[lake_number], list(lakes["longitude"])[lake_number],
                           list(lakes["latitude"])[lake_number], list(lakes["volume"])[lake_number], 2)


    test = True
    print( "%s/Calibration_Completescore.csv"%lake.calibration_path)
    if os.path.exists( "%s/Calibration_Complete.csv"%lake.calibration_path):
        print("Calibration for {} is already complete.\n".format(lake_name))
        lake = CalibrationInfo(lake_name, list(lakes["lake_id"])[lake_number], list(lakes["subid"])[lake_number],
                               list(lakes["ebhex"])[lake_number], list(lakes["area"])[lake_number],
                               list(lakes["depth"])[lake_number], list(lakes["longitude"])[lake_number],
                               list(lakes["latitude"])[lake_number], list(lakes["volume"])[lake_number], 2)
        lake.lake_input(2, 2)
        return None
    else:
        lake.lake_input(2,2)
        lake.variables_by_depth()
        test = lake.runlake(2, 2)


    if test == False:
        if not os.path.exists("{}/2020input".format(lake.calibration_path)):
            print("Calibration can't {} be completed. input file doesn't exist.\n".format(lake_name))
            return None
        elif not os.path.exists(os.path.join(lake.calibration_path, "2020input")) and \
                not os.path.exists(os.path.join(lake.calibration_path, "2020init")) and \
                not os.path.exists(os.path.join(lake.calibration_path, "2020par")):
            print("not all initial files existing for %s" % lake_name)
        elif os.path.exists(os.path.join(lake.calibration_path, "Calibration_problem.txt")):
            print("Unable to calibration {}.\n".format(lake_name))
            return None

        else:
            cmd = "matlab -wait -r -nosplash -nodesktop MyLake_optimizer(%d,%d,'%s','%s','%s','%s',%f,%f);quit" % \
                  (lake.start_year, lake.end_year, os.path.join(lake.calibration_path, "2020par"),
                   os.path.join(lake.calibration_path, "2020input"), os.path.join(lake.calibration_path, "2020init"),
                   lake.calibration_path, lake.latitude, lake.longitude)

            print(cmd)

            os.system(cmd)
            lake = CalibrationInfo(lake_name, list(lakes["lake_id"])[lake_number], list(lakes["subid"])[lake_number],
                                   list(lakes["ebhex"])[lake_number], list(lakes["area"])[lake_number],
                                   list(lakes["depth"])[lake_number], list(lakes["longitude"])[lake_number],
                                   list(lakes["latitude"])[lake_number], list(lakes["volume"])[lake_number], 2)
            lake.lake_input(2, 2)

        if os.path.exists("%s/Calibration_Completescore.csv" % lake.calibration_path):
            print("Calibration for {} is now complete.\n".format(lake_name))
            # cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\');quit' % (
            #     os.path.join(lake.calibration_path, "2020init"), os.path.join(lake.calibration_path, "2020par"),
            #     os.path.join(lake.calibration_path, "2020input"), lake.start_year, lake.end_year, lake.calibration_path)
            #
            #
            # print(cmd)
            #
            # os.system(cmd)
            lake = CalibrationInfo(lake_name, list(lakes["lake_id"])[lake_number], list(lakes["subid"])[lake_number],
                                   list(lakes["ebhex"])[lake_number], list(lakes["area"])[lake_number],
                                   list(lakes["depth"])[lake_number], list(lakes["longitude"])[lake_number],
                                   list(lakes["latitude"])[lake_number], list(lakes["volume"])[lake_number], 2)
            lake.lake_input(2, 2)

    else:
        cmd = "matlab -wait -r -nosplash -nodesktop MyLake_optimizer(%d,%d,'%s','%s','%s','%s',%f,%f);quit" % \
              (lake.start_year, lake.end_year, os.path.join(lake.calibration_path, "2020par"),
               os.path.join(lake.calibration_path, "2020input"), os.path.join(lake.calibration_path, "2020init"),
               lake.calibration_path, lake.latitude, lake.longitude)

        print(cmd)

def summary_characteristics_lake():
    lakes_list = list(lakes_data.get("name").values())
    summary = pd.DataFrame(index=range(len(lakes_list)),
                           columns=[ "lake_id","subid","lake_name","ebhex", "i_scdoc", "swa_b1", "swa_b0", "k_bod", "k_sod", "kz_n0",
                                     "c_shelter", "alb_snow", "alb_ice", "i_scv", "i_sct","i_scO", "area", "depth", "longitude", "latitude", "volume", "Score","nrmseT","nrmseO","nrmseS","rmseT","rmseO","rmseS","rT","rO","rS"])
    i = 0

    for lake_number in range(0, len(lakes_list)):
        lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                            list(lakes["volume"])[lake_number])
        #lakeinfo.lake_input(2,2)
        if os.path.exists(os.path.join(lakeinfo.calibration_path,"Calibration_Completescore.csv")):

            #caracteristics
            summary.loc[i, 'lake_name'] = lakeinfo.lake_name
            summary.loc[i, "subid"] = lakeinfo.subid
            summary.loc[i, "lake_id"] = lakeinfo.lake_id
            summary.loc[i, "ebhex"] = lakeinfo.ebhex
            summary.loc[i, "longitude"] = lakeinfo.longitude
            summary.loc[i, "latitude"] = lakeinfo.latitude
            summary.loc[i, "depth"] = lakeinfo.depth
            summary.loc[i, "area"] = lakeinfo.area
            summary.loc[i, "volume"] = lakeinfo.volume

            #paramters
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



            score_data = pd.read_csv(os.path.join(lakeinfo.calibration_path,"Calibration_Completescore.csv"),header=None)


            rmse,rmsen,r_sq = lakeinfo.performance_analysis()

            # scores
            summary.loc[i, "Score"] = sum(rmsen)
            summary.loc[i, "nrmseT"] = rmsen[0]
            summary.loc[i, "nrmseO"] = rmsen[1]
            summary.loc[i, "nrmseS"] = rmsen[2]

            #rmse for comparison
            summary.loc[i, "rmseT"] = rmse[0]
            summary.loc[i, "rmseO"] = rmse[1]
            summary.loc[i, "rmseS"] = rmse[2]
            summary.loc[i, "rT"] = r_sq[0]
            summary.loc[i, "rO"] = r_sq[1]
            summary.loc[i, "rS"] = r_sq[2]

            lakeinfo.outputfile()
            if os.path.exists(os.path.join(lakeinfo.calibration_path, "strat.csv")) and os.path.exists(
                    os.path.join(lakeinfo.calibration_path, "watertemp.csv")) \
                    and os.path.exists(os.path.join(lakeinfo.calibration_path, "thermodepth.csv")):
                thermodepth = pd.read_csv(os.path.join(lakeinfo.calibration_path, "thermodepth.csv"),header=None)
                thermodepth= thermodepth[thermodepth[1] != 0]

                thermocline = thermodepth.mean()[1]
                #lakeinfo.comparison_obs_sims(thermocline)

            i += 1

    summary.to_csv("summary_info_lake_lakes.csv", index=False)
    colorpalette = sns.color_palette("colorblind", 7)

    #sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.set(font_scale=2)
    plt.pyplot.figure(figsize=(10,8))
    sns.set_style("ticks")

    plt.pyplot.grid(False)
    sns.distplot(summary["rmseT"], hist = False, kde = True,kde_kws = {'shade': True, 'linewidth': 3},color=colorpalette[0], label=None)
    plt.pyplot.ylabel("Density")
    plt.pyplot.xlabel("RMSE (Celcius)")
    plt.pyplot.savefig("densityrmseT.png")
    plt.pyplot.close()
    sns.set(font_scale=2)
    plt.pyplot.figure(figsize=(10,8))
    sns.set_style("ticks")
    plt.pyplot.grid(False)
    sns.distplot(summary["nrmseT"], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},color=colorpalette[0], label="Temperature")
    sns.distplot(summary["nrmseO"], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},color=colorpalette[1], label="Oxygen")
    sns.distplot(summary["nrmseS"], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},color=colorpalette[2], label="Secchi depth")
    plt.pyplot.ylabel("Density")
    plt.pyplot.xlabel("Normalised RMSE")
    plt.pyplot.savefig("density.png")
    plt.pyplot.close()

    dataall = pd.read_csv("2017SwedenList.csv",encoding='ISO-8859-1')
    dataall["area"] = dataall["area"]*1e-7
    dataall["volume"] = dataall["volume"] * 1e-9
    dataall["sedimentArea"] = dataall["sedimentArea"]*1e-18
    sns.set(font_scale=2)
    plt.pyplot.figure(figsize=(10,8))
    sns.set_style("ticks")
    plt.pyplot.grid(False)
    sns.distplot(dataall["area"],color=colorpalette[0], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},label=None)
    plt.pyplot.ylabel("Density")
    plt.pyplot.xlabel("Area (x10 km2)")
    plt.pyplot.savefig("area.png")
    plt.pyplot.close()
    sns.set(font_scale=2)
    plt.pyplot.figure(figsize=(10,8))
    sns.set_style("ticks")
    plt.pyplot.grid(False)
    sns.distplot(dataall["depth"], color=colorpalette[2], hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3},label=None)
    plt.pyplot.ylabel("Density")
    plt.pyplot.xlabel("Depth (m)")
    plt.pyplot.savefig("depth.png")
    plt.pyplot.close()
    sns.set(font_scale=2)
    plt.pyplot.figure(figsize=(10,8))
    sns.set_style("ticks")
    plt.pyplot.grid(False)
    sns.distplot(dataall["longitude"], color=colorpalette[4], hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3},label=None)
    plt.pyplot.ylabel("Density")
    plt.pyplot.xlabel("Longitude")
    plt.pyplot.savefig("lon.png")
    plt.pyplot.close()
    sns.set(font_scale=2)
    plt.pyplot.figure(figsize=(10,8))
    sns.set_style("ticks")
    plt.pyplot.grid(False)
    sns.distplot(dataall["latitude"], color=colorpalette[5], hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3},label=None)
    plt.pyplot.ylabel("Density")
    plt.pyplot.xlabel("Latitude")
    plt.pyplot.savefig("latitude.png")
    plt.pyplot.close()
    sns.set(font_scale=2)
    plt.pyplot.figure(figsize=(10,8))
    sns.set_style("ticks")
    plt.pyplot.grid(False)
    sns.distplot(dataall["volume"], color=colorpalette[1], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},label=None)
    plt.pyplot.ylabel("Density")
    plt.pyplot.xlabel("Volume (km3)")
    plt.pyplot.savefig("vol.png")
    plt.pyplot.close()
    sns.set(font_scale=2)
    plt.pyplot.figure(figsize=(10,8))
    sns.set_style("ticks")
    plt.pyplot.grid(False)
    mean = dataall["depth.mean"].dropna()
    sns.distplot(mean, color=colorpalette[3], hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3},label=None)
    plt.pyplot.ylabel("Density")
    plt.pyplot.xlabel("Mean depth (m)")
    plt.pyplot.savefig("mean.png")
    plt.pyplot.close()
    sns.set(font_scale=2)
    plt.pyplot.figure(figsize=(10,8))
    sns.set_style("ticks")
    plt.pyplot.grid(False)
    sns.distplot(dataall["sedimentArea"], color=colorpalette[6], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},label=None)
    plt.pyplot.ylabel("Density")
    plt.pyplot.xlabel("sedimentArea (x10^12 km\u00b2) ")
    plt.pyplot.savefig("sediment.png")
    plt.pyplot.close()


def FishNiche_mean_secchi_graph():

    y1A, y2B = 2001-5,2010

    all_mean_secchi_model, all_mean_secchi_data = [], []
    all_std_secchi_model, all_std_secchi_data = [], []

    lakes_list = list(lakes_data.get("name").values())


    for lake_number in range(0, len(lakes_list)):
        lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                            list(lakes["volume"])[lake_number])
        lakeinfo.lake_input(2, 2)
        if os.path.exists(os.path.join(lakeinfo.calibration_path, "Calibration_Completescore.csv")):
            if os.path.exists("{}/Secchicompare.csv".format(lakeinfo.calibration_path)):
                try:
                    secchi = pd.read_csv("{}/Secchicompare.csv".format(lakeinfo.calibration_path),header=None)
                    all_mean_secchi_data.append(secchi.mean()[2])
                    all_mean_secchi_model.append(secchi.mean()[3])
                    all_std_secchi_data.append(secchi.std()[2])
                    all_std_secchi_model.append(secchi.std()[3])
                except:
                    print("Secchicompare.csv lake %s does not exist"%lakeinfo.lake_name)



    mean_model_data = all_mean_secchi_model
    std_model_data = all_std_secchi_model
    slope, intercept, r_value, p_value, std_err = linregress(all_mean_secchi_data, mean_model_data)

    results = graphique(all_mean_secchi_data, mean_model_data, all_std_secchi_data, std_model_data, r_value, slope, intercept)


def FishNiche_TO_graph():

    y1A, y2B = 2001 - 5, 2010


    all_temp_model, all_temp_data,all_temp_depth, all_lake_temp = [], [], [], []
    all_oxygen_model, all_oxygen_data, all_oxygen_depth, all_lake_oxygen = [], [], [], []

    lakes_list = list(lakes_data.get("name").values())

    for lake_number in range(0, len(lakes_list)):
        lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                            list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                            list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                            list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                            list(lakes["volume"])[lake_number])
        lakeinfo.lake_input(2, 2)
        if os.path.exists(os.path.join(lakeinfo.calibration_path, "Calibration_Completescore.csv")):
            if os.path.exists("{}/Tztcompare.csv".format(lakeinfo.calibration_path)):
                try:
                    secchi = pd.read_csv("{}/Tztcompare.csv".format(lakeinfo.calibration_path), header=None)
                    secchi['lake'] = lake_number
                    all_temp_data.append(secchi[2].tolist())
                    all_temp_model.append(secchi[3].tolist())
                    secchi['depths'] = secchi[1]/lakeinfo.depth
                    all_temp_depth.append(secchi['depths'].tolist())
                    all_lake_temp.append(secchi['lake'].tolist())
                except:
                    print("Tztcompare.csv lake %s does not exist" % lakeinfo.lake_name)

            if os.path.exists("{}/O2ztcompare.csv".format(lakeinfo.calibration_path)):
                try:
                    secchi = pd.read_csv("{}/O2ztcompare.csv".format(lakeinfo.calibration_path), header=None)
                    secchi['lake'] = lake_number
                    all_oxygen_data.append(secchi[2].tolist())
                    all_oxygen_model.append(secchi[3].tolist())
                    secchi['depths'] = secchi[1] / lakeinfo.depth
                    all_oxygen_depth.append(secchi['depths'].tolist())
                    all_lake_oxygen.append(secchi['lake'].tolist())
                except:
                    print("O2ztcompare.csv lake %s does not exist" % lakeinfo.lake_name)

    xtemp = [item for sublist in all_temp_data for item in sublist]
    ytemp = [item for sublist in all_temp_model for item in sublist]
    ztemp = [item for sublist in all_temp_depth for item in sublist]
    symboltemp = [item for sublist in all_lake_temp for item in sublist]
    xoxy = [item for sublist in all_oxygen_data for item in sublist]
    yoxy = [item for sublist in all_oxygen_model for item in sublist]
    zoxy = [item for sublist in all_oxygen_depth for item in sublist]
    symboloxy = [item for sublist in all_lake_oxygen for item in sublist]

    slope, intercept, r_value, p_value, std_err = linregress(xtemp,ytemp)
    results = graphiqueTO(all_temp_data,all_temp_model,all_temp_depth,symboltemp,r_value,slope,intercept,"Temperature (°C)")

    slope, intercept, r_value, p_value, std_err = linregress(xoxy,yoxy)
    results = graphiqueTO(all_oxygen_data,all_oxygen_model,all_oxygen_depth, symboloxy, r_value, slope, intercept, "Oxygen (mg/L)")


def Area_at_depth():
    lakes = pd.read_csv("2017SwedenList_only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_small = lakes[lakes["volume"]< 1.0e7]['lake_id'].tolist()
    lakes_medium = lakes[lakes["volume"]>= 1.0e7 ]
    lakes_medium = lakes_medium[lakes_medium['volume'] <= 5.0e9]['lake_id'].tolist()
    lakes_large = lakes[lakes["volume"] >  5.0e9]['lake_id'].tolist()

    T_list = [2, 4, 6, 8, 10, 12, 13, 14, 15]
    light_list = [0.5, 1, 2, 4, 8, 16, 24, 32, 48]
    AreaDays=[]
    Areaday=[]
    NTG=[]
    for taille in ['small','medium','large']:
        for x in T_list:
            AreaDaysT = []
            AreadayT = []
            NTGT = []
            for y in light_list:
                if y == 0.5:
                    y = '0.50'
                df = pd.read_csv(os.path.join("F:\output", "fish_niche_Area_Light%s_T%s_2001-2010.csv"%(y,x)))
                if taille == 'small':
                    dataset = df[df['LakeId'].isin(lakes_small)].mean()
                elif taille== 'medium':
                    dataset = df[df['LakeId'].isin(lakes_medium)].mean()
                else:
                    dataset = df[df['LakeId'].isin(lakes_large)].mean()
                AreaDaysT.append(dataset['AreaDays'])
                AreadayT.append(dataset['Areaday'])
                NTGT.append(dataset['NTGdays'])
            AreaDays.append(AreaDaysT)
            Areaday.append(AreadayT)
            NTG.append(NTGT)

        x = np.array(T_list)
        y = np.array(light_list)

        #x = np.linspace(0, 5, 50)
        #y = np.linspace(0, 5, 40)
        X, Y = np.meshgrid(x, y)
        #Z = f(X, Y)

        ZAreaday = np.array(Areaday)

        ZAreadays = np.array(AreaDays)
        ZNTG = np.array(NTG)


        sns.set(font_scale=2)
        plt.pyplot.figure(figsize=(10,8))
        sns.set_style("ticks")
        plt.pyplot.grid(False)

        plt.pyplot.imshow(ZAreaday, extent=[2, 15, 0.5, 48], origin='lower',cmap='RdGy',interpolation='nearest', aspect='auto')
        plt.pyplot.colorbar()

        plt.pyplot.savefig("Areaday_%s.png"%taille)
        plt.pyplot.close()
        sns.set(font_scale=2)
        plt.pyplot.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.pyplot.grid(False)
        plt.pyplot.imshow(ZAreadays, extent=[2, 15, 0.5, 48],origin='lower',cmap='ocean',interpolation='nearest', aspect='auto')
        plt.pyplot.colorbar()
        plt.pyplot.savefig("Areadays_%s.png"%taille)
        plt.pyplot.close()
        sns.set(font_scale=2)
        plt.pyplot.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.pyplot.grid(False)
        plt.pyplot.subplot(313)
        plt.pyplot.imshow(ZNTG, extent=[2, 15, 0.5, 48], origin='lower',cmap='bone',interpolation='nearest', aspect='auto')
        plt.pyplot.colorbar()
        plt.pyplot.axis(aspect='image')
        plt.pyplot.savefig("NTG_%s.png"%taille)
        plt.pyplot.close()


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

def violin_parallel():
    """
    Simple function to call a parallel calibration of all lakes.
    :return: None
    """
    print(num_cores)
    # for lake in lakes_list:
    #     run_calibrations(lake)
    Parallel(n_jobs=num_cores, verbose=10)(delayed(violin_plot)(lakenumber) for lakenumber in range(0, len(lakes_list)))


def violin_plot():
    #lake = lakes_list[lake_number]
    sns.set_color_codes("colorblind")
    sns.set_context("paper", 2.5)
    variables = [["Tzt.csv", "Change in Surface Temperature ($^\circ$C)"],
                 ["O2zt.csv", "Change in Bottom Oxygen\n Concentration (mg m-2)"],
                 ["His.csv", "Change in Ice Cover Duration (day)"]]
    model_data = [["model", "lake","volume","depth", "scenario", variables[0][1], variables[1][1] ,variables[2][1]]]
    lakesss_data = pd.DataFrame(columns=["lake","model","volume","depth",
"dateM","dateD","historicalT", "rcp45T", "rcp85T", "diff45T", "diff85T",
                                              "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                                              "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
    # kernel = [["model", "lake", "scenario", variables[0][1], variables[1][1], variables[2][1]]]
    aaaa = 0
    for modelid in [2]:
        m1, m2 = models[modelid]
        # if 1==1:
        try:


            n = 1
            if aaaa == 0:
                lakess_data = pd.DataFrame(
                    columns=["lake","model","volume","depth",
"dateM","dateD", "historicalT", "rcp45T", "rcp85T",
                             "diff45T", "diff85T",
                             "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                             "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
            else:
                lakesss_data = lakesss_data.append(lakess_data,ignore_index=True)
            aaaa = 0
            lake_number = -1
            # if 1==1:
            for lake in lakes_list:
                lake_number += 1
                lake_data = pd.DataFrame(
                    columns=["lake","model","volume","depth",
"dateM","dateD", "historicalT", "rcp45T", "rcp85T",
                             "diff45T", "diff85T",
                             "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                             "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
                print(m2, lake, n,lake_number)
                n += 1

                volume = lakes_data.get('volume').get(lake)
                depth = lakes_data.get('depth').get(lake)
                eh = lakes_data.get('ebhex').get(lake)
                eh = eh[2:] if eh[:2] == '0x' else eh
                while len(eh) < 6:
                    eh = '0' + eh
                d1, d2, d3 = eh[:2], eh[:4], eh[:6]

                for scenarioid in [1,  8]:
                    exA, y1A, exB, y1B = scenarios[scenarioid]
                    # y2A = y1A + 4
                    y2B = y1B + 4
                    outdir = os.path.join(output_path, d1, d2, d3,
                                          'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % ( m1, exA, exB, m2, y1A, y2B))

                    lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                             list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                             list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                             list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                             list(lakes["volume"])[lake_number],scenarioid)

                    #lake.variables_by_depth()
                    lakeinfo.runlake(modelid,scenarioid)
                    

                    if scenarioid == 1:
                        dstart = date(y1A, 1, 1)
                        dend = date(y2B, 12, 31)

                        # this will give you a list containing all of the dates
                        dd = [dstart + timedelta(days=x) for x in range((dend - dstart).days + 1)]

                        date_stringsM = [d.strftime('%m') for d in dd]
                        date_stringsD = [d.strftime('%d') for d in dd]
                        lake_data["dateM"] = date_stringsM
                        lake_data["dateD"] = date_stringsD
                        lake_data["lake"] = lake
                        lake_data["model"] = modelid
                        lake_data["volume"] = volume
                        lake_data["depth"] = depth


                    for variable in [0, 1, 2]:
                        data = pd.read_csv(os.path.join(outdir, variables[variable][0]), header=None)

                        if variable == 0:
                            lake_data["%sT" % exA] = data[0]
                        elif variable == 1:
                            lake_data["%sO" % exA] = data.iloc[:, -1]*0.001
                        else:
                            icecoverduration = (data[6].sum()) / 10
                            lake_data["%sI" % exA] = icecoverduration

                data_summary = lake_data.mean()
                for rcp in [ "85"]:
                    for letter in ["T", "O", "I"]:
                        data_summary["diff%s%s" % (rcp, letter)] = data_summary["rcp%s%s" % (rcp, letter)] - data_summary[
                            "historical%s" % letter]
                if aaaa == 0:
                    lakess_data = lake_data
                    aaaa += 1
                else:
                    lakess_data = lakess_data.append(lake_data, ignore_index=True)

                for rcp in ["85"]:
                    model_code = {1: 'KNM',
                                  2: 'DMI',
                                  3: 'MPI',
                                  4: 'MOH',
                                  5: 'IPS',
                                  6: 'CNR'}
                    model_data.append([model_code.get(modelid), lake,volume,depth, "rcp%s" % rcp, data_summary["diff%sT" % rcp],
                                       data_summary["diff%sO" % rcp], data_summary["diff%sI" % rcp]])

    #
        except:
            print("model %s doesnt exist" % (m1 + m2))
    headers = model_data.pop(0)
    final_data = pd.DataFrame(model_data, columns=headers)
    lakesss_data.to_csv("annually_average_T_Ice_cover_Oxygen_originall.csv")
    final_data.to_csv("annually_average_T_Ice_cover_Oxygen.csv")
    plotT = sns.catplot(x="model", y=variables[0][1], col="scenario", data=final_data, kind="violin",color="orange", split=True,
                        height=8, aspect=.8)
    plotT.savefig("violinT.png")
    print("save T")
    plotO = sns.catplot(x="model", y=variables[1][1], col="scenario", data=final_data, kind="violin",color="red", split=True,
                        height=8, aspect=.9)
    plotO.savefig("violinO.png")
    print("save O")
    plotI = sns.catplot(x="model", y=variables[2][1], col="scenario", data=final_data, kind="violin",color="forestgreen", split=True,
                        height=8, aspect=.7)
    plotI.savefig("violinI.png")
    print("save Ice")
    print("end violin")


def data_descriptif():
    xlxs_datat = r"C:\Users\macot620\Documents\GitHub\Fish_niche\sweden_inflow_data\Validation_data_for_lookup12.xlsx"
    lakes12 = pd.read_csv("2017SwedenList_only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_id = list(lakes12.set_index("subid").to_dict().get("lake_id").values())
    data_lakes = pd.DataFrame()
    for lake_id in lakes_id:
        obs_file = pd.read_excel(xlxs_datat, sheet_name="%s"%lake_id)
        obs_file["date"] = pd.to_datetime(obs_file["date"])
        for variable in ["O2(mg/l)","Water temp (°C)","Siktdjup (m)"]:
            y = np.array(obs_file["depth(max)"])
            depths = np.unique(y)
            x = np.array(obs_file["date"])
            dates = np.unique(x)
            obs_file1 = obs_file.set_index("date")
            data_final = pd.DataFrame(index=dates,columns=depths)
            for depth in depths:
                selected = obs_file1.loc[obs_file1['depth(max)'] == depth]
                selected = selected[variable]
                test = selected.index
                for date in list(test):
                    try:

                        data_final.loc[date, depth] = selected[date].mean()

                    except:
                        print("here")

                #result = pd.concat([data_final, selected], axis=1).reindex(data_final.index)
                #data_final.loc[variable] = selected

            #data = obs_file.pivot(index="date", columns="depth(max)", values="%s"%variable)
            numbers = data_final.count()
            max = depths[-1]
            aa = numbers[:max/2]
            epi = numbers[:max/2].idxmax()
            if numbers.loc[depths[0]] > (epi-5):
                epi = depths[0]
            bb =  numbers[max/2:]
            hypo = numbers[max/2:].idxmax()
            if numbers.loc[depths[-1]] > (hypo-5):
                hypo = depths[-1]
            if variable == "Siktdjup (m)":
                data_final1 = data_final[[epi]]
            else:
                data_final1 = data_final[[epi,hypo]]
            ax = data_final1.plot(lw=2, colormap='jet', style='o--', title='lake %s'%lake_id,figsize=(20,5))
            ax.set_xlabel("Date")
            if variable == "O2(mg/l)":
                ax.set_ylabel("Oxygen Concentration (mg/l)")
                vari = "O2"
            elif variable == "Water temp (°C)":
                ax.set_ylabel("Water Temperature (°C)")
                vari = "Temp"
            else:
                ax.set_ylabel("Secchi Depth (m)")
                vari="Secchi"

            ax.legend(["%s m"%epi, "%s m"%hypo]);
            fig = ax.get_figure()
            fig.savefig("data_descriptif_%s_%s"%(lake_id,vari))


if __name__ == "__main__":
    #FishNiche_mean_secchi_graph()
    #Area_at_depth()
    #FishNiche_TO_graph()
    summary_characteristics_lake()
    #data_descriptif()
    #run_calibrations(11)
    #calibration_iteration()
    #calibration_iteration()
    #violin_plot()
    #violin_parallel()
    # lakes_list = ["Langtjern"]
    # input_files_parallel()
    # calibration_parallel()
    # mylake_parallel()
    # format_parallel()

    #calibration_iteration()
    #violin_plot()
    #violin_parallel()
    # lakes_list = ["Langtjern"]
    # input_files_parallel()
    # calibration_parallel()
    # mylake_parallel()
    # format_parallel()
