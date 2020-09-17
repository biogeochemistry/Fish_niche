#!/usr/bin/env python

""" Script for MyLake - ISIMIP
The main script launching the calibration and run of each lake with MyLake model.
"""

__author__ = "Julien Bellavance and Marianne Cote"

import multiprocessing as mp
import os
import shutil
from datetime import datetime, timedelta

import pandas as pd
import pysftp
from joblib import Parallel, delayed
from netCDF4 import Dataset, date2num
from numpy import arange, nan, reshape, sqrt
import seaborn as sns
from calibration_fish_niche import CalibrationInfo

#from lake_information import LakeInfo, outputfile, simulation_years


num_cores = mp.cpu_count() - 1
output_path = r"C:\Users\macot620\Documents\GitHub\Fish_niche\output"
input_folder = r"C:\Users\macot620\Documents\GitHub\Fish_niche\sweden_inflow_data\Validation_data_for_lookup.csv"
forcing_data_path = r"F:\cordex"
#output_path = r"F:\output-21-08-2018"

lakes_data = pd.read_csv("2017SwedenList.csv",encoding='ISO-8859-1')
lakes_data = lakes_data.set_index("lake_id").to_dict()
lakes_list = list(lakes_data.get("name").keys())

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
output_variables = ["Tzt.csv", "O2zt.csv", "lambdazt.csv", "DOCzt.csv", "Qst.csv", "Attn_zt.csv","PARzt.csv", "His.csv"]

output_unit = {"Tzt.csv":"deg C", "O2zt.csv":"mg m-2", "lambdazt.csv":"m-1", "DOCzt.csv":"mg m-3",
               "Qst.csv":"W m-2", "Attn_zt.csv":"-","PARzt.csv":"umol m-2 s-1",
               "His.csv": {"Hi":"m", "Hs":"m", "Hsi":"m", "Tice":"deg C", "Tair":"deg C", "rho_snow":"kg m-3", "IceIndicator":"-"}}

output_long = {"Tzt.csv":"Predicted temperature profile", "O2zt.csv":"Predicted dissolved oxygen profile",
               "lambdazt.csv":"Predicted average total light attenuation coefficient down to depth z",
               "DOCzt.csv":"Predicted dissolved organic carbon (DOC) profile", "Qst.csv":"Estimated surface heat fluxes",
               "Attn_zt.csv":"","PARzt.csv":"Predicted irradiance down to depth z",
               "His.csv": {"Hi":"total ice thickness", "Hs":"snow thickness", "Hsi":"snow ice thickness",
                           "Tice":"ice surface temperature", "Tair": "surface temperature",
                           "rho_snow": "Initial snow density", "IceIndicator": "Indicator of ice cover"}}



def calibration_parallel():
    """
    Simple function to call a parallel calibration of all lakes.
    :return: None
    """
    print(num_cores)
    # for lake in lakes_list:
    #     run_calibrations(lake)
    Parallel(n_jobs=num_cores, verbose=10)(delayed(run_calibrations)(lake) for lake in range(1,len(lakes_list)))


def run_calibrations(lake_number):
    """
        Intermediary function to call Nelder-mead optimization function for a single lake.

        :param lake_name: Type string. The name of the lake to calibrate.
        :return: If Calibration_Complete file is found, returns None. Else, return the Nelder-mead optimization function
        from myLake_post module for the given lake.
    """
    lake_name = lakes_list[lake_number]
    print(lake_name)

    lake = CalibrationInfo(lake_name,list(lakes_data.get("lake_id").keys()),list(lakes_data.get("subid").keys()),
                           list(lakes_data.get("ebhex").keys()),list(lakes_data.get("area").keys()),
                           list(lakes_data.get("depth").keys()),list(lakes_data.get("longitude").keys()),
                           list(lakes_data.get("latitude").keys()),list(lakes_data.get("volume").keys()))

    lake.variables_by_depth()
    if os.path.exists(os.path.join(lake.output_path, "EWEMBI/historical/Calibration_Complete.txt")):
        print("Calibration for {} is already complete.\n".format(lake_name))
        return None

    elif not os.path.exists("{}/{}_EWEMBI_historical_input".format(lake.input_path, lake.prefix)):
        print("Calibration can't {} be completed. input file doesn't exist.\n".format(lake_name))
        return None
    elif not os.path.exists(os.path.join(lake.input_path, "{}_EWEMBI_historical_input".format(lake.prefix))) and \
            not os.path.exists(os.path.join(lake.input_path, "{}_init".format(lake.prefix))) and \
            not os.path.exists(os.path.join(lake.input_path, "{}_par".format(lake.prefix))):
        print("not all initial files existing for %s" % lake_name)
    elif os.path.exists(os.path.join(lake.output_path, "EWEMBI/historical/Calibration_problem.txt")):
        print("Unable to calibration {}.\n".format(lake_name))
        return None

    else:
        lake.generate_input_files()
        return lake.optimize_differential_evolution("EWEMBI", "historical")

def input_files_parallel():
    """
    Simple function to call a parallel initiation of the input file of all lakes.
    :return: None
    """
    print("start")
    # for lake in lakes_list:
    #    input_files_loop(lake)
    Parallel(n_jobs=num_cores, verbose=10)(delayed(input_files_loop)(lake) for lake in lakes_list)


def input_files_loop(lake_name):
    """
    Create init, input and par file for all combinaison of lakes, models and scenarios.
    :param lake_name: Type string. The name of the lake.
    :return: None
    """
    lake = LakeInfo(lake_name)
    try:
        print("download for %s" % lake_name)
        download_forcing_data(lake_name)
        print('download for %s completed' % lake_name)
    except:
        print('unable to download of %s' % lake_name)

    try:
        print("{}/hurs_EWEMBI_historical_{}.allTS.nc".format(forcing_data_path, lake_name))

        if lake.region is None:
            print("Cannot find {}'s region".format(lake_name))
            return None
        else:
            print("{}/{}_hypsometry_modified.csv".format(lake.observation_path, lake_name))

        lake.initiate_init().init_file()
        for model in models:
            for scenario in scenarios:
                if not os.path.exists("{}/{}/{}".format(lake.output_path, "EWEMBI", "historical")):
                    os.makedirs("{}/{}/{}".format(lake.output_path, "EWEMBI", "historical"))
                if not os.path.exists("/{}/EWEMBI/historical/Calibration_Complete.txt".format(lake.output_path)):
                    try:
                        if (model == "EWEMBI" and scenario == "historical") or model != "EWEMBI":
                            if not (
                            os.path.exists(r"{}\{}_{}_{}_input".format(lake.input_path, lake.prefix, model, scenario))) \
                                    or not (os.path.exists(r"{}\{}_par".format(lake.input_path, lake.prefix))) \
                                    or not (os.path.exists(r"{}\{}_init".format(lake.input_path, lake.prefix))):
                                lake.initiate_input(model, scenario).mylakeinput()

                                if not (os.path.exists(
                                        r"{}/{}/{}/Observed_Temperatures.csv".format(lake.output_path, "EWEMBI",
                                                                                     "historical"))):
                                    if os.path.exists(r"{}\{}_temp_daily.csv".format(lake.observation_path, lake_name)):
                                        CalibrationInfo(lake_name).temperatures_by_depth()
                                        print("obsertvation done")
                                    else:
                                        print('no daily data for %s' % lake_name)
                            else:
                                print('Already done')
                                if not (os.path.exists(
                                        r"{}/{}/{}/Observed_Temperatures.csv".format(lake.output_path, "EWEMBI",
                                                                                     "historical"))):
                                    if os.path.exists(r"{}\{}_temp_daily.csv".format(lake.observation_path, lake_name)):
                                        CalibrationInfo(lake_name).temperatures_by_depth()
                                        print("obsertvation done")
                                    else:
                                        print('no daily data for %s' % lake_name)
                    except:
                        print("Issue when generating the input files of {} {} {}".format(lake_name, model, scenario))

                    if model == "EWEMBI":
                        if scenario == "historical":
                            if not (
                            os.path.exists(r"{}\{}_{}_{}_input".format(lake.input_path, lake.prefix, model, scenario))) \
                                    or not (os.path.exists(r"{}\{}_init".format(lake.input_path, lake.prefix))):
                                print("not all initial files existing for %s %s %s" % (model, scenario, lake))
                    else:
                        if not (
                        os.path.exists(r"{}\{}_{}_{}_input".format(lake.input_path, lake.prefix, model, scenario))) \
                                or not (os.path.exists(r"{}\{}_par".format(lake.input_path, lake.prefix))) \
                                or not (os.path.exists(r"{}\{}_init".format(lake.input_path, lake.prefix))):
                            print("not all initial files existing for %s %s %s" % (model, scenario, lake))

    except:
        print("missing file climatic probably or others...")


def download_forcing_data(lake_name):
    """
    A function to download forcing data from dkrz server.
    :param lake_name: Type string. The name of the lake.
    :return: None
    """
    countto = 0
    for model in models:
        for scenario in scenarios:
            if not os.path.exists(os.path.join(
                    r"{}/{}/Complete_Download_{}_{}_{}.txt".format(forcing_data_path, lake_name, model, scenario,
                                                                                 lake_name))):
                done = 0
                if model == "EWEMBI":
                    if scenario == "historical":

                        with pysftp.Connection('mistralpp.dkrz.de', username='b380750', password='TwopFaP5') as sftp:
                            sftp.cwd(
                                "/mnt/lustre01/work/bb0820/ISIMIP/ISIMIP2b/InputData/OBS_atmosphere/local_lakes/EWEMBI/historical")
                            print(sftp.listdir())
                            for var in input_variables:

                                if not os.path.exists(r"{}/{}_EWEMBI_historical_{}.allTS.nc".format(forcing_data_path,
                                                                                                    var, lake_name)):
                                    print("start scenario EWE histo")
                                    try:
                                        sftp.get(
                                            "{}/{}_EWEMBI_historical_{}.allTS.nc".format(lake_name, var, lake_name),
                                            localpath=r"{}/{}_EWEMBI_historical_{}.allTS.nc".format(forcing_data_path,
                                                                                                    var, lake_name))
                                        print("end")
                                        done += 1

                                    except:
                                        print("enable to get {}/{}_EWEMBI_historical_{}.allTS.nc".format(lake_name, var,
                                                                                                         lake_name))
                                else:
                                    done += 1

                                    print('download already done %s \n' % lake_name)

                else:

                    with pysftp.Connection('mistralpp.dkrz.de', username='b380750', password='TwopFaP5') as sftp:
                        sftp.cwd(
                            "/mnt/lustre01/work/bb0820/ISIMIP/ISIMIP2b/InputData/GCM_atmosphere/biascorrected/local_lakes")
                        print(sftp.listdir())
                        for var in input_variables:
                            print(done)
                            if not os.path.exists(
                                    "D:/forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, lake_name)):
                                print("start scenario %s" % scenario)
                                try:
                                    sftp.get(
                                        "{}/{}_{}_{}_{}.allTS.nc".format(lake_name, var, model, scenario, lake_name),
                                        localpath="D:/forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario,
                                                                                                 lake_name))
                                    print("end")
                                    done += 1
                                except:
                                    print(
                                        "enable to get {}/{}_{}_{}_{}.allTS.nc".format(lake_name, var, model, scenario,
                                                                                       lake_name))
                            else:
                                done += 1

                                print('download already done %s \n' % lake_name)

                if done == 6:
                    outdirl = os.path.join("D:/forcing_data\\{}".format(lake_name))
                    if not os.path.exists(outdirl):
                        os.makedirs(outdirl)
                    with open(os.path.join(
                            "{}\\Complete_Download_{}_{}_{}.txt".format(outdirl, model, scenario, lake_name)),
                              'w') as f:
                        f.write("Done")
                    print("Done")
                    countto += 1
            else:
                countto += 1
    if countto == 21:
        outdirl = os.path.join("D:/forcing_data\\{}".format(lake_name))
        with open(os.path.join("{}\\Complete_Download_all_{}.txt".format(outdirl, lake_name)), 'w') as f:
            f.write("Done")
            print("Done")


def mylake_parallel():
    """
    Simple function to call a parallel run of all lakes.
    :return: None
    """
    # for lake in lakes_list:
    #    model_scenario_loop(lake)

    Parallel(n_jobs=num_cores, verbose=10)(delayed(model_scenario_loop)(lake) for lake in full_lake_list)


def model_scenario_loop(lake_name):
    """
    Run MyLake model for all combinations of models and scenarios for a lake.

    :param lake_name: Type string. The name of the lake to calibrate.
    :return:
    """

    lake = LakeInfo(lake_name)

    if os.path.exists("D:\output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(lake.region, lake_name)):

        for model in models:
            if model == "EWEMBI":
                for scenario in scenarios:
                    lake.generate_input_files(model, scenario)
                    print(lake.input_path + "/{}_{}_{}_input".format(lake.prefix, model, scenario))
                    if os.path.exists(
                            "D:\output/{}/{}/{}/{}/RunComplete1".format(lake.region, lake_name, model, scenario)):
                        print("{} {} {} Run is already completed.\n".format(lake_name, model, scenario))

                    elif os.path.exists("D:\output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(lake.region,
                                                                                                            lake_name)):
                        if os.path.exists(lake.input_path + "/{}_{}_{}_input".format(lake.prefix, model, scenario)):
                            #try:
                                print("start")
                                lake.run_mylake(model, scenario)
                                print("Run of {} {} {} Completed.\n".format(lake, model, scenario))
                            # except:
                            #     print("problem with {} {} {}.\n".format(lake, model, scenario))
                        else:
                            print("input doesnt exist")
                    else:
                        print("{} Calibration have not been done.\n".format(lake))


def format_parallel():
    """
    Simple function to call a parallel formatting of all lakes files.
    :return: None
    """
    # for lake in lakes_list:
    #     format_txt(lake)

    Parallel(n_jobs=num_cores, verbose=10)(delayed(format)(lake) for lake in lakes_list)


def format_txt(lake_name):
    """
    Format the data creating from the run of the model
    :param lake_name: Type string. The name of the lake to calibrate.
    :return:
    """

    index = range(0, len(lakes_list) * 21 * 10)
    columns = ['lake', 'model', 'scenario', 'Calibration', "Date"] + [vari[:-4] for vari in output_variables]
    tableau = pd.DataFrame(index=index, columns=columns)
    index = 0

    lake = LakeInfo(lake_name)
    if lake.region is not None:
        for modelid in models:
            for scenarioid in scenarios:
                if modelid != "EWEMBI":
                    print(lake_name, modelid, scenarioid)

                    if os.path.exists(os.path.join(lake.output_path, "%s/%s" % (modelid, scenarioid))):
                        if os.path.exists(
                                os.path.join(lake.output_path, "%s/%s/RunComplete1" % (modelid, scenarioid))):

                            if modelid == "EWEMBI":
                                y11, y21 = [1979], [2016]
                            elif modelid == "GFDL-ESM2M" and scenarioid == 'piControl':
                                y11, y21 = [1661, 1861, 2006], [1860, 2005, 2099]
                            elif modelid == "GFDL-ESM2M" and scenarioid == 'rcp26':
                                y11, y21 = [2006], [2099]
                            elif modelid == "IPSL-CM5A-LR" and scenarioid == 'rcp85':
                                y11, y21 = [2006], [2099]
                            else:
                                y11, y21 = simulation_years(modelid, scenarioid)

                            subindex = index + 1
                            nbrfile = len(y11)
                            for vari in output_variables:
                                for i in range(0, nbrfile):
                                    y1, y2 = y11[i], y21[i]
                                    variable = vari[:-4]
                                    model_name = "MyLake"
                                    bias = modelid
                                    climate = scenarioid
                                    socio = "nosoc"
                                    sens = "co2"
                                    region = "local"
                                    timestep = "daily"
                                    unit = output_unit.get(vari)

                                    file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                        model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                        y2)

                                    file_namel = file_name2.lower()

                                    if not (os.path.exists(os.path.join(r"D:\ready", "%s.nc4" % file_namel))) and not (
                                            os.path.exists(os.path.join(r"D:\ready", "%s.nc" % file_namel))) and \
                                            not (
                                                    os.path.exists(
                                                        os.path.join(r"D:\move", "%s.nc" % file_namel))) and not (
                                            os.path.exists(os.path.join(r"D:\move", "%s.nc4" % file_namel))) and \
                                            not (
                                                    os.path.exists(
                                                        os.path.join(r"D:\remove", "%s.nc" % file_namel))) and not (
                                            os.path.exists(os.path.join(r"D:\remove", "%s.nc4" % file_namel))):
                                        if not (os.path.exists(os.path.join("%s.nc4" % file_namel))) and not (
                                                os.path.exists(os.path.join("%s.nc" % file_namel))):
                                            if y21[-1] - y11[0] > 100 or modelid == "IPSL-CM5A-LR":

                                                data_set = pd.DataFrame()
                                                yinit1, yend1 = [], []
                                                if y1 == 1661:
                                                    yinit1 = [1661, 1761]
                                                    yend1 = [1760, 1860]
                                                elif y1 == 1861:
                                                    if y21[-1] == 2005:
                                                        yinit1 = [1861, 1961]
                                                        yend1 = [1960, 2005]
                                                    else:
                                                        yinit1 = [1861, 1961]
                                                        yend1 = [1960, 2060]
                                                elif y1 == 2006:
                                                    if y21[0] == 2099:
                                                        if i != 0:
                                                            yinit1 = [1961, 2061]
                                                            yend1 = [2060, 2099]
                                                        else:
                                                            yinit1 = [2006, 2011]
                                                            yend1 = [2010, 2110]

                                                    elif y21[-1] == 2299:
                                                        if y11[0] == 2006:
                                                            yinit1 = [2006, 2011]
                                                            yend1 = [2010, 2110]
                                                        elif y11[0] == 1661:
                                                            yinit1 = [1961, 2061]
                                                            yend1 = [2060, 2160]
                                                elif y1 == 2100:
                                                    if y11[0] == 2006:
                                                        yinit1 = [2011, 2111, 2211]
                                                        yend1 = [2110, 2210, 2299]
                                                    elif y11[0] == 1661:
                                                        yinit1 = [2061, 2161, 2261]
                                                        yend1 = [2160, 2260, 2299]

                                                for j in range(0, len(yinit1)):
                                                    yinit = yinit1[j]
                                                    yend = yend1[j]
                                                    outputdir = os.path.join(lake.output_path, "/%s/%s/%s_%s" % (
                                                        modelid, scenarioid, yinit, yend))

                                                    if not os.path.exists(
                                                            os.path.join(outputdir, vari)) and os.path.exists(
                                                        os.path.join(outputdir, 'Tzt.csv')):
                                                        outputfile(yinit, yend, outputdir)

                                                    ###Main program###
                                                    print(os.path.join(outputdir, vari))
                                                    if os.path.exists(os.path.join(outputdir, vari)):

                                                        y3, y4 = yinit, yend

                                                        tableau.loc[subindex, 'lake'] = lake_name
                                                        tableau.loc[subindex, 'model'] = modelid
                                                        tableau.loc[subindex, 'scenario'] = scenarioid
                                                        tableau.loc[subindex, 'Calibration'] = "Done"
                                                        tableau.loc[subindex, 'Date'] = "%s_%s" % (y3, y4)

                                                        file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                            model_name, bias, climate, socio, sens, variable, lake_name,
                                                            timestep, y1, y2)

                                                        if not (os.path.exists(
                                                                os.path.join("%s.nc4" % file_name2.lower()))) or not (
                                                                os.path.exists(
                                                                    os.path.join("%s.nc" % file_name2.lower()))):

                                                            data = pd.read_csv(os.path.join(outputdir, vari),
                                                                               header=None)

                                                            if data.iloc[0, 0] != '%s-01-01, 00:00:00' % y3:
                                                                r = range((datetime(y4, 12, 31) - datetime(y3, 1,
                                                                                                           1)).days + 1)
                                                                dates = [datetime(y3, 1, 1) + timedelta(days=x) for x in
                                                                         r]
                                                                data[0] = pd.Series(dates)
                                                            if y3 >= y1:
                                                                if y4 <= y2:
                                                                    data = data.set_index(data[0])
                                                                    if y3 == y1:
                                                                        data_set = data[0:]
                                                                    else:
                                                                        data_set = data_set.append(data[0:],
                                                                                                   ignore_index=True)
                                                                        data_set = data_set.set_index(data_set[0])

                                                                else:
                                                                    data = data.set_index(data[0])
                                                                    data = data.loc[:'%s-01-01' % (y2 + 1)]
                                                                    data_set = data_set.append(data, ignore_index=True)
                                                                    data_set = data_set.set_index(data_set[0])
                                                            else:
                                                                data = data.set_index(data[0])
                                                                data_set = data.loc['%s-01-01' % y1:]




                                                    else:
                                                        print(os.path.join(outputdir, vari, " doesn't exist"))
                                                        tableau.loc[index, 'lake'] = lake_name
                                                        tableau.loc[index, 'model'] = modelid
                                                        tableau.loc[index, 'scenario'] = scenarioid
                                                        tableau.loc[index, 'Calibration'] = "Done"
                                                        tableau.loc[index, vari[:-4]] = "csv is missing"
                                                        break


                                            else:
                                                data_set = pd.DataFrame()

                                                outputdir = os.path.join(lake.output_path, "%s/%s" % (
                                                    modelid, scenarioid))

                                                if not os.path.exists(os.path.join(outputdir, vari)) and os.path.exists(
                                                        os.path.join(outputdir, 'Tzt.csv')):
                                                    outputfile(y1, y2, outputdir)

                                                ###Main program###
                                                if os.path.exists(os.path.join(outputdir, vari)):

                                                    file_name2 = "%s_%sfile_name2%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                        model_name, bias, climate, socio, sens, variable, lake_name,
                                                        timestep, y1, y2)

                                                    file_namel = file_name2.lower()

                                                    if not (
                                                            os.path.exists(
                                                                os.path.join("%s.nc4" % file_namel))) or not (
                                                            os.path.exists(os.path.join("%s.nc" % file_namel))):
                                                        data = pd.read_csv(os.path.join(outputdir, vari), header=None)
                                                        data = data.fillna(method='ffill')
                                                        y3, y4 = y1, y2

                                                        tableau.loc[subindex, 'lake'] = lake_name
                                                        tableau.loc[subindex, 'model'] = modelid
                                                        tableau.loc[subindex, 'scenario'] = scenarioid
                                                        tableau.loc[subindex, 'Calibration'] = "Done"
                                                        tableau.loc[subindex, 'Date'] = "%s_%s" % (y3, y4)

                                                        data = data.set_index(data[0])
                                                        data_set = data_set.append(data[0:])


                                                else:
                                                    print(os.path.join(outputdir, vari, " doesn't exist"))
                                                    tableau.loc[index, 'lake'] = lake_name
                                                    tableau.loc[index, 'model'] = modelid
                                                    tableau.loc[index, 'scenario'] = scenarioid
                                                    tableau.loc[index, 'Calibration'] = "Done"
                                                    tableau.loc[index, vari[:-4]] = "csv is missing"
                                                    break

                                            file_name = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                model_name, bias, climate, socio, sens,
                                                variable, region, timestep, y1, y2)

                                            if variable == "lakeicefrac":
                                                variable = "icetick"
                                            file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                                y2)

                                            file_namel = file_name2.lower()

                                            if not os.path.exists(os.path.join(lake.output_path, "%s.txt" % file_name)):

                                                if not os.path.exists(lake.output_path):
                                                    os.makedirs(lake.output_path)
                                                if len(data_set) != 0:
                                                    if len(data_set) == (
                                                            (datetime(y2, 12, 31) - datetime(y1, 1, 1)).days + 1):
                                                        data_set.to_csv(
                                                            os.path.join(lake.output_path, "%s.txt" % file_name),
                                                            header=None, index=None, sep=' ', mode='w')
                                                    else:
                                                        print('problem!')
                                                        # x = 5 / 0

                                                else:
                                                    print("data is empty!!!")
                                                    # x=5/0

                                                tableau.loc[subindex, vari[:-4]] = "Done"

                                            if os.path.exists(os.path.join(lake.output_path, "%s.txt" % file_name)):
                                                if (not os.path.exists(os.path.join(r"D:\ready", "%s.nc4" % (
                                                        file_namel))) and not os.path.exists(
                                                    os.path.join(r"D:\ready", "%s.nc" % file_namel))) and \
                                                        (not os.path.exists(os.path.join(r"D:\final_files", "%s.nc4" % (
                                                                file_namel))) and not os.path.exists(
                                                            os.path.join(r"D:\final_files",
                                                                         "%s.nc" % file_namel))) and \
                                                        (not os.path.exists(os.path.join(r"D:\move",
                                                                                         "%s.nc" % file_namel)) and not os.path.exists(
                                                            os.path.join(r"D:\move", "%s.nc4" % file_namel))) and \
                                                        (not os.path.exists(os.path.join(r"D:\remove",
                                                                                         "%s.nc" % file_namel)) and not os.path.exists(
                                                            os.path.join(r"D:\remove", "%s.nc4" % file_namel))):

                                                    creation = netcdf(lake.output_path, "%s.txt" % file_name,
                                                                      "%s.nc4" % file_namel, y1, y2, lake.latitude,
                                                                      lake.longitude, unit, variable)

                                                    if creation == 1:

                                                        if os.path.exists(
                                                                os.path.join(lake.output_path, "%s.txt" % file_name)):
                                                            os.remove(
                                                                os.path.join(lake.output_path, "%s.txt" % file_name))
                                                        if os.path.exists(
                                                                os.path.join(lake.output_path, "%s.txt" % file_name)):
                                                            os.remove(
                                                                os.path.join(lake.output_path, "%s.txt" % file_name))

                                                    if (os.path.exists(os.path.join(r"D:\ready", "%s.nc4" % (
                                                            file_namel))) or os.path.exists(
                                                        os.path.join(r"D:\ready", "%s.nc" % file_namel))) \
                                                            or (os.path.exists(os.path.join(r"D:\final_files",
                                                                                            "%s.nc4" % (
                                                                                                    file_namel))) or os.path.exists(
                                                        os.path.join(r"D:\final_files", "%s.nc" % file_namel))) \
                                                            or (os.path.exists(os.path.join(r"D:\move",
                                                                                            "%s.nc" % file_namel)) or os.path.exists(
                                                        os.path.join(r"D:\move", "%s.nc4" % file_namel))) \
                                                            or (os.path.exists(
                                                        os.path.join("%s.nc" % file_namel)) or os.path.exists(
                                                        os.path.join("%s.nc4" % file_namel))) \
                                                            or (os.path.exists(os.path.join(r"D:\remove",
                                                                                            "%s.nc" % file_namel)) or os.path.exists(
                                                        os.path.join(r"D:\remove", "%s.nc4" % file_namel))):
                                                        if os.path.exists(
                                                                os.path.join(lake.output_path, "%s.txt" % file_name)):
                                                            os.remove(
                                                                os.path.join(lake.output_path, "%s.txt" % file_name))
                                                        if os.path.exists(
                                                                os.path.join(lake.output_path, "%s.txt" % file_name)):
                                                            os.remove(
                                                                os.path.join(lake.output_path, "%s.txt" % file_name))


                                            else:
                                                print("file exists")
                                                tableau.loc[subindex, vari[:-4]] = "Done"

                                            subindex += 1

                            index = subindex

                        else:
                            tableau.loc[index, 'lake'] = lake_name
                            tableau.loc[index, 'model'] = modelid
                            tableau.loc[index, 'scenario'] = scenarioid
                            tableau.loc[index, 'Calibration'] = "not Done"
                            index += 1
                    else:
                        print(os.path.join(lake.output_path, "%s/%s doesn't exist" % (modelid, scenarioid)))
                        tableau.loc[index, 'lake'] = lake_name
                        tableau.loc[index, 'model'] = modelid
                        tableau.loc[index, 'scenario'] = scenarioid
                        tableau.loc[index, 'Calibration'] = "folder is missing"
                        index += 1

    else:
        print("Lake is not in regions")
    tableau.to_csv(r"all_variable_lakes_combinaisonfinall.csv", index=False)
    format_nc41(lake_name)


def format_nc4_par():
    """
    Simple function to call a parallel formatting of all lakes files.
    :return: None
    """
    # for lake_name in lakes_list:
    #    format_nc41(lake_name)
    Parallel(n_jobs=num_cores, verbose=10)(delayed(format_nc41)(lake) for lake in lakes_list)


def format_nc41(lake_name):
    """

    :param lake_name:
    :return:
    """
    index = range(0, len(lakes_list) * 21 * 10)
    columns = ['lake', 'model', 'scenario', 'date', 'Calibration'] + ['%s.txt' % vari[:-4] for vari in
                                                                      output_variables] + [
                  '%s.nc4' % vari[:-4] for vari in output_variables]
    tableau = pd.DataFrame(index=index, columns=columns)

    tableau2 = [
        ['lake', 'model', 'scenario', 'variable', "file", "levlak_min", "levlak_max", 'lat', 'lon', 'time_start',
         'time_end', 'variable_min', "variable_max"]]

    index = 0
    print(lake_name)
    lake = LakeInfo(lake_name)

    outputnc = r"D:\final_files"

    for modelid in models:
        for scenarioid in scenarios:

            print(lake_name, modelid, scenarioid)
            # path = os.path.join(output, "%s/%s/%s/%s/RunComplete" % (reg, lake, modelid, scenarioid))
            if os.path.exists(os.path.join(lake.output_path, "%s/%s/RunComplete1" % (modelid, scenarioid))):
                tableau.loc[index, 'lake'] = lake_name
                tableau.loc[index, 'model'] = modelid
                tableau.loc[index, 'scenario'] = scenarioid

                tableau.loc[index, 'Calibration'] = 'Done'
                if (modelid == "EWEMBI" and scenarioid == "historical") or modelid != "EWEMBI":

                    if modelid == "EWEMBI":
                        y11, y21 = [1979], [2016]
                    elif modelid == "GFDL-ESM2M" and scenarioid == 'piControl':
                        y11, y21 = [1661, 1861, 2006], [1860, 2005, 2099]
                    elif modelid == "GFDL-ESM2M" and scenarioid == 'rcp26':
                        y11, y21 = [2006], [2099]
                    elif modelid == "IPSL-CM5A-LR" and scenarioid == 'rcp85':
                        y11, y21 = [2006], [2099]
                    else:
                        y11, y21 = simulation_years(modelid, scenarioid)

                    nbrfile = len(y11)
                    for vari in output_variables:
                        for i in range(0, nbrfile):
                            y1, y2 = y11[i], y21[i]
                            listtableau2 = []

                            variable = vari[:-4]
                            model_name = "MyLake"

                            bias = modelid
                            climate = scenarioid
                            socio = "nosoc"
                            sens = "co2"
                            region = "local"
                            timestep = "daily"

                            if bias == "HadGEM2-ES" and climate == "rcp26":
                                print("stop")
                            y3, y4 = y1, y2

                            file_name = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                model_name, bias, climate, socio, sens, variable, region, timestep, y3, y4)

                            file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                model_name, bias, climate, socio, sens, variable, lake, timestep, y3, y4)
                            if variable == "lakeicefrac":
                                file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                    model_name, bias, climate, socio, sens, "icetick", lake, timestep, y3, y4)

                            print(os.path.join(lake.output_path, file_name))
                            file_namel = file_name2.lower()
                            if bias == "HadGEM2-ES" and climate == "rcp26":
                                print("stop")

                            if os.path.exists(os.path.join(lake.output_path, "%s.txt" % file_name)) or os.path.exists(
                                    os.path.join(outputnc, "%s.nc4" % file_namel)) or os.path.exists(
                                    os.path.join(outputnc, "%s.nc" % file_namel)):
                                print("a;;p")
                                # command = comm.split(' ')
                                # if os.path.exists(os.path.join(output, "%s.txt" % file_name)):
                                #     tableau.loc[index, '%s.txt' % variable] = 'Exists'

                                # if not os.path.exists(os.path.join(outputnc, "%s.nc4" % (file_namel))) and not os.path.exists(os.path.join(outputnc, "%s.nc" % (file_namel))) :
                                # if 1==1:#try:
                                # creation = txt_to_netcdf.netcdf(output, "%s.txt"%( file_name), "%s.nc4"%file_namel, y3,y4, float(run_myLake_ISIMIP.get_latitude(lake, "D:/forcing_data", "EWEMBI",
                                #                             "historical")), float(run_myLake_ISIMIP.get_longitude(lake, "D:/forcing_data", "EWEMBI",
                                #                             "historical")),unit, variable)
                                # if creation == 1:
                                #     tableau.loc[index, '%s.nc4' % variable] = 'Exists'
                                #     if os.path.exists(os.path.join(output, "%s.txt" % file_name)):
                                #         os.remove(os.path.join(output, "%s.txt" % file_name))
                                # else:
                                #     tableau.loc[index, '%s.nc4' % variable] = 'Not created yet'

                                # except:
                                #     tableau.loc[index, '%s.nc4' % variable] = 'Bug Not created yet'

                                # print(command)
                                # commandall = commandall + command.lower() + " && "
                                # #os.system(command)

                                # else:
                                #     tableau.loc[index, '%s.txt' % variable] = 'doesnt Exist'
                                #     tableau.loc[index, '%s.nc4' % variable] = 'Exists'

                                # else:
                                #     tableau.loc[index, '%s.txt' % variable] = 'Not created'

                                # if os.path.exists(os.path.join(output, "%s.txt" % file_name)) and (os.path.exists(os.path.join(outputnc, "%s.nc4" % (file_namel))) or os.path.exists(os.path.join(outputnc, "%s.nc" % (file_namel)))):
                                #    os.remove(os.path.join(output, "%s.txt" % file_name))

                            if 1 == 1:  # os.path.exists(
                                # os.path.join(outputnc, "%s.nc4" % (file_namel))) or os.path.exists(
                                # os.path.join(outputnc, "%s.nc" % (file_namel))):

                                listtableau2.append(lake_name)
                                listtableau2.append(modelid)
                                listtableau2.append(scenarioid)
                                if variable == "lakeicefrac":
                                    variable = "icetick"
                                listtableau2.append(variable)
                                if os.path.exists("{}/{}_hypsometry_modified.csv".format(lake.observation_path, lake)):
                                    hypso = pd.read_csv(
                                        "{}/{}_hypsometry_modified.csv".format(lake.observation_path, lake))
                                    maxdepth = len(hypso)

                                    validationresult = validationnetcdf("%s.nc4" % file_namel, variable, lake.latitude,
                                                                        lake.longitude, maxdepth, y3, y4)
                                    for result in validationresult:
                                        listtableau2.append(result)
                                    tableau2.append(listtableau2)


            else:
                tableau.loc[index, 'lake'] = lake_name
                tableau.loc[index, 'model'] = modelid
                tableau.loc[index, 'scenario'] = scenarioid
                tableau.loc[index, 'Calibration'] = 'Not Done'

            index += 1

    if len(tableau2) != 1:
        df = pd.DataFrame.from_records(tableau2)
        df.to_csv("validation_list%s.csv" % lake, index=False)
    else:
        df = pd.DataFrame.from_records(tableau2)
        df.to_csv("validation_list_empty%s.csv" % lake, index=False)


def format_nc4():
    """

    :return:
    """
    index = range(0, len(lakes_list) * 21 * 10)
    columns = ['lake', 'model', 'date', 'scenario', 'Calibration'] + ['%s.nc4' % vari[:-4] for vari in output_variables]
    tableau = pd.DataFrame(index=index, columns=columns)

    index = 0

    for lake_num in lakes_list:
        lake = LakeInfo(lake_num)

        outputnc = r"D:\final_files"

        for modelid in models:
            if modelid != "EWEMBI":
                for scenarioid in scenarios:

                    print(lake_num, modelid, scenarioid)

                    if modelid == "EWEMBI":
                        y11, y21 = [1979], [2016]
                    elif modelid == "GFDL-ESM2M" and scenarioid == 'piControl':
                        y11, y21 = [1661, 1861, 2006], [1860, 2005, 2099]
                    elif modelid == "GFDL-ESM2M" and scenarioid == 'rcp26':
                        y11, y21 = [2006], [2099]
                    elif modelid == "IPSL-CM5A-LR" and scenarioid == 'rcp85':
                        y11, y21 = [2006], [2099]
                    else:
                        y11, y21 = simulation_years(modelid, scenarioid)

                    nbrfile = len(y11)

                    for i in range(0, nbrfile):
                        if os.path.exists(os.path.join(lake.output_path, "%s/%s/RunComplete1" % (modelid, scenarioid))):
                            tableau.loc[index, 'lake'] = lake_num
                            tableau.loc[index, 'model'] = modelid
                            tableau.loc[index, 'scenario'] = scenarioid

                            tableau.loc[index, 'Calibration'] = 'Done'
                        else:
                            tableau.loc[index, 'lake'] = lake_num
                            tableau.loc[index, 'model'] = modelid
                            tableau.loc[index, 'scenario'] = scenarioid
                            tableau.loc[index, 'Calibration'] = 'not Done'
                        y1, y2 = y11[i], y21[i]
                        tableau.loc[index, 'date'] = '%s_%s' % (y1, y2)
                        for vari in output_variables:

                            variable = vari[:-4]
                            model_name = "MyLake"
                            bias = modelid
                            climate = scenarioid
                            socio = "nosoc"
                            sens = "co2"
                            region = "local"
                            timestep = "daily"

                            file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                y2)

                            file_name = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                model_name, bias, climate, socio, sens, variable, region, timestep, y1, y2)

                            print(os.path.join(lake.output_path, file_name))
                            file_namel = file_name2.lower()

                            if os.path.exists(os.path.join(lake.output_path, "%s.txt" % file_name)):
                                # tableau.loc[index, '%s.txt' % variable] = 'Exists'

                                if os.path.exists(os.path.join(outputnc, "%s.nc4" % file_namel)):
                                    tableau.loc[index, '%s.nc4' % variable] = 'Exist'
                                elif os.path.exists(os.path.join(outputnc, "%s.nc" % file_namel)):
                                    tableau.loc[index, '%s.nc4' % variable] = 'Exist'
                                else:
                                    tableau.loc[index, '%s.nc4' % variable] = 'Not created yet'

                            else:
                                if os.path.exists(os.path.join(outputnc, "%s.nc4" % file_namel)):
                                    tableau.loc[index, '%s.nc4' % variable] = 'Exist'
                                elif os.path.exists(os.path.join(outputnc, "%s.nc" % file_namel)):
                                    tableau.loc[index, '%s.nc4' % variable] = 'Exist'
                                else:
                                    tableau.loc[index, '%s.nc4' % variable] = 'Not created yet'
                        index += 1

    tableau.to_csv(r"all_variable_lakes_combinaison_update2020.csv", index=False)


def netcdf(output, file_name_txt, file_name_nc, datestart, dateend, lats, lons, unit, variable):
    """

    :param output:
    :param file_name_txt:
    :param file_name_nc:
    :param datestart:
    :param dateend:
    :param lats:
    :param lons:
    :param unit:
    :param variable:
    :return:
    """
    try:
        file_name_txt = "%s/%s" % (output, file_name_txt)
        data = pd.read_csv(file_name_txt, sep=" ", header=None)
        print(data.loc[0][0])
        try:
            int(data.loc[0][0])
        except:
            data = data.drop([0], axis=1)
            nbrcol = len(data.columns)
            data.columns = [x for x in range(0, nbrcol)]

        data.round(-3)
        data = data.replace(nan, 1.e+20)
        data = data.replace('1.e+20f', 1.e+20)
        data_a = data.to_numpy()
        x = len(data)
        y = len(data.columns)

        file_name = Dataset(file_name_nc, 'w', format='NETCDF4')
        print(file_name.file_format)

        def walktree(top):
            values = top.groups.values()
            yield values
            for value in top.groups.values():
                for children in walktree(value):
                    yield children

        print(file_name)
        for children in walktree(file_name):
            for child in children:
                print(child)

        # dimensions.
        if y != 1:
            level = file_name.createDimension('levlak', y)
        time = file_name.createDimension('time', None)
        lat = file_name.createDimension('lat', 1)
        lon = file_name.createDimension('lon', 1)

        print(file_name.dimensions)

        print(time)

        if variable == 'lakeicefrac':
            variable = 'icetick'
        # variables.
        times = file_name.createVariable('time', 'f8', ('time',))
        if y != 1:
            levels = file_name.createVariable('levlak', 'f8', ('levlak',))
        latitudes = file_name.createVariable('lat', 'f4', ('lat',))
        longitudes = file_name.createVariable('lon', 'f4', ('lon',))
        if y != 1:
            temp = file_name.createVariable('%s' % variable, 'f4', ('time', 'levlak', 'lat', 'lon',), fill_value=1.e+20)
        else:
            temp = file_name.createVariable('%s' % variable, 'f4', ('time', 'lat', 'lon',), fill_value=1.e+20)
        print(temp)

        # attributes.
        # import time
        file_name.contact = "Raoul-Marie Couture <Raoul.Couture@chm.ulaval.ca>"
        file_name.institution = "Universite Laval (Ulaval)"
        file_name.comment = "Data prepared for ISIMIP2b"

        latitudes.longname = "latitude"
        # time.longname = "time"
        longitudes.longname = "longitude"
        temp.longname = output_long.get(variable)
        latitudes.units = 'degrees_north'
        longitudes.units = 'degrees_east'
        latitudes.axis = "Y"
        longitudes.axis = "X"

        if y != 1:
            levels.units = 'm'
            levels.axis = "Z"
            levels.longname = "depth_below_water_surface"
            levels.positive = "down"
        temp.units = unit
        temp.missing_value = 1.e+20
        times.units = 'days since 1661-01-01 00:00:00'

        times.calendar = 'proleptic_gregorian'

        for name in file_name.ncattrs():
            print('Global attr', name, '=', getattr(file_name, name))

        file_name.close()

        # create some groups.
        file_name = Dataset(file_name_nc, 'a')

        latitudes[:] = lats
        longitudes[:] = lons
        print('latitudes =\n', latitudes[:])
        print('longitudes =\n', longitudes[:])

        # append along two unlimited dimensions by assigning to slice.
        nlats = len(file_name.dimensions['lat'])
        nlons = len(file_name.dimensions['lon'])

        print('start')
        if y != 1:
            data_b = reshape(data_a, (x, y, 1, 1))
        else:
            data_b = reshape(data_a, (x, 1, 1))
        print(data_b.shape)
        print('temp shape before adding data = ', temp.shape)
        if y != 1:
            temp[0:x, 0:y, :, :] = data_b
        else:
            temp[0:x, :, :] = data_b
        print('temp shape after adding data = ', temp.shape)
        # levels have grown, but no values yet assigned.
        if y != 1:
            print('levels shape after adding pressure data = ', levels.shape)
            levels[:] = [x for x in arange(0, y, 1)]

        r = range((datetime(dateend, 12, 31) - datetime(datestart, 1, 1)).days + 1)
        print(r)
        dates = [datetime(datestart, 1, 1) + timedelta(days=x) for x in r]
        times[:] = date2num(dates, units=times.units, calendar=times.calendar)
        print('time values (in units %s): ' % times.units + '\\n', times[:])

        # dates = num2date(times[:],units=times.units,calendar=times.calendar)
        # print('dates corresponding to time values:\\n',dates)

        file_name.close()

        if y != 1:
            print(file_name_nc[:-1])
            # if not os.path.exists(os.path.join(file_name_nc[:-1])):
            #     os.system("ncks -h -7 -L 9 %s %s"%(file_name_nc,file_name_nc[:-1]))
            #     os.remove(file_name_nc)

        if os.path.exists(file_name_nc[:-1]) or os.path.exists(file_name_nc):
            os.remove(os.path.join(output, file_name_txt))

        finaloutput = r"D:\final_files"
        try:
            if os.path.exists(os.path.join(file_name_nc[:-1])):
                shutil.copyfile(os.path.join(file_name_nc[:-1]), os.path.join(finaloutput, file_name_nc[:-1]))
                os.remove(os.path.join(file_name_nc[:-1]))
            elif os.path.exists(os.path.join(file_name_nc)):
                shutil.copyfile(os.path.join(file_name_nc), os.path.join(finaloutput, file_name_nc))
                os.remove(os.path.join(file_name_nc))
        except:
            print("issue with moving file from isimip to final_files")
        return 1

    except:
        print("bug creation file file_name_nc")

        return 0


def validationnetcdf(file_name_nc, vari, lats, lons, maxdepth, datestart, dateend):
    """

    :param file_name_nc:
    :param vari:
    :param lats:
    :param lons:
    :param maxdepth:
    :param datestart:
    :param dateend:
    :return:
    """

    limit_variables = {"strat": [0, 1], "watertemp": [272, 320],
                       "thermodepth": [0, maxdepth], "ice": [0, 1],
                       "icetick": [0, 16], "snowtick": [0, 10],
                       "sensheatf": [-800, 10000], "latentheatf": [-10000, 2500],
                       "lakeheatf": [-3000, 10000], "albedo": [0.001, 1],
                       "turbdiffheat": [0, 50], "sedheatf": [-50, 50]}
    finaloutput = r"D:\final_files"
    readyoutput = r"D:\ready"

    stop = False

    if os.path.exists(os.path.join(finaloutput, file_name_nc[:-1])):
        netcdf_dataset = Dataset(os.path.join(finaloutput, file_name_nc[:-1]))
        # if not os.path.exists(os.path.join(finaloutput,file_name_nc[:-1])):
        #    shutil.copyfile(os.path.join(file_name_nc[:-1]),os.path.join(finaloutput,file_name_nc[:-1]))
    elif os.path.exists(os.path.join(finaloutput, file_name_nc)):
        netcdf_dataset = Dataset(os.path.join(finaloutput, file_name_nc))
        # if not os.path.exists(os.path.join(finaloutput, file_name_nc)):
        #    shutil.copyfile(os.path.join( file_name_nc), os.path.join(finaloutput, file_name_nc))
    else:
        stop = True

        if os.path.exists(os.path.join(readyoutput, file_name_nc[:-1])) or os.path.exists(
                os.path.join(readyoutput, file_name_nc)) or os.path.exists(
                os.path.join(r"D:\move", file_name_nc[:-1])) or os.path.exists(os.path.join(r"D:\move", file_name_nc)):
            return ["file exist", "ok", "ok", "ok", "ok", "ok"]
        else:
            return ["file does not exist", "", "", "", "", ""]

    if not stop:

        listinfor = {"file": "file exists"}

        if not "levlak" in netcdf_dataset.variables.keys():
            listinfor["levlak_min"] = " "
            listinfor["levlak_max"] = " "
        for variable in netcdf_dataset.variables.keys():
            var = netcdf_dataset.variables[variable]
            min_var = min(var)
            max_var = max(var)
            if variable == "levlak":
                if min_var == 0:
                    listinfor["levlak_min"] = "ok min levlak"
                else:
                    listinfor["levlak_min"] = "not_ok min levlak"
                    print('prblem min = %s and not 0' % min_var)
                if max_var == maxdepth or max_var == (maxdepth - 1) or max_var == (maxdepth - 2) or max_var == (
                        maxdepth - 3):
                    listinfor["levlak_max"] = "ok max levlak"
                else:
                    listinfor["levlak_max"] = "not_ok max levlak"
                    print('prblem max = %s and not %s +- 3' % max_var)
            elif variable == "lat":
                if len(var) == 1:
                    if min_var == lats:
                        listinfor["lat"] = "ok lat"
                    else:
                        listinfor["lat"] = "not_ok lat"
                else:
                    listinfor["lat"] = "not_ok long lat"
            elif variable == "lon":
                if len(var) == 1:
                    if min_var == lons:
                        listinfor["lon"] = "ok lon"
                    else:
                        listinfor["lon"] = "not_ok lon"
                else:
                    listinfor["lon"] = "not_ok long lon"

            elif variable == "time":
                print(len(var), (datetime(dateend, 12, 31) - datetime(datestart, 1, 1)).days + 1)
                if len(var) == ((datetime(dateend, 12, 31) - datetime(datestart, 1, 1)).days + 1):
                    first = date2num(datetime(datestart, 1, 1), units='days since 1661-01-01 00:00:00',
                                     calendar='proleptic_gregorian')
                    last = date2num(datetime(dateend, 12, 31), units='days since 1661-01-01 00:00:00',
                                    calendar='proleptic_gregorian')

                    var1 = netcdf_dataset.variables[vari]
                    min1 = min(var1)
                    max1 = max(var1)
                    print(min1, limit_variables.get(vari)[0], max1, limit_variables.get(vari)[1])
                    if min1 >= limit_variables.get(vari)[0] or min1 > 1e+19:
                        if max1 <= limit_variables.get(vari)[1] or max1 > 1e+19:
                            if os.path.exists(os.path.join(finaloutput, file_name_nc[:-1])):
                                if not os.path.exists(os.path.join(r"D:\ready", file_name_nc[:-1])):
                                    shutil.copyfile(os.path.join(finaloutput, file_name_nc[:-1]),
                                                    os.path.join(r"D:\ready", file_name_nc[:-1]))

                            elif os.path.exists(os.path.join(finaloutput, file_name_nc)):
                                if not os.path.exists(os.path.join(r"D:\ready", file_name_nc)):
                                    shutil.copyfile(os.path.join(finaloutput, file_name_nc),
                                                    os.path.join(r"D:\ready", file_name_nc))

                    if min_var == first:
                        listinfor["time_start"] = "ok start time"
                    else:
                        listinfor["time_start"] = "not_ok start time VALUE %s" % min_var

                    if max_var == last:
                        listinfor["time_end"] = "ok end time"
                    else:
                        listinfor["time_end"] = "not_ok end time VALUE %s" % max_var

                else:
                    first = date2num(datetime(datestart, 1, 1), units='days since 1661-01-01 00:00:00',
                                     calendar='proleptic_gregorian')
                    last = date2num(datetime(dateend, 12, 31), units='days since 1661-01-01 00:00:00',
                                    calendar='proleptic_gregorian')
                    # if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc))
                    #     if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #         shutil.copyfile(os.path.join(finaloutput, file_name_nc), os.path.join(file_name_nc))
                    if min_var == first:
                        listinfor["time_start"] = "ok start time"
                    else:
                        listinfor["time_start"] = "not_ok start time VALUE %s" % min_var

                    if max_var == last:
                        listinfor["time_end"] = "ok_errorlen end time"

                    else:
                        listinfor["time_end"] = "not_ok_errorlen end time VALUE %s" % max_var


            elif variable == vari:
                if min_var >= limit_variables.get(vari)[0] or min_var > 1e+19:
                    listinfor["variable_min"] = "ok min variable"
                else:
                    listinfor["variable_min"] = "not_ok min variable VALUE %s" % min_var
                    # if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc))
                    #     if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #         shutil.copyfile(os.path.join(finaloutput, file_name_nc), os.path.join( file_name_nc))

                if max_var <= limit_variables.get(vari)[1] or max_var > 1e+19:
                    listinfor["variable_max"] = "ok max variable"
                else:
                    listinfor["variable_max"] = "not_ok max variable VALUE %s" % max_var

                    # if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc))
                    #     if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #         shutil.copyfile(os.path.join(finaloutput, file_name_nc), os.path.join( file_name_nc))


            else:
                return ["file exist but unknow variable", "", "", "", "", ""]
        netcdf_dataset.close()
        if os.path.exists((os.path.join(r"D:\ready", file_name_nc[:-1]))):
            os.remove(os.path.join(finaloutput, file_name_nc[:-1]))
        elif os.path.exists((os.path.join(r"D:\ready", file_name_nc))):
            os.remove(os.path.join(finaloutput, file_name_nc))
        return [listinfor["file"], listinfor['levlak_min'], listinfor["levlak_max"], listinfor["lat"], listinfor["lon"],
                listinfor["time_start"], listinfor["time_end"], listinfor["variable_min"], listinfor["variable_max"]]


def summary_characteristics_lake():
    summary = pd.DataFrame(index=range(len(lakes_list)),columns=["lake_name","country","swa_b1","swa_b0","C_shelter","Kz_N0","alb_melt_ice","alb_melt_snow","longitude","latitude","max_depth","Area","Volume"])
    i=0
    for lake in lakes_list:

        lakeinfo = LakeInfo(lake).initiate_init()
        summary.loc[i,'lake_name'] = lakeinfo.lake_name
        summary.loc[i,"country"] = lakeinfo.region
        summary.loc[i, "swa_b1"] = lakeinfo.swa_b1
        summary.loc[i,"swa_b0"] = lakeinfo.swa_b0
        summary.loc[i, "C_shelter"] = lakeinfo.c_shelter
        summary.loc[i, "Kz_N0"] = lakeinfo.kz_n0
        summary.loc[i,"alb_melt_ice"] = lakeinfo.alb_melt_ice
        summary.loc[i,"alb_melt_snow"] = lakeinfo.alb_melt_snow
        summary.loc[i,"longitude"] = lakeinfo.longitude
        summary.loc[i,"latitude"] = lakeinfo.latitude
        summary.loc[i,"max_depth"] = lakeinfo.depth_levels[-1]
        summary.loc[i,"Area"] = lakeinfo.areas[0]
        summary.loc[i,"Volume"] = sum(lakeinfo.areas)
        i+=1

    summary.to_csv("summary_info_lake1.csv", index=False)



def calibration_matlab_script_parallel():
    """
        Simple function to call a parallel calibration of all lakes.
        :return: None
        """
    print(num_cores)
    # for lake in lakes_list:
    #     run_calibrations(lake)
    Parallel(n_jobs=num_cores, verbose=10)(delayed(calibration_matlab_script)(lake) for lake in lakes_list)


def calibration_matlab_script(lake_name):
    """
        Intermediary function to call Nelder-mead optimization function for a single lake.

        :param lake_name: Type string. The name of the lake to calibrate.
        :return: If Calibration_Complete file is found, returns None. Else, return the Nelder-mead optimization function
        from myLake_post module for the given lake.
    """
    print(lake_name)
    lake = CalibrationInfo(lake_name)

    if os.path.exists(os.path.join(lake.output_path, "EWEMBI/historical/Calibration_Complete.txt")):
        print("Calibration for {} is already complete.\n".format(lake_name))
        return None

    elif not os.path.exists("{}/{}_EWEMBI_historical_input".format(lake.input_path, lake.prefix)):
        print("Calibration can't {} be completed. input file doesn't exist.\n".format(lake_name))
        return None
    elif not os.path.exists(os.path.join(lake.input_path, "{}_EWEMBI_historical_input".format(lake.prefix))) and \
            not os.path.exists(os.path.join(lake.input_path, "{}_init".format(lake.prefix))) and \
            not os.path.exists(os.path.join(lake.input_path, "{}_par".format(lake.prefix))):
        print("not all initial files existing for %s" % lake_name)
    elif os.path.exists(os.path.join(lake.output_path, "EWEMBI/historical/Calibration_problem.txt")):
        print("Unable to calibration {}.\n".format(lake_name))
        return None

    else:
        lake.temperatures_by_depth()

        cmd = 'matlab -wait -r -nosplash -nodesktop MyLake_optimizer(%d,%d,\'%s\',\'%s\',\'%s\',\'%s\',%d,%d);' \
              'quit' % (lake.start_year, lake.end_year, os.path.join(lake.calibration_path, "2020_par"), os.path.join(lake.calibration_path, "2020_input"), os.path.join(lake.calibration_path, "2020_init"), lake.output_path, lake.latitude,
                        lake.longitude)
        print(cmd)

        os.system(cmd)

        lake.generate_input_files()
        return lake.optimize_differential_evolution("EWEMBI", "historical")


def violin_parallel():
    """
    Simple function to call a parallel calibration of all lakes.
    :return: None
    """
    print(num_cores)
    # for lake in lakes_list:
    #     run_calibrations(lake)
    Parallel(n_jobs=num_cores, verbose=10)(delayed(run_calibrations)(lake) for lake in lakes_list)

def violin_plot():
    sns.set_color_codes("colorblind")
    sns.set_context("paper",2.5)
    variables = [["Tzt.csv","Change in Surface Temperature ($^\circ$C)"],["O2zt.csv","Change in Bottom Oxygen\n Concentration (mg m-2)"],["His.csv","Change in Ice Cover Duration (day)"]]
    model_data = [["model","lake","scenario",variables[0][1],variables[1][1],variables[2][1]]]
    kernel = [["model","lake","scenario",variables[0][1],variables[1][1],variables[2][1]]]
    for modelid in [1,2,3,4,5,6]:
        m1, m2 = models[modelid]
        #if 1==1:
        try:

            lake_data = pd.DataFrame( columns=["historicalT","rcp45T","rcp85T","diff45T","diff85T",
                                                    "historicalO","rcp45O","rcp85O","diff45O","diff85O",
                                                    "historicalI","rcp45I","rcp85I","diff45I","diff85I"])
            n = 1
            for lake in lakes_list:
                print(m2,lake, n)
                n+=1
                eh = lakes_data.get('ebhex').get(lake)
                eh = eh[2:] if eh[:2] == '0x' else eh
                while len(eh) < 6:
                    eh = '0' + eh
                d1, d2, d3 = eh[:2], eh[:4], eh[:6]
                for scenarioid in [1,5,8]:
                    exA, y1A, exB, y1B = scenarios[scenarioid]
                    y2A = y1A + 4
                    y2B = y1B + 4
                    outdir = os.path.join(output_path, d1, d2, d3,
                                          'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (
                                              m1, exA, exB, m2, y1A, y2B))
                    for variable in [0,1,2]:
                        data = pd.read_csv(os.path.join(outdir,variables[variable][0]),header=None)
                        if variable == 0:
                            lake_data["%sT"%exA] = data[0]
                        elif variable == 1:
                            lake_data["%sO"%exA] = data.iloc[:,-1]
                        else:
                            icecoverduration = (data[6].sum())/10
                            lake_data["%sI"%exA] = icecoverduration

                for rcp in ["45","85"]:
                    for letter in ["T","O","I"]:
                        lake_data["diff%s%s"%(rcp,letter)] = lake_data["rcp%s%s"%(rcp,letter)]-lake_data["historical%s"%(letter)]
                data_summary = lake_data.mean()
                for rcp in ["45","85"]:
                    model_code = {1: 'KNM',
                              2: 'DMI',
                              3: 'MPI',
                              4: 'MOH',
                              5: 'IPS',
                              6: 'CNR'}
                    model_data.append([model_code.get(modelid),lake,"rcp%s"%rcp,data_summary["diff%sT"%rcp],data_summary["diff%sO"%rcp],data_summary["diff%sI"%rcp]])
        except:
            print("model %s doesnt exist"%(m1+m2))
    headers = model_data.pop(0)
    final_data = pd.DataFrame(model_data, columns=headers)
    final_data.to_csv("annually_average_T_Ice_cover_Oxygen.csv")
    plotT = sns.catplot(x = "model",y = variables[0][1],col = "scenario",data=final_data,kind="violin",split=True,height=8,aspect=.8)
    plotT.savefig("violinT.png")
    print("save T")
    plotO = sns.catplot(x = "model",y = variables[1][1],col = "scenario",data=final_data,kind="violin",split=True,height=8,aspect=.9)
    plotO.savefig("violinO.png")
    print("save O")
    plotI = sns.catplot(x = "model",y = variables[2][1],col = "scenario",data=final_data,kind="violin",split=True,height=8,aspect=.7)
    plotI.savefig("violinI.png")
    print("save Ice")
    print("end violin")

if __name__ == "__main__":
    #summary_characteristics_lake()
    run_calibrations(32276)
    #violin_plot()
    # lakes_list = ["Langtjern"]
    # input_files_parallel()
    # calibration_parallel()
    # mylake_parallel()
    # format_parallel()
