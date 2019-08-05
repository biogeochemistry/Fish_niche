"""
run_mylake_parallel.py

Module that allows running multiple simulations or calibrations in parallel.
"""
import run_myLake_ISIMIP
import myLake_post
import sys
import math
import os
import pysftp
from joblib import Parallel, delayed
import multiprocessing as mp

num_cores = mp.cpu_count()

lake_list = ["Allequash", "Alqueva", "Annecy", "Annie", "Argyle", "Biel", "BigMuskellunge", "BlackOak", "Bourget", "BurleyGriffin",
             "Crystal", "CrystalBog", "Delavan", "Dickie", "Eagle", "Ekoln", "Erken", "EsthwaiteWater", "FallingCreek",
             "Feeagh", "Fish", "Geneva", "GreatPond", "Green", "Harp", "Kilpisjarvi", "Kinneret", "Kivu", "Klicava", "Kuivajarvi",
             "Langtjern", "Laramie", "LowerLakeZurich", "Mendota", "Monona", "Mozaisk", "MtBold", "Muggelsee", "Neuchatel",
             "Ngoring", "NohipaloMustjarv", "NohipaloValgejarv", "Okauchee", "Paajarvi", "Rappbode", "Rimov", "Rotorua",
             "Sammamish", "Sau", "Sparkling", "Stechlin", "Sunapee", "Tahoe", "Tarawera", "Taupo", "Toolik", "Trout", "TroutBog",
             "TwoSisters", "Vendyurskoe", "Vortsjarv", "Waahi", "Washington", "Windermere", "Wingra", "Zlutice"
            ]

regions = {"US": ["Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal", "CrystalBog", "Delavan", "Fish", "Laramie", "Mendota", "Monona",
                  "Okauchee", "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik", "Trout", "TroutBog", "TwoSisters",
                  "Washington", "Wingra"],
           "CH": ["Biel", "LowerLakeZurich", "Neuchatel"],
           "PT": ["Alqueva"],
           "FR": ["Annecy", "Bourget", "Geneva"],
           "AU": ["Argyle", "BurleyGriffin", "MtBold"],
           "CA": ["Dickie", "Eagle", "Harp"],
           "SE": ["Ekoln", "Erken"],
           "UK": ["EsthwaiteWater", "Windermere"],
           "IE": ["Feeagh"],
           "FI": ["Kilpisjarvi", "Kuivajarvi", "Paajarvi"],
           "IL": ["Kinneret"],
           "RW": ["Kivu"],
           "CZ": ["Klicava", "Rimov", "Zlutice"],
           "NO": ["Langtjern"],
           "RU": ["Mozaisk", "Vendyurskoe"],
           "DE": ["Muggelsee", "Rappbode", "Stechlin"],
           "CN": ["Ngoring"],
           "EE": ["NohipaloMustjarv", "NohipaloValgejarv", "Vortsjarv"],
           "ES": ["Sau"],
           "NZ": ["Rotura", "Tarawera", "Taupo", "Waahi"]}

models = ["GFDL-ESM2M",
          "HadGEM2-ES",
          "IPSL-CM5A-LR",
          "MIROC5"
          ]
scenarios = ["historical",
             "piControl",
             "rcp26",
             "rcp60"
             ]
input_variables = ["hurs",
                   "pr",
                   "ps",
                   "rsds",
                   "sfcWind",
                   "tas"
                    ]


def input_files_parallel():
    Parallel(n_jobs=num_cores, verbose=10)(delayed(input_files_loop(lake)) for lake in lake_list)

def input_files_loop(lake):


    download_forcing_data(lake)
    for model in models:
        for scenario in scenarios:
            run_myLake_ISIMIP.generate_input_files("observations/{}/{}_hypsometry.csv".format(lake, lake),
                                                   "observations/{}/{}_temp_daily.csv".format(lake, lake), lake,
                                                   "forcing_data", run_myLake_ISIMIP.get_longitude(lake, "forcing_data"),
                                                   run_myLake_ISIMIP.get_latitude(lake, "forcing_data"), model, scenario)
    for model in models:
        for scenario in scenarios:
            for var in input_variables:
                os.remove("forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, lake))


def download_forcing_data(lake):
    """
    A function to download forcing data from dkrz server.
    :param lake:
    :return:
    """

    corrected_names = ["Allequash_Lake", "Big_Muskellunge_Lake", "Black_Oak_Lake", "Burley_Griffin", "Crystal_Bog", "Crystal_Lake",
                       "Dickie_Lake", "Eagle_Lake", "Ekoln_basin_of_Malaren", "Esthwaite_Water", "Falling_Creek_Reservoir",
                       "Fish_Lake", "Great_Pond", "Green_Lake", "Harp_Lake", "Laramie_Lake", "Lower_Zurich", "Mt_Bold",
                       "Nohipalo_Mustjarv", "Nohipalo_Valgejarv", "Okauchee_Lake", "Rappbode_Reservoir", "Sau_Reservoir",
                       "Sparkling_Lake", "Toolik_Lake", "Trout_Bog", "Trout_Lake", "Two_Sisters_Lake"]

    for name in corrected_names:
        if lake in name:
            lake = name
            break

    with pysftp.Connection('mistralpp.dkrz.de', username='b380750', password='TwopFaP5') as sftp:
        sftp.cwd("/mnt/lustre01/work/bb0820/ISIMIP/ISIMIP2b/InputData/GCM_atmosphere/biascorrected/local_lakes")

        for model in models:
            for scenario in scenarios:
                for var in input_variables:
                   sftp.get("{}/{}_{}_{}_{}.allTS.nc".format(lake, var, model, scenario, lake), localpath="forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, lake))


def mylake_parallel():

    Parallel(n_jobs=num_cores, verbose=10)(delayed(model_scenario_loop(lake)) for lake in lake_list)


def model_scenario_loop(lake):


    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    if reg == None:
        print("Cannot find specified lake's region")
        return None

    for model in models:
        for scenario in scenarios:
            if os.path.exists("output/{}/{}/{}/{}/RunComplete".format(reg, lake, model, scenario)):
                print("{} {} {} Run is already completed.\n".format(lake, model, scenario))

            else:
                print("Running {} {} {}.\n".format(lake, model, scenario))
                run_myLake_ISIMIP.run_myLake("observations/{}".format(lake), "input/{}/{}".format(reg, lake[:3]), reg, lake, model, scenario)

def make_parameters_file_parallel():
    """
    Parallelise the function get_best_parameters.
    """
    Parallel(n_jobs=num_cores, verbose=10)(delayed(get_best_parameters(lake)) for lake in lake_list)

def get_best_parameters(lake):
    """
    Looks for the results of calibration for a lake and returns the value for the parameters.
    :param lake:
    :return:
    """
    reg = ''
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    with open("output/{}/{}/GFDL-ESM2M/historical".format(reg, lake)) as results:
        pass


def calibration_parallel():
    """
    Simple function to call a parallel calibration of all lakes.
    :return:
    """

    Parallel(n_jobs=num_cores, verbose=10) (delayed (run_calibrations) (lake) for lake in lake_list)

def run_calibrations(lake):
    """
    Intermediary function to call nelder-mead optimisation function for a single lake.

    :param lake: Type string. The name of the lake to calibrate.
    :return: If Calibration_Complete file is found, returns None. Else, return the nelder-mead optimisation function
    from myLake_post module for the given lake.
    """

    for region in regions:
        if lake in regions[region]:
            if os.path.exists("output/{}/{}/GFDL-ESM2M/rcp26/Calibration_Complete.txt".format(region, lake)):
                print("Calibration for {} is already complete.\n".format(lake))
                return None

            else:
                return myLake_post.optimize_differential_evolution(lake, "observations/{}".format(lake),
                                                     "input/{}/{}".format(region, lake), region,
                                                     "forcing_data/{}".format(lake), "output/{}/{}".format(region, lake),
                                                     "GFDL-ESM2M", "historical")
    print("Cannot find specified lake's region")

if __name__ == "__main__":
    input_files_parallel()