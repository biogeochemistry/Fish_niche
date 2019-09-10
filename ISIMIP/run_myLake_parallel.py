"""
run_mylake_parallel.py

Module that allows running multiple simulations or calibrations in parallel.
"""
import run_myLake_ISIMIP
import myLake_post
import csv
import sys
import math
import os
import pysftp
from joblib import Parallel, delayed
import multiprocessing as mp

num_cores = mp.cpu_count() - 2
problematic = ["Annie","Eagle",  "Ekoln", "Fish", "Klicava", "Monona", "Mozhaysk", "MtBold","Muggelsee",
             "Ngoring","Sau", "Tahoe", "Taupo","TwoSisters", "Vendyurskoe", "Waahi", "Washington","Vortsjarv","Wingra","Zlutice", "Trout"]
full_lake_list = ["Allequash", "Alqueva", "Annecy",  "Argyle", "Biel", "BigMuskellunge", "BlackOak", "Bourget", "BurleyGriffin",
             "Crystal", "CrystalBog", "Delavan", "Dickie", "Erken", "EsthwaiteWater", "FallingCreek",
             "Feeagh", "Geneva", "GreatPond", "Green", "Harp", "Kilpisjarvi", "Kinneret", "Kivu", "Kuivajarvi",
             "Langtjern", "Laramie", "LowerZurich", "Mendota",  "Neuchatel",
             "NohipaloMustjarv", "NohipaloValgejarv", "Okauchee", "Paajarvi", "Rappbode", "Rimov", "Rotorua",
             "Sammamish",  "Sparkling", "Stechlin", "Sunapee",  "Tarawera",  "Toolik","Trout", "TroutBog"
             ]


full_lake_lit=[  "Allequash"]
lake_list = ["Allequash", "Annecy", "Biel", "BigMuskellunge", "BlackOak",
             "BurleyGriffin", "Crystal", "Delavan",
             "Dickie", "Erken",
             "Fish", "Geneva", "Green",
             "Harp",
             "Kilpisjarvi", "Kinneret", "Kivu", "Langtjern", "Laramie",
             "LowerZurich", "Mendota", "Mozaisk", "Neuchatel", "Okauchee", "Paajarvi", "Rotorua", "Sparkling", "Stechlin",
             "Sunapee", "Tarawera", "Toolik", "Trout", "TroutBog", "TwoSisters",
             "Wingra"
             ]


regions = {"US": ["Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal", "CrystalBog", "Delavan",
                  "FallingCreek", "Fish", "GreatPond", "Green", "Laramie", "Mendota", "Monona",
                  "Okauchee", "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik", "Trout", "TroutBog", "TwoSisters",
                  "Washington", "Wingra"],
           "CH": ["Biel", "LowerZurich", "Neuchatel"],
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
           "RU": ["Mozhaysk", "Vendyurskoe"],
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

report = 'report.txt'
def input_files_parallel():
    with open(report, 'w') as f:
        f.write('\nrunning _parallel\n' )
        f.close()


    Parallel(n_jobs=num_cores, verbose=10)(delayed(input_files_loop(lake)) for lake in full_lake_list)

def input_files_loop(lake):


    corrected_names = ["Allequash_Lake", "Big_Muskellunge_Lake", "Black_Oak_Lake", "Burley_Griffin", "Crystal_Bog",
                       "Crystal_Lake",
                       "Dickie_Lake", "Eagle_Lake", "Ekoln_basin_of_Malaren", "Esthwaite_Water",
                       "Falling_Creek_Reservoir",
                       "Fish_Lake", "Great_Pond", "Green_Lake", "Harp_Lake", "Laramie_Lake", "Lower_Zurich", "Mt_Bold",
                       "Nohipalo_Mustjarv", "Nohipalo_Valgejarv", "Okauchee_Lake", "Rappbode_Reservoir",
                       "Sau_Reservoir",
                       "Sparkling_Lake", "Toolik_Lake", "Trout_Bog", "Trout_Lake", "Two_Sisters_Lake"]

    f_lake = lake
    for name in corrected_names:
        if lake == "Crystal":
            f_lake = "Crystal_Lake"
            break
        elif lake == "Trout":
            f_lake = "Trout_Lake"
            break

        if lake in name.replace("_", ''):
            f_lake = name
            break
    try:
        with open(report, 'a') as f:
            f.write('running lake %s \n'% (lake))
            f.close()
        print("download for %s"%lake)
        download_forcing_data(f_lake)
        with open(report, 'a') as f:
            f.write('download for %s completed\n'% (lake))
            f.close()
            print('download for %s completed'% lake)
    except:
        with open(report, 'a') as f:
            f.write('unable download for %s\n'% (lake))
            f.close()
        print('unable to download of %s' % lake)

    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    if reg == None:
        print("Cannot find {}'s region".format(lake))
        return None
    for model in models:
        for scenario in scenarios:

            run_myLake_ISIMIP.generate_input_files("observations/{}/{}".format(reg, lake), lake, f_lake,
                                                   "D:/forcing_data", run_myLake_ISIMIP.get_longitude(f_lake, "D:/forcing_data"),
                                                   run_myLake_ISIMIP.get_latitude(f_lake, "D:/forcing_data"), model, scenario)
    #for model in models:
    #    for scenario in scenarios:
    #        for var in input_variables:
    #            os.remove("forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, f_lake))


def download_forcing_data(lake):
    """
    A function to download forcing data from dkrz server.
    :param lake:
    :return:
    """

    with pysftp.Connection('mistralpp.dkrz.de', username='b380750', password='TwopFaP5') as sftp:
        sftp.cwd("/mnt/lustre01/work/bb0820/ISIMIP/ISIMIP2b/InputData/GCM_atmosphere/biascorrected/local_lakes")

        for model in models:
            for scenario in scenarios:
                for var in input_variables:
                    if not os.path.exists("D:/forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, lake)):
                        print("start")
                        sftp.get("{}/{}_{}_{}_{}.allTS.nc".format(lake, var, model, scenario, lake), localpath="D:/forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, lake))
                        print("end")
                    else:
                        with open(report, 'a') as f:
                            f.write('download already done %s \n' % (lake))
                            f.close()
                        print('download already done %s \n' % (lake))

def mylake_parallel():

    Parallel(n_jobs=num_cores, verbose=10)(delayed(model_scenario_loop(lake)) for lake in lake_list)


def model_scenario_loop(lake):



    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    if reg == None:
        print("Cannot find {}'s region".format(lake))
        return None
    else:
        with open("observations/{}/{}/{}_hypsometry.csv".format(reg,lake, lake)) as obs:
            reader = list(csv.reader(obs))
            prefix = reader[1][0][3:]

        for model in models:
            for scenario in scenarios:

                if os.path.exists("output/{}/{}/{}/{}/RunComplete".format(reg, lake, model, scenario)):
                    print("{} {} {} Run is already completed.\n".format(lake, model, scenario))

                elif os.path.exists("output/{}/{}/GFDL-ESM2M/rcp26/Calibration_Complete.txt".format(reg, lake)):
                    print("Running {} {} {}.\n".format(lake, model, scenario))
                    run_myLake_ISIMIP.run_myLake("observations/{}/{}".format(reg, lake), "input/{}/{}".format(reg, prefix), reg, lake, model, scenario)


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

    Parallel(n_jobs=num_cores, verbose=10) (delayed (run_calibrations) (lake) for lake in full_lake_list)

def run_calibrations(lake):
    """
    Intermediary function to call nelder-mead optimisation function for a single lake.

    :param lake: Type string. The name of the lake to calibrate.
    :return: If Calibration_Complete file is found, returns None. Else, return the nelder-mead optimisation function
    from myLake_post module for the given lake.
    """
    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    if reg == None:
        print("Cannot find {}'s region".format(lake))
        return None
    else:
        with open("observations/{}/{}/{}_hypsometry.csv".format(reg,lake, lake)) as obs:
            reader = list(csv.reader(obs))
            prefix = reader[1][0][3:]

    for region in regions:
        if lake in regions[region]:
            if os.path.exists("output/{}/{}/GFDL-ESM2M/rcp26/Calibration_Complete.txt".format(region, lake)):
                print("Calibration for {} is already complete.\n".format(lake))
                return None

            else:
                return myLake_post.optimize_differential_evolution(lake, "observations/{}/{}".format(reg,lake),
                                                     "input/{}/{}".format(region, prefix), region,
                                                     "output/{}/{}/{}/{}".format(region, lake, "GFDL-ESM2M", "rcp26"),
                                                     "GFDL-ESM2M", "rcp26")
    print("Cannot find {}'s region".format(lake))

if __name__ == "__main__":
    #input_files_parallel()
    calibration_parallel()
    mylake_parallel()