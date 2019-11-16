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
from multiprocessing import Pool
import numpy as np

num_cores = mp.cpu_count() - 6
hyspomissing =[ "Monona","Ngoring",]
allmissing = ["Annie" ,"Klicava","Zlutice",]
forcingdatamissing=["Mozhaysk","Muggelsee","Taupo","Waahi",]
order = ["Eagle", "Ekoln","Sau","Tahoe", "TwoSisters","Vendyurskoe","Washington",]

problematic = ["Trout","FallingCreek", "GreatPond","Feeagh","MtBold","Rotorua",]

optimisation_completed = ["BigMuskellunge", "Allequash","CrystalBog","Erken","BurleyGriffin",
                          "Dickie", "Langtjern","LowerZurich","Mendota","Delavan",
                          "Sparkling", "Trout", "TroutBog","Fish","Wingra",
                          "Harp","NohipaloValgejarv","Alqueva","Toolik","Sunapee",
                          "Kuivajarvi","Crystal"]
test =[]
coundntop =[ "Sammamish",
           "Geneva","Kinneret","Bourget","Stechlin","Argyle","Rappbode","BlackOak",
           "NohipaloMustjarv","Green_lake","Paajarvi","Annecy","Okauchee","EsthwaiteWater",  "Laramie","Rimov",
           "Biel","Tarawera", "Neuchatel",
           "Kilpisjarvi","Vortsjarv","Kivu"]


lake_list = ["Allequash", "Annecy", "Biel", "BigMuskellunge", "BlackOak",
             "BurleyGriffin", "Crystal", "Delavan","Dickie", "Erken",
             "Fish", "Geneva", "Green","Harp","Kilpisjarvi",
             "Kinneret", "Kivu", "Langtjern", "Laramie","LowerZurich",
             "Mendota", "Mozaisk", "Neuchatel", "Okauchee", "Paajarvi",
             "Rotorua", "Sparkling", "Stechlin","Sunapee", "Tarawera",
             "Toolik", "Trout", "TroutBog", "TwoSisters","Wingra"
             ]
problem = []
full_lake_list =["Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal",
                 "CrystalBog", "Delavan","FallingCreek", "Fish", "Great_Pond",
                 "Green_lake", "Laramie", "Mendota", "Monona","Okauchee",
                 "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik",
                 "Trout", "TroutBog", "Two_Sisters","Washington", "Wingra",
                 "Biel", "LowerZurich", "Neuchatel","Alqueva","Annecy",
                 "Bourget", "Geneva","Argyle", "BurleyGriffin", "MtBold",
                 "Dickie", "Eagle_lake", "Harp","Ekoln", "Erken",
                 "EsthwaiteWater", "Windermere","Feeagh","Kilpisjarvi", "Kuivajarvi",
                 "Paajarvi","Kinneret","Kivu","Klicava", "Rimov",
                 "Zlutice","Langtjern","Mozhaysk", "Vendyurskoe","Muggelsee",
                 "Rappbode", "Stechlin","Ngoring","NohipaloMustjarv", "NohipaloValgejarv",
                 "Vortsjarv","Sau","Rotorua", "Tarawera", "Taupo",
                 "Waahi"]

#"Ngoring","Klicava",
full_lake_list =["Argyle","Alqueva","Rotorua","Muggelsee","BigMuskellunge","Vendyurskoe",
                 "EsthwaiteWater","Allequash", "Annie",  "BlackOak", "Crystal",
                 "CrystalBog", "Delavan","FallingCreek", "Fish", "GreatPond",
                 "Green_lake", "Laramie", "Mendota", "Okauchee",
                  "Sparkling", "Sunapee", "Toolik",
                 "Trout", "TroutBog", "Two_Sisters", "Wingra",
                 "Biel", "LowerZurich", "Neuchatel","Annecy",
                 "Bourget",  "BurleyGriffin", "MtBold",
                 "Dickie", "Eagle_lake", "Harp","Ekoln", "Erken",
                 "EsthwaiteWater", "Windermere","Feeagh","Kilpisjarvi", "Kuivajarvi",
                 "Paajarvi","Kinneret","Kivu", "Rimov",
                 "Zlutice","Langtjern","Mozhaysk",
                 "Rappbode", "Stechlin","NohipaloMustjarv", "NohipaloValgejarv",
                 "Vortsjarv","Sau", "Tarawera", "Taupo",
                 "Waahi","Washington","Sammamish", "Geneva","Tahoe"]


#full_lake_list = "Allequash"
regions = {"US": ["Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal", "CrystalBog", "Delavan",
                  "Falling_Creek", "Fish", "GreatPond", "Green_lake", "Laramie", "Mendota", "Monona",
                  "Okauchee", "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik", "Trout", "TroutBog", "TwoSisters",
                  "Washington", "Wingra"],
           "CH": ["Biel", "LowerZurich", "Neuchatel"],
           "PT": ["Alqueva"],
           "FR": ["Annecy", "Bourget", "Geneva"],
           "AU": ["Argyle", "BurleyGriffin", "MtBold"],
           "CA": ["Dickie", "Eagle_lake", "Harp"],
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
           "NZ": ["Rotorua", "Tarawera", "Taupo", "Waahi"]}

models = [ "EWEMBI","GFDL-ESM2M",
          "HadGEM2-ES",
          "IPSL-CM5A-LR",
          "MIROC5"
          ]
scenarios = ["historical",
             "piControl",
             "rcp26",
             "rcp60",
             "rcp85"
             ]

input_variables = ["hurs",
                   "pr",
                   "ps",
                   "rsds",
                   "sfcWind",
                   "tas"
                    ]
params_0 = np.array([0, 0.3, 0.55, 1, 0, 2.5, 1])
report = 'report.txt'




def input_files_parallel():
    with open(report, 'w') as f:
        f.write('\nrunning _parallel\n' )
        f.close()

    print("start")

    Parallel(n_jobs=num_cores, verbose=10)(delayed(input_files_loop)(lake) for lake in full_lake_list)

def input_files_loop(lake):


    corrected_names = ["Allequash_Lake", "Big_Muskellunge_Lake", "Black_Oak_Lake", "Burley_Griffin", "Crystal_Bog",
                       "Crystal_Lake",
                       "Dickie_Lake", "Eagle_Lake", "Ekoln_basin_of_Malaren", "Esthwaite_Water",
                       "Falling_Creek",
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
        elif lake == "TroutBog":
            f_lake = "Trout_Bog"
            break
        elif lake == "Two_Sisters":
            f_lake = "Two_Sisters_Lake"
            break
        elif lake == "Toolik":
            f_lake = "Toolik_Lake"
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
            try:
            # if scenario == "historical" and model == 'EWEMBI':
            #     print(r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\input\{}\{}_{}_{}_input".format(reg,lake[:3], model, scenario))
            #
            #     if not (os.path.exists( r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\input/{}/{}\{}_{}_{}_input".format(reg,lake[:3],lake[:3], model, scenario))):
            #         try:
            #
            #             run_myLake_ISIMIP.generate_input_files("observations/{}/{}".format(reg, lake), lake, f_lake,
            #                                        "D:/forcing_data", run_myLake_ISIMIP.get_longitude(f_lake, "D:/forcing_data",model,scenario),
            #                                        run_myLake_ISIMIP.get_latitude(f_lake, "D:/forcing_data",model,scenario), model, scenario)
            #         except:
            #             print("missing data")
            #         if (os.path.exists(
            #                 r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations/{}/{}\{}_temp_daily.csv".format(
            #                         reg, lake, lake))):
            #
            #             myLake_post.temperatures_by_depth(
            #                 r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations/{}/{}/".format(reg,
            #                                                                                                lake),
            #                 lake,
            #                 os.path.join(r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\output", reg, lake),
            #                 model, scenario)
            #         else:
            #             print('no daily data for %s'% lake)
            #     else:
            #         print('Already done')
            #         print(r"output/{}/{}/{}/{}/Observed_Temperatures.csv".format(reg, lake, model, scenario))
            #         if not (os.path.exists(r"output/{}/{}/{}/{}/Observed_Temperatures.csv".format(reg, lake, model, scenario))):
            #             if (os.path.exists(
            #                     r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations/{}/{}\{}_temp_daily.csv".format(
            #                         reg, lake, lake))):
            #                 myLake_post.temperatures_by_depth("observations/{}/{}".format(reg, lake), lake,
            #                                           "output/{}/{}/{}/{}".format(reg, lake, model, scenario),model,scenario)
            #             else:
            #                 print('no daily data for %s' % lake)
            # else:
                if not model == "EWEMBI":
                    run_myLake_ISIMIP.generate_input_files("observations/{}/{}".format(reg, lake), lake, f_lake,
                                                           "D:/forcing_data",
                                                           run_myLake_ISIMIP.get_longitude(f_lake, "D:/forcing_data"),
                                                           run_myLake_ISIMIP.get_latitude(f_lake, "D:/forcing_data"),
                                                           model, scenario)
            except:
               print("problem when doing %s %s %s" % (lake, model, scenario))

            if not os.path.exists("input/{}/{}/{}_{}_{}_input".format(reg, lake[:3],lake[:3],model,scenario)) or not os.path.exists("input/{}/{}/{}_init".format(reg, lake[:3],lake[:3])) or not os.path.exists("input/{}/{}/{}_par".format(reg, lake[:3],lake[:3])):
                print("not all initial files existing for %s %s %s" % ( model, scenario,lake))
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

    for model in models:
        for scenario in scenarios:
            if model == "EWEMBI" and "historical":

                with pysftp.Connection('mistralpp.dkrz.de', username='b380750', password='TwopFaP5') as sftp:
                    sftp.cwd(
                        "/mnt/lustre01/work/bb0820/ISIMIP/ISIMIP2b/InputData/OBS_atmosphere/local_lakes/EWEMBI/historical")

                    for var in input_variables:
                        if not os.path.exists(
                                "D:/forcing_data\\{}_EWEMBI_historical_{}.allTS.nc".format(var, lake)):
                            print("start scenario EWE histo")
                            try:
                                sftp.get("{}/{}_EWEMBI_historical_{}.allTS.nc".format(lake, var, lake),
                                         localpath="D:/forcing_data\\{}_EWEMBI_historical_{}.allTS.nc".format(var,
                                                                                                              lake))
                                print("end")

                            except:
                                print("enable to get {}/{}_EWEMBI_historical_{}.allTS.nc".format(lake, var, lake))
                        else:
                            with open(report, 'a') as f:
                                f.write('download already done %s \n' % (lake))
                                f.close()
                            print('download already done %s \n' % (lake))

            else:

                with pysftp.Connection('mistralpp.dkrz.de', username='b380750', password='TwopFaP5') as sftp:
                    sftp.cwd( "/mnt/lustre01/work/bb0820/ISIMIP/ISIMIP2b/InputData/GCM_atmosphere/biascorrected/local_lakes")
                    for var in input_variables:
                        if not os.path.exists("D:/forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, lake)):
                            print("start scenario %s"%(scenario))
                            try:
                                sftp.get("{}/{}_{}_{}_{}.allTS.nc".format(lake, var, model, scenario, lake), localpath="D:/forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, lake))
                                print("end")
                            except:
                                print("enable to get {}/{}_{}_{}_{}.allTS.nc".format(lake, var, model, scenario, lake))
                        else:
                            with open(report, 'a') as f:
                                f.write('download already done %s \n' % (lake))
                                f.close()
                            print('download already done %s \n' % (lake))





def mylake_parallel():

    Parallel(n_jobs=num_cores, verbose=10)(delayed(model_scenario_loop)(lake) for lake in full_lake_list)


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
        with open("observations/{}/{}/{}_hypsometry2.csv".format(reg,lake, lake)) as obs:
            reader = list(csv.reader(obs))
            prefix = reader[1][0][3:]

        for model in models:
            for scenario in scenarios:

                if os.path.exists("output/{}/{}/{}/{}/RunComplete".format(reg, lake, model, scenario)):
                    print("{} {} {} Run is already completed.\n".format(lake, model, scenario))

                elif os.path.exists("output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(reg, lake)):
                    print("Running {} {} {}.\n".format(lake, model, scenario))
                    try:
                        run_myLake_ISIMIP.run_myLake("observations/{}/{}".format(reg, lake), "input/{}/{}".format(reg, prefix), reg, lake, model, scenario)
                        print("Run of {} {} {} Completed.\n".format(lake, model, scenario))
                    except:
                        print("problem with {} {} {}.\n".format(lake, model, scenario))
                else:
                    print("{} Calibration have not been done.\n".format(lake))


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
    print(num_cores)
    with open(report, 'w') as f:
        f.write('\nrunning _parallel\n' )
        f.close()

    #run_calibrations(full_lake_list[0])
    Parallel(n_jobs=num_cores, verbose=10)(delayed(run_calibrations)(lake) for lake in full_lake_list)


def run_calibrations(lake):
    """
    Intermediary function to call nelder-mead optimisation function for a single lake.

    :param lake: Type string. The name of the lake to calibrate.
    :return: If Calibration_Complete file is found, returns None. Else, return the nelder-mead optimisation function
    from myLake_post module for the given lake.
    """
    print(lake)
    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    if reg == None:
        print("Cannot find {}'s region".format(lake))
        return None
    else:
        with open("observations/{}/{}/{}_hypsometry2.csv".format(reg, lake, lake)) as obs:
            reader = list(csv.reader(obs))
            prefix = reader[1][0][3:]
    optimisation_completed_unsuccessfull = []
    for region in regions:
        if lake in regions[region]:

            if os.path.exists("output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(region, lake)):
                print("Calibration for {} is already complete.\n".format(lake))
                return None
            elif not os.path.exists("input/{}/{}/{}_EWEMBI_historical_input".format(region, prefix,prefix)):
                print("Calibration can't {} be completed. input file doesn't exist.\n".format(lake))
                return None
            elif not os.path.exists("input/{}/{}/{}_EWEMBI_historical_input".format(region, prefix,prefix)) or not os.path.exists("input/{}/{}/{}_init".format(region, prefix,prefix)) or not os.path.exists("input/{}/{}/{}_par".format(region, prefix,prefix)):
                print("not all initial files existing for %s" % ( lake))
            elif os.path.exists("output/{}/{}/EWEMBI/historical/Calibration_problem.txt".format(region, lake)):
                print("Unable to calibration {}.\n".format(lake))
                return None


            else:

                return myLake_post.optimize_differential_evolution(lake, "observations/{}/{}".format(reg,lake),
                                                     "input/{}/{}".format(region, prefix), region,
                                                     "output/{}/{}/{}/{}".format(region, lake, "EWEMBI","historical"),"EWEMBI","historical")


    print("Cannot find {}'s region".format(lake))

if __name__ == "__main__":
    #
    input_files_parallel()
    # lake ="BurleyGriffin"
    # reg = "AU"
    # model="EWEMBI"
    # scenario= "historical"
    #for lake in full_lake_list:
    #    input_files_loop(lake)
       # myLake_post.temperatures_by_depth(r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations/{}/{}/".format(reg, lake), lake,
       #                                   os.path.join(r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\output",reg, lake),model,scenario)
    #input_files_loop("Washington")
    # #Parallel(n_jobs=num_cores)(delayed(run_calibrations(lake)) for lake in full_lake_list)
    #for lake in ['Allequash_Lake', 'Alqueva', 'Annecy', 'Annie', 'Argyle', 'Biel', 'Big_Muskellunge_Lake', 'Black_Oak_Lake', 'Bourget', 'Burley_Griffin', 'Crystal_Bog', 'Crystal_Lake', 'Delavan', 'Dickie_Lake', 'Eagle_Lake', 'Ekoln_basin_of_Malaren', 'Erken', 'Esthwaite_Water', 'Falling_Creek_Reservoir', 'Feeagh', 'Fish_Lake', 'Geneva', 'Great_Pond', 'Green_Lake', 'Harp_Lake', 'Kilpisjarvi', 'Kinneret', 'Kivu', 'Klicava', 'Kuivajarvi', 'Langtjern', 'Laramie_Lake', 'Lower_Zurich', 'Mendota', 'Monona', 'Mozaisk', 'Mt_Bold', 'Mueggelsee', 'Neuchatel', 'Ngoring', 'Nohipalo_Mustjarv', 'Nohipalo_Valgejarv', 'Okauchee_Lake', 'Paajarvi', 'Rappbode_Reservoir', 'Rimov', 'Rotorua', 'Sammamish', 'Sau_Reservoir', 'Sparkling_Lake', 'Stechlin', 'Sunapee', 'Tahoe', 'Tarawera', 'Toolik_Lake', 'Trout_Bog', 'Trout_Lake', 'Two_Sisters_Lake', 'Vendyurskoe', 'Victoria', 'Vortsjarv', 'Washington', 'Windermere', 'Wingra', 'Zlutice']:
    #    download_forcing_data(lake)
    #calibration_parallel()
    mylake_parallel()
    #model_scenario_loop("Langtjern")
    #run_calibrations(full_lake_list[1])