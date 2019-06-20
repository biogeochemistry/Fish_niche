"""
run_mylake_parallel.py

Module that allows running multiple simulations or calibrations in parallel.
"""
import run_myLake_ISIMIP
import myLake_post
import sys
import math
from joblib import Parallel, delayed
import multiprocessing as mp

num_cores = mp.cpu_count()

lake_list = ["Allequash", "Alqueva", "Annecy", "Annie", "Argyle", "Biel", "BigMuskellunge", "BlackOak", "Bourget", "BurleyGriffin",
             "Crystal", "CrystalBog", "Delavan", "Dickie", "Eagle", "Ekoln", "Erken", "EsthwaiteWater", "FallingCreek",
             "Feeagh", "Fish", "Geneva", "GreatPond", "Green", "Harp", "Kilpisjarvi", "Kinneret", "Kivu", "Klicava", "Kuivajarvi",
             "Langtjern", "Laramie", "LowerLakeZurich", "Mendota", "Monona", "Mozaisk", "MtBold", "Muggelsee", "Neuchatel",
             "Ngoring", "NohipaloMustjarv", "NohipaloValgejarv", "Okauchee", "Paajarvi", "Rappbode", "Rimov", "Rotorua",
             "Sammamish", "Sau", "Sparkling", "Stechlin", "Sunapee", "Tahoe", "Tarawera", "Taupo", "Toolik", "Trout", "TroutBog",
             "TwoSisters", "Vendyurskoe", "Vortsjarv", "Waahi", "Washington", "Windermere", "Wingra", "Zlutice"]

regions = {"US": ["Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal", "CrystalBog", "Delavan", "Fish", "Laramie", "Mendota", "Monona",
                  "Okauchee", "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik", "Trout", "TroutBog", "TwoSisters",
                  "Washington", "Wingra"],
           "CH": ["Biel", "LowerLakeZurich", "Neuchatel", ],
           "PT": ["Alqueva"],
           "FR": ["Annecy", "Bourget", "Geneva"],
           "AU": ["Argyle", "BurleyGriffin", "MtBold"],
           "CA": ["Dickie", "Eagle", "Harp"],
           "SE": ["Ekoln", "Erken"],
           "UK": ["EsthwaiteWater", "Windermere"]}

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

def model_scenario_loop():
    """
    Loops through all models and scenarios
    :return:
    """

    pass

def calibration_parallel():




    Parallel(n_jobs=num_cores, verbose=10) (delayed (myLake_post.optimize_Nelder_Meald) (lake, "observations/{}_{}".format(region, lake),
            "input/{}/{}".format(region, lake), region, "forcing_data/{}".format(lake), "output/{}/{}".format(region, lake), model, scenario)
            for (lake, model, scenario) in (lake_list, models, scenarios))

    pass

def run_calibrations():

    region = lake[]

    return myLake_post.optimize_Nelder_Meald(lake)