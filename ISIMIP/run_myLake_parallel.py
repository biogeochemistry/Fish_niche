
import run_myLake_ISIMIP
import myLake_post
import sys
import math
from joblib import Parallel, delayed
import multiprocessing

"""
Module that allows running multiple simulations or calibrations in parallel.
"""

lake_list = ["Allequash", "Alqueva", "Annecy", "Annie", "Argyle", "Biel", "BigMuskellunge", "BlackOak", "Bourget", "BurleyGriffin",
             "Crystal", "CrystalBog", "Delavan", "Dickie", "Eagle", "Ekoln", "Erken", "EsthwaiteWater", "FallingCreek",
             "Feeagh", "Fish", "Geneva", "GreatPond", "Green", "Harp", "Kilpisjarvi", "Kinneret", "Kivu", "Klicava", "Kuivajarvi",
             "Langtjern", "Laramie", "LowerLakeZurich", "Mendota", "Monona", "Mozaisk", "MtBold", "Muggelsee", "Neuchatel",
             "Ngoring", "NohipaloMustjarv", "NohipaloValgejarv", "Okauchee", "Paajarvi", "Rappbode", "Rimov", "Rotorua",
             "Sammamish", "Sau", "Sparkling", "Stechlin", "Sunapee", "Tahoe", "Tarawera", "Taupo", "Toolik", "Trout", "TroutBog",
             "TwoSisters", "Vendyurskoe", "Vortsjarv", "Waahi", "Washington", "Windermere", "Wingra", "Zlutice"]

def mylake_loop():
    """
    Loops through all lakes and runs
    :return:
    """

    pass

def calibration_loop():