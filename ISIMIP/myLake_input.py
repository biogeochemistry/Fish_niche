import numpy as np
import pandas as pd
import datetime
import csv
import os
import shutil
import bz2
import math
import sys
import netCDF4 as ncdf

# Ouvrir les fichiers
# Extraire la liste de chaque variable
# Copier dans fichier input
# Faire un fichier pour chaque model/scenario

variables = ["hurs", "pr", "ps", "rsds", "sfcWind", "tas"]
models = ["GFDL-ESM2M", "HadGEM2-ES", "IPSL-CM5A-LR", "MIROC5"]
scenarios = ["historical", "piControl", "rcp26", "rcp60"]


def myLake_input(lake_name, forcing_data_directory, output_directory):
    """"""
    for model in models:
        for scenario in scenarios:
            list_dict = {"Year": [], "Month": [], "Day": [], "hurs": [], "pr": [], "ps": [], "rsds": [], "sfcWind": [], "tas": []}

            with open(os.path.join(output_directory, "{}_{}_{}_input.csv".format(lake_name[:3], model, scenario)), "w") as input_file:
                file = csv.writer(input_file, lineterminator="\n")
                file.writerows([[-999, "ISIMIP input tests"], ["Year", "Month","Day","Global radiation","Cloud cover",
                                       "Air temperature","Relative humidity","Air pressure","Wind speed","Precipitation",
                                       "Inflow_V","Inflow_T", "Inflow_PT", "Inflow_PST", "Inflow_DP", "Inflow_C", "Inflow_PP"]])

                for variable in variables:
                    ncdf_file = ncdf.Dataset(forcing_data_directory + "/{}_{}_{}_{}.allTS.nc".format(variable, model, scenario, lake_name), "r", format = "NETCDF4")


                    for x in ncdf_file.variables[variable][:]:
                        list_dict[variable].append(float(x))

                    if variable is variables[0]:
                        for y in ncdf_file.variables["time"][:]:
                            list_dict["Day"].append(float(y))

                    ncdf_file.close()

                file.writerows(zip(list_dict["Year"],
                                    list_dict["Month"],
                                    list_dict["Day"],
                                    list_dict["hurs"],
                                    list_dict["pr"],
                                    list_dict["ps"],
                                    list_dict["rsds"],
                                    list_dict["sfcWind"],
                                    list_dict["tas"]))


if __name__ == "__main__":
    myLake_input("Langtjern","C:/Users/User/Documents/GitHub/Fish_niche/ISIMIP/forcing_data/Langtjern", "C:/Users/User/Documents/GitHub/Fish_niche/ISIMIP/output/NO/Lan")