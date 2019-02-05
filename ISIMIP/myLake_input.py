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
            with open(os.path.join(output_directory, "{}_{}_{}_input".format(lake_name[:3], model, scenario)), "w") as input_file:
                writer = csv.writer(input_file)
                writer.writerows([[-999, "ISIMIP input tests"], ["Z (m)", "Az (m2)","Tz (deg C)","Cz","Sz",
                                      "Pz (ug/l)","Chlaz","PPz","Chlaz_sed","PPz_sed","Hice","Hsnow"]])

                for variable in variables:
                    ncdf_file = ncdf.Dataset(forcing_data_directory + "/{}_{}_{}_{}.allTS.nc".format(variable, model, scenario, lake_name), "r", format = "NETCDF4")
                    writer.writerows(ncdf_file.variables[:])

                    ncdf_file.close()

                input_file.close()


if __name__ == "__main__":
    myLake_input("Langtjern","C:/Users/User/Documents/GitHub/Fish_niche/ISIMIP/forcing_data/Langtjern", "C:/Users/User/Documents/GitHub/Fish_niche/ISIMIP/output/NO/Lan")