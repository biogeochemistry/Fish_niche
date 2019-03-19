import mylake_init_ISIMIP

# Pour un lac
# Doit appeler les autres scripts pour créer les fichiers
# Le outpath est déterminé par mylake_init et doit ensuite être passé aux scripts suivants
#

""" Main script for MyLake - ISIMIP
Calls the init, input and par scripts to create the appropriate files for MyLake model
Then launches MyLake for the specified lake
"""

#def run_myLake(hypsometry_path, temperature_path):




def generate_data_files(hypsometry_path, temperature_path, lake_name, forcing_data_directory, longitude, latitude):
    """
    Creates all files needed for a run of mylake model with a single lake. The input function will generate ALL needed
    input files(one for each combination of scenario, model and variable)
    :param hypsometry_path:
    :param temperature_path:
    :param lake_name:
    :param forcing_data_directory:
    :param longitude:
    :param latitude:
    :return:
    """
    outdir = mylakeinit(init_info(hypsometry_path, temperature_path))
    myLake_input(lake_name, forcing_data_directory, outdir)
    mylakepar(longitude, latitude, outdir)


if __name__ == "__main__":
    run_myLake("observations/NO_Lan/Langtjern_hypsometry.csv", "observations/NO_Lan/Langtjern_temperature.csv")