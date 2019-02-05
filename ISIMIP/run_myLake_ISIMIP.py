from mylake_init_ISIMIP import mylakeinit, init_info


# Pour un lac
# Doit appeler les autres scripts pour créer les fichiers
# Le outpath est déterminé par mylake_init et doit ensuite être passé aux scripts suivants
#

""" Main script for MyLake - ISIMIP
Calls the init, input and par scripts to create the appropriate files for MyLake model
Then launches MyLake for the specified lake
"""

def run_myLake(hypsometry_path, temperature_path):

    outdir = mylakeinit(init_info(hypsometry_path, temperature_path))
    print(outdir)

if __name__ == "__main__":
    run_myLake("observations/NO_Lan/Langtjern_hypsometry.csv", "observations/NO_Lan/Langtjern_temperature.csv")