import sys
import run_myLake_ISIMIP
import myLake_post
"""
Module for simulation optimisation.
"""

#Parameters to optimise:
param = []

def optimise_lake(lake_name, observation_path, input_directory, region, forcing_data_directory, outdir, modelid, scenarioid):
    """
    This function launches simulations with a number of predefined parameters, calls post-processing functions for each
    run, and records the results in a log file for visualisation adn further optimisation.
    :param lake_name:
    :param forcing_data_directory:
    :param outdir:
    :return:
    """
    orig_stdout = sys.stdout

    with open("{}/optimisation_log.txt".format(outdir), "w") as log:
        sys.stdout = log

        for x in range(y): #find lenght of range
            run_myLake_ISIMIP.mylakepar(run_myLake_ISIMIP.get_longitude(lake_name, forcing_data_directory),
                                        run_myLake_ISIMIP.get_latitude(lake_name, forcing_data_directory),
                                        lake_name,outdir, param[x])

            run_myLake_ISIMIP.run_myLake(observation_path, input_directory, region, lake_name, modelid, scenarioid)

            myLake_post.temperatures_by_depth(observation_path, lake_name, outdir)

            myLake_post.make_comparison_file(outdir)

            myLake_post.performance_analysis(outdir)

    sys.stdout = orig_stdout