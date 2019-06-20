import csv
import os
from math import sqrt
import sys
import numpy as np
import datetime
import netCDF4 as ncdf
import datetime
from sklearn import linear_model
from scipy.optimize import minimize
import run_myLake_ISIMIP

"Post-processing script for myLake simulations. For ISIMIP."

test_cases = 10
parameters = {"Swa_b0" : [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
              "Swa_b1" : [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
              "C_shelter" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              "I_ScV" : [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8],
              "I_ScT" : [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8],
              "Alb_melt_ice" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              "Alb_melt_snow" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

def temperatures_by_depth(observation_folder, lakeName, output_folder):
    """
    Creates a new csv file with the observed temperatures separated in columns by depths.
    :param observation_folder: String
    :param lakeName: String
    :param output_folder: String
    :return: None
    """
    observation_path = os.path.join(observation_folder, lakeName)

    with open("{}_hypsometry.csv".format(observation_path)) as bathymetric_file:
        maxDepth = int(list(csv.reader(bathymetric_file))[-1][2])
        depthLevels = list(range(maxDepth + 1))


    with open("{}_temperature.csv".format(observation_path)) as obs_file:
        reader = list(csv.reader(obs_file))[1:]


        with open ("{}\\Observed_Temperatures.csv".format(output_folder), "w", newline = '') as csvfile:

            header = "Date, {}\n".format(depthLevels)
            csvfile.write(header.translate({ord(i): None for i in '[]'}))

            out = csv.writer(csvfile)
            rows = {}
            dates = []
            for i in depthLevels:
                rows[i] = []

            for observation in reader:
                if int(observation[2]) not in dates:
                    dates.append(int(observation[2]))

            temp_list = []
            for date in dates:
                for observation in reader:
                    if int(observation[2]) == date:
                        temp_list.append(observation)

                for depth in depthLevels:

                    missing_temp = True

                    for t in temp_list:
                        if int(t[3]) == depth:
                            rows[depth].append(float(t[4]))
                            missing_temp = False

                    if missing_temp:
                        rows[depth].append("None")

                temp_list.clear()

            temp_list.clear()
            for date in dates:
                temp_list.append(date)
                for x in depthLevels:
                     temp_list.append(rows[x][dates.index(date)])
                out.writerow(temp_list)
                temp_list.clear()

def get_dates_of_simulation(start_year, stop_year):
    """
    Finds the dates for each day of simulation. The dates are found by beginning at the first of january of the given
    start year and adding a day until the 31st of december of the stop year, accounting for leap years.
    :param start_year: An integer, the chosen year for the beginning of the simulation.
    :param stop_year: An integer, the chosen year for the end of the simulation.
    :return: A list of all the dates of simulation, in order from first to last. Dates are integers in the form YYYYMMDD.
    """
    date_list = []
    for i in range(start_year, stop_year + 1):
        if i % 400 == 0 or (i % 4 == 0 and i % 100 != 0):
            for x in range(1, 367):
                date = 0
                str_x = str(x)

                if x <= 31:
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "01" + str_x)
                elif 31 < x <= 60:
                    str_x = str(int(str_x) - 31)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "02" + str_x)
                elif 60 < x <= 91:
                    str_x = str(int(str_x) - 60)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "03" + str_x)
                elif 91 < x <= 121:
                    str_x = str(int(str_x) - 91)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "04" + str_x)
                elif 121 < x <= 152:
                    str_x = str(int(str_x) - 121)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "05" + str_x)
                elif 152 < x <= 182:
                    str_x = str(int(str_x) - 152)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "06" + str_x)
                elif 182 < x <= 213:
                    str_x = str(int(str_x) - 182)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "07" + str_x)
                elif 213 < x <= 243:
                    str_x = str(int(str_x) - 213)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "08" + str_x)
                elif 243 < x <= 274:
                    str_x = str(int(str_x) - 243)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "09" + str_x)
                elif 274 < x <= 305:
                    str_x = str(int(str_x) - 274)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "10" + str_x)
                elif 305 < x <= 335:
                    str_x = str(int(str_x) - 305)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "11" + str_x)
                elif 335 < x <= 366:
                    str_x = str(int(str_x) - 335)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "12" + str_x)

                date_list.append(date)
        else:
            for x in range(1, 366):
                date = 0
                str_x = str(x)

                if x <= 31:
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "01" + str_x)
                elif 31 < x <= 59:
                    str_x = str(int(str_x) - 31)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "02" + str_x)
                elif 59 < x <= 90:
                    str_x = str(int(str_x) - 59)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "03" + str_x)
                elif 90 < x <= 120:
                    str_x = str(int(str_x) - 90)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "04" + str_x)
                elif 120 < x <= 151:
                    str_x = str(int(str_x) - 120)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "05" + str_x)
                elif 151 < x <= 181:
                    str_x = str(int(str_x) - 151)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "06" + str_x)
                elif 181 < x <= 212:
                    str_x = str(int(str_x) - 181)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "07" + str_x)
                elif 212 < x <= 242:
                    str_x = str(int(str_x) - 212)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "08" + str_x)
                elif 242 < x <= 273:
                    str_x = str(int(str_x) - 242)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "09" + str_x)
                elif 273 < x <= 304:
                    str_x = str(int(str_x) - 273)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "10" + str_x)
                elif 304 < x <= 334:
                    str_x = str(int(str_x) - 304)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "11" + str_x)
                elif 334 < x <= 365:
                    str_x = str(int(str_x) - 334)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(start_year) + "12" + str_x)

                date_list.append(date)

    return date_list


def make_comparison_file(output_folder):
    """
    Search a given output folder for an observation file, containing mesured temperatures for a lake on a finite period,
    and a simulated temperatures file. Then writes corresponding observed and simulated temperatures to a csv file,
    where each column is a list of temperatures for a given depth.
    :param output_folder: A string containing the folder to search and write to.
    :return: None
    """
    with open("{}/T_comparison.csv".format(output_folder), "w", newline="\n") as file:
        observation_dict = {}
        simulation_dict = {}
        depth_levels = []

        with open("{}/Observed_Temperatures.csv".format(output_folder), "r") as observation_file:
            reader = list(csv.reader(observation_file))
            depth_levels.append(reader[0][2])
            depth_levels.append(reader[0][-2])
            start_year = int(reader[1][0][:4])
            end_year = int(reader[-1][0][:4])

            for obs in reader[1:]:
                observation_dict[int(obs[0])] = [obs[2], obs[-2]]

        sims_dates = get_dates_of_simulation(start_year, end_year)

        with open("{}/Tzt.csv".format(output_folder), "r") as simulation_file:
            reader = list(csv.reader(simulation_file))
            for sim in reader:
                simulation_dict[sims_dates[reader.index(sim)]] = [sim[1], sim[-2]]

        csvFile = csv.writer(file)
        csvFile.writerow(["Date", "Observations",  depth_levels[0], depth_levels[1], "Simulations", depth_levels[0], depth_levels[1]])
        for date in sims_dates:
            if date in observation_dict.keys():
                csvFile.writerow([date, '', observation_dict[date][0], observation_dict[date][1], '', simulation_dict[date][0], simulation_dict[date][1]])

def performance_analysis(lake_name, input_folder, output_folder):
    """
    Opens the comparison file created by make_comparison_file, and prints the results of analysis functions.
    :param output_folder: A string, containing the folder containing the comparison file.
    :return: None
    """

    with open("{}/{}_par".format(input_folder, lake_name[:3]), "r") as param_file:
        for line in param_file.readlines():
            if "C_shelter" in line:
                if float(line.split('\t')[1]) < 0: return 10000
            elif "swa_b0" in line:
                if float(line.split('\t')[1]) < 0: return 10000
            elif "swa_b01" in line:
                if float(line.split('\t')[1]) < 0: return 10000
            elif "I_scV" in line:
                if float(line.split('\t')[1]) < 0: return 10000
            elif "I_scT" in line:
                if float(line.split('\t')[1]) < 0: return 10000
            elif "alb_melt_ice" in line:
                if float(line.split('\t')[1]) < 0: return 10000
            elif "alb_melt_snow" in line:
                if float(line.split('\t')[1]) < 0: return 10000


    with open("{}/T_comparison.csv".format(output_folder), "r") as file:
        reader = list(csv.reader(file))

        date_list = []
        obs_list_1 = []
        obs_list_2 = []
        sims_list_1 = []
        sims_list_2 = []
        depth_levels = [reader[0][2], reader[0][3]]

    for item in reader[1:]:
        date_list.append(item[0])
        obs_list_1.append(item[2])
        obs_list_2.append(item[3])
        sims_list_1.append(item[5])
        sims_list_2.append(item[6])

    sos1 = sums_of_squares(obs_list_1, sims_list_1)
    sos2 = sums_of_squares(obs_list_2, sims_list_2)
    rms1 = root_mean_square(obs_list_1, sims_list_1)
    rms2 = root_mean_square(obs_list_2, sims_list_2)
    r_squ1 = r_squared(date_list, obs_list_1, sims_list_1)
    r_squ2 = r_squared(date_list, obs_list_2, sims_list_2)
    score = (sos1 + sos2)/100 + (rms1 + rms2 )* 100 + (((1 - r_squ1) * 1000 + (1 - r_squ2)) * 1000)

    print("Analysis of {}.".format(output_folder[10:]))
    print("Sums of squares : {}, {}".format(sos1, sos2))
    print("RMSE : {}, {}".format(rms1, rms2))
    print("R squared : {}, {}".format(r_squ1, r_squ2))
    print("Score : {}".format(score))

    return score

def sums_of_squares(obs_list, sims_list):
    """
    Finds the sums of squares for all temperatures listed in the comparison file.
    :param obs_list: A list of observed temperatures.
    :param sims_list: A list of simulated temperatures.
    :return: The result of the sums of square as a float.
    """
    sum = 0
    for x in range(len(obs_list)):
        sum += (float(obs_list[x]) - float(sims_list[x]))**2

    return sum

def root_mean_square(obs_list, sims_list):
    """
    Finds the root_mean_square for the temperatures listed in the comparison file.
    :param obs_list: A list of observed temperatures.
    :param sims_list: A list of simulated temperatures.
    :return: The result of the root mean square as a float.
    """
    lenght = len(obs_list) + len(sims_list)
    return sqrt(sums_of_squares(obs_list, sims_list)/lenght)

def r_squared(dates, obs_list, sims_list):
    """
    Find the R squared for the simulations compared to the expected observations
    :param dates: the list of dates
    :param obs_list: A list of observed temperatures.
    :param sims_list: A list of simulated temperatures.
    :return: results of R squared, as a float
    """
    linear = linear_model.LinearRegression()

    x = []
    y = []

    for i in obs_list:
        try:
            x.append([float(dates[obs_list.index(i)]), float(i)])
        except IndexError: break

    for i in sims_list:
        try:
            y.append([float(dates[sims_list.index(i)]), float(i)])
        except IndexError : break

    linear.fit(x, y)

    return linear.score(x, y)

def optimise_lake(lake_name, observation_path, input_directory, region, forcing_data_directory, outdir, modelid, scenarioid):
    """
    This function launches simulations with a number of predefined parameters, calls post-processing functions for each
    run, and records the results in a log file for visualisation and further optimisation.
    :param lake_name:
    :param forcing_data_directory:
    :param outdir:
    :return:
    """

    orig_stdout = sys.stdout

    with open("{}/optimisation_log.txt".format(outdir), "w") as log:
        sys.stdout = log

        print("Optimisation log for {}.\n\n".format(lake_name))

        for x in range(len(parameters)):
            for y in range(test_cases):
                print("Run # {}_{} at {}, using the following parameters:".format(x, y, datetime.datetime.now()))
                if x is 0:
                    print("C_shelter: {}\tAlb_melt_ice: {}\tAlb_melt_snow: {}\tI_ScV: {}\tI_ScT: {}\tSwa_b0: {}\tSwa_b1: {}".format(
                        parameters["C_shelter"][y],
                        parameters["Alb_melt_ice"][x],
                        parameters["Alb_melt_snow"][x],
                        parameters["I_ScV"][x],
                        parameters["I_ScT"][x],
                        parameters["Swa_b0"][x],
                        parameters["Swa_b1"][x]))

                    run_myLake_ISIMIP.mylakepar(run_myLake_ISIMIP.get_longitude(lake_name, forcing_data_directory),
                                                run_myLake_ISIMIP.get_latitude(lake_name, forcing_data_directory),
                                                lake_name,input_directory,
                                                parameters["C_shelter"][y],
                                                parameters["Alb_melt_ice"][x],
                                                parameters["Alb_melt_snow"][x],
                                                parameters["I_ScV"][x],
                                                parameters["I_ScT"][x],
                                                parameters["Swa_b0"][x],
                                                parameters["Swa_b1"][x]
                                                )

                elif x is 1:
                    print("C_shelter: {}\tAlb_melt_ice: {}\tAlb_melt_snow: {}\tI_ScV: {}\tI_ScT: {}\tSwa_b0: {}\tSwa_b1: {}".format(
                        parameters["C_shelter"][x],
                        parameters["Alb_melt_ice"][y],
                        parameters["Alb_melt_snow"][x],
                        parameters["I_ScV"][x],
                        parameters["I_ScT"][x],
                        parameters["Swa_b0"][x],
                        parameters["Swa_b1"][x]))

                    run_myLake_ISIMIP.mylakepar(run_myLake_ISIMIP.get_longitude(lake_name, forcing_data_directory),
                                                run_myLake_ISIMIP.get_latitude(lake_name, forcing_data_directory),
                                                lake_name, input_directory,
                                                parameters["C_shelter"][x],
                                                parameters["Alb_melt_ice"][y],
                                                parameters["Alb_melt_snow"][x],
                                                parameters["I_ScV"][x],
                                                parameters["I_ScT"][x],
                                                parameters["Swa_b0"][x],
                                                parameters["Swa_b1"][x]
                                                )
                elif x is 2:
                    print("C_shelter: {}\tAlb_melt_ice: {}\tAlb_melt_snow: {}\tI_ScV: {}\tI_ScT: {}\tSwa_b0: {}\tSwa_b1: {}".format(
                        parameters["C_shelter"][x],
                        parameters["Alb_melt_ice"][x],
                        parameters["Alb_melt_snow"][y],
                        parameters["I_ScV"][x],
                        parameters["I_ScT"][x],
                        parameters["Swa_b0"][x],
                        parameters["Swa_b1"][x]))

                    run_myLake_ISIMIP.mylakepar(run_myLake_ISIMIP.get_longitude(lake_name, forcing_data_directory),
                                                run_myLake_ISIMIP.get_latitude(lake_name, forcing_data_directory),
                                                lake_name, input_directory,
                                                parameters["C_shelter"][x],
                                                parameters["Alb_melt_ice"][x],
                                                parameters["Alb_melt_snow"][y],
                                                parameters["I_ScV"][x],
                                                parameters["I_ScT"][x],
                                                parameters["Swa_b0"][x],
                                                parameters["Swa_b1"][x]
                                                )
                elif x is 3:
                    print("C_shelter: {}\tAlb_melt_ice: {}\tAlb_melt_snow: {}\tI_ScV: {}\tI_ScT: {}\tSwa_b0: {}\tSwa_b1: {}".format(
                        parameters["C_shelter"][x],
                        parameters["Alb_melt_ice"][x],
                        parameters["Alb_melt_snow"][x],
                        parameters["I_ScV"][y],
                        parameters["I_ScT"][x],
                        parameters["Swa_b0"][x],
                        parameters["Swa_b1"][x]))

                    run_myLake_ISIMIP.mylakepar(run_myLake_ISIMIP.get_longitude(lake_name, forcing_data_directory),
                                                run_myLake_ISIMIP.get_latitude(lake_name, forcing_data_directory),
                                                lake_name, input_directory,
                                                parameters["C_shelter"][x],
                                                parameters["Alb_melt_ice"][x],
                                                parameters["Alb_melt_snow"][x],
                                                parameters["I_ScV"][y],
                                                parameters["I_ScT"][x],
                                                parameters["Swa_b0"][x],
                                                parameters["Swa_b1"][x]
                                                )
                elif x is 4:
                    print("C_shelter: {}\tAlb_melt_ice: {}\tAlb_melt_snow: {}\tI_ScV: {}\tI_ScT: {}\tSwa_b0: {}\tSwa_b1: {}".format(
                        parameters["C_shelter"][x],
                        parameters["Alb_melt_ice"][x],
                        parameters["Alb_melt_snow"][x],
                        parameters["I_ScV"][x],
                        parameters["I_ScT"][y],
                        parameters["Swa_b0"][x],
                        parameters["Swa_b1"][x]))

                    run_myLake_ISIMIP.mylakepar(run_myLake_ISIMIP.get_longitude(lake_name, forcing_data_directory),
                                                run_myLake_ISIMIP.get_latitude(lake_name, forcing_data_directory),
                                                lake_name, input_directory,
                                                parameters["C_shelter"][x],
                                                parameters["Alb_melt_ice"][x],
                                                parameters["Alb_melt_snow"][x],
                                                parameters["I_ScV"][x],
                                                parameters["I_ScT"][y],
                                                parameters["Swa_b0"][x],
                                                parameters["Swa_b1"][x]
                                                )
                elif x is 5:
                    print("C_shelter: {}\tAlb_melt_ice: {}\tAlb_melt_snow: {}\tI_ScV: {}\tI_ScT: {}\tSwa_b0: {}\tSwa_b1: {}".format(
                            parameters["C_shelter"][x],
                            parameters["Alb_melt_ice"][x],
                            parameters["Alb_melt_snow"][x],
                            parameters["I_ScV"][x],
                            parameters["I_ScT"][x],
                            parameters["Swa_b0"][y],
                            parameters["Swa_b1"][x]))

                    run_myLake_ISIMIP.mylakepar(run_myLake_ISIMIP.get_longitude(lake_name, forcing_data_directory),
                                                run_myLake_ISIMIP.get_latitude(lake_name, forcing_data_directory),
                                                lake_name, input_directory,
                                                parameters["C_shelter"][x],
                                                parameters["Alb_melt_ice"][x],
                                                parameters["Alb_melt_snow"][x],
                                                parameters["I_ScV"][x],
                                                parameters["I_ScT"][x],
                                                parameters["Swa_b0"][y],
                                                parameters["Swa_b1"][x]
                                                )
                elif x is 6:
                    print("C_shelter: {}\tAlb_melt_ice: {}\tAlb_melt_snow: {}\tI_ScV: {}\tI_ScT: {}\tSwa_b0: {}\tSwa_b1: {}".format(
                        parameters["C_shelter"][x],
                        parameters["Alb_melt_ice"][x],
                        parameters["Alb_melt_snow"][x],
                        parameters["I_ScV"][x],
                        parameters["I_ScT"][x],
                        parameters["Swa_b0"][x],
                        parameters["Swa_b1"][y]))

                    run_myLake_ISIMIP.mylakepar(run_myLake_ISIMIP.get_longitude(lake_name, forcing_data_directory),
                                                run_myLake_ISIMIP.get_latitude(lake_name, forcing_data_directory),
                                                lake_name, input_directory,
                                                parameters["C_shelter"][x],
                                                parameters["Alb_melt_ice"][x],
                                                parameters["Alb_melt_snow"][x],
                                                parameters["I_ScV"][x],
                                                parameters["I_ScT"][x],
                                                parameters["Swa_b0"][x],
                                                parameters["Swa_b1"][y]
                                                )

                try:

                    run_myLake_ISIMIP.run_myLake(observation_path, input_directory, region, lake_name, modelid, scenarioid)

                    temperatures_by_depth(observation_path, lake_name, outdir)

                    make_comparison_file(outdir)

                    performance_analysis(outdir)

                    print("\nEnd of case.\n\n")

                except KeyError:
                    print("\nCould not run MyLake simulation with these parameters.\nEnd of case.\n\n")
                except IndexError:
                    print("\nError in optimisation function.\nEnd of case.\n\n")

    print("Optimisation results: \n")

    find_best_parameters("{}/optimisation_log.txt".format(outdir))
    sys.stdout = orig_stdout

def make_optimisation_dictionnary(log_file):
    """
    Reads the log file and makes a dictionary where the keys are a string containing the parameters for the given
     simulation, and the values are lists of 3 floats, containing the results of sums of squares, RMSE and r squared for
     the simulation, always in that sequence.
    :param log_file: Type string. The log file containing the information on all calibration runs.
    :return: The comparison dictionary
    """
    comparison_dict = {}
    with open(log_file, "r") as log:
        file_content = log.readlines()

        opt_list = []
        param = ""
        for line in file_content:
            if "C_shelter" in line:
                param = line

            elif "Sums of squares" in line:
                opt_list.append(float(line[18:]))

            elif "RMSE" in line:
                opt_list.append(float(line[7:]))

            elif "R squared" in line:
                opt_list.append(float(line[12:]))

            if param != "" and len(opt_list) == 3:
                comparison_dict[param] = opt_list
                param = ""
                opt_list = []

            elif "End of case" in line:
                param = ""
                opt_list = []


    return comparison_dict

def find_best_parameters(log_file):
    """
    Using the comparison dictionary, finds the set of parameters giving the best sums of squares, best RMSE, best r
    squared and best overall performances.
    :param log_file: Type string. The log file containing the information on all calibration runs.
    :return: None
    """
    comparison_dict = make_optimisation_dictionnary(log_file)
    list_score = []
    best_SoS = ("", 1000000)
    best_RMSE = ("", 50)
    best_r_squared = ("", 0)
    best_score = ("", 99999999)

    for param in comparison_dict:
        if comparison_dict[param][0] < best_SoS[1]:
            best_SoS = (param, comparison_dict[param][0])
        if comparison_dict[param][1] < best_RMSE[1]:
            best_RMSE = (param, comparison_dict[param][1])
        if comparison_dict[param][2] > best_r_squared[1]:
            best_r_squared = (param, comparison_dict[param][2])

        score = comparison_dict[param][0]/100 + comparison_dict[param][1]*100 + (1 - comparison_dict[param][2]) * 1000
        list_score.append((param, score))

        if score < best_score[1]:
            best_score = (param, score)

    print("Best sums of squares: {}".format(best_SoS))
    print("Best RMSE: {}".format(best_RMSE))
    print("Best r squared: {}".format(best_r_squared))
    print("Best overall score: {}".format(best_score))

def optimize_Nelder_Meald(lake_name, observation_path, input_directory, region, forcing_data_directory, outdir, modelid, scenarioid):
    """
    Alternative optimizing function using scipy.optimize.minimize

    :param lake_name: Type string. The name of the lake to optimise
    :param observation_path: Type string. Observed temperatures file
    :param input_directory: Type string. The folder containing all input files (input, parameters, init)
    :param region: Type string. The abreviation for the lake's region (Ex: United-States = US, Norway = NO)
    :param forcing_data_directory: Type string. The folder containing all forcing data for the given lake
    :param outdir: Type string. The output folder
    :param modelid: Type string. Prediction model used.
    :param scenarioid: Type string. The prediction scenario. For optimization purpose this is always historical
    :return: None
    """

    func = lambda params: run_optimization_Mylake(lake_name, observation_path, input_directory, region, forcing_data_directory, outdir, modelid, scenarioid, params)
    params_0 = np.array([0, 0.3, 0.55, 1, 0, 2.5, 1])

    res = minimize(func, params_0, method= "nelder-mead", options={'xtol':0, 'disp': True})

    print(res)

    with open("{}/Calibration_Complete.txt".format(outdir), "w") as end_file:
        end_file.writelines(["Calibration results:", res])

    return res

def run_optimization_Mylake(lake_name, observation_path, input_directory, region, forcing_data_directory, outdir, modelid, scenarioid, params):
    """

    :param lake_name: Type string. The name of the lake to optimise
    :param observation_path: Type string. Observed temperatures file
    :param input_directory: Type string. The folder containing all input files (input, parameters, init)
    :param region: Type string. The abreviation for the lake's region (Ex: United-States = US, Norway = NO)
    :param forcing_data_directory: Type string. The folder containing all forcing data for the given lake
    :param outdir: Type string. The output folder
    :param modelid: Type string. Prediction model used.
    :param scenarioid: Type string. The prediction scenario. For optimization purpose this is always historical
    :param swa_b0:
    :param swa_b1:
    :param c_shelter:
    :param i_scv:
    :param i_sct:
    :param alb_melt_ice:
    :param alb_melt_snow:
    :return: performance analysis, which itself returns a score as float
    """

    c_shelter, alb_melt_snow, alb_melt_ice, i_scv, i_sct, swa_b0, swa_b1 = params

    run_myLake_ISIMIP.mylakepar(run_myLake_ISIMIP.get_longitude(lake_name, forcing_data_directory),
                                run_myLake_ISIMIP.get_latitude(lake_name, forcing_data_directory),
                                lake_name, input_directory, c_shelter, alb_melt_ice, alb_melt_snow, i_scv, i_sct, swa_b0, swa_b1)

    run_myLake_ISIMIP.run_myLake(observation_path, input_directory, region, lake_name, modelid, scenarioid)

    temperatures_by_depth(observation_path, lake_name, outdir)

    make_comparison_file(outdir)

    return performance_analysis(lake_name, input_directory, outdir)


if __name__ == "__main__":
    #temperatures_by_depth("observations/Langtjern", "Langtjern", "output/NO/Langtjern")
    #make_comparison_file("output/NO/Langtjern")
    #performance_analysis("output/NO/Langtjern")
    #optimise_lake("Langtjern", "observations/Langtjern", "input/NO/Lan", "NO", "forcing_data/Langtjern", "output/NO/Langtjern", "GFDL-ESM2M", "historical")
    #find_best_parameters("output/NO/Langtjern/optimisation_log.txt")
    optimize_Nelder_Meald("Langtjern", "observations/Langtjern", "input/NO/Lan", "NO", "forcing_data/Langtjern", "output/NO/Langtjern/GFDL-ESM2M/historical", "GFDL-ESM2M", "historical")