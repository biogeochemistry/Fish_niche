import csv
import os
from math import sqrt
import sys
import numpy
import datetime
import netCDF4 as ncdf
from sklearn import linear_model
import run_myLake_ISIMIP

"Post-processing script for myLake simulations. For ISIMIP."

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

        with open("{}/observed_temp.csv".format(output_folder), "r") as observation_file:
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

def performance_analysis(output_folder):
    """
    Opens the comparison file created by make_comparison_file, and prints the results of analysis functions.
    :param output_folder: A string, containing the folder containing the comparison file.
    :return: None
    """
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

    print("Analysis of {}.".format(output_folder[10:]))
    print("Sums of squares : {}".format(sums_of_squares(obs_list_1, obs_list_2, sims_list_1, sims_list_2)))
    print("RMSE : {}".format(root_mean_square(obs_list_1, obs_list_2, sims_list_1, sims_list_2)))
    print("R squared : {}".format(r_squared(date_list, obs_list_1, obs_list_2, sims_list_1, sims_list_2)))

def sums_of_squares(obs_list_1, obs_list_2, sims_list_1, sims_list_2):
    """
    Finds the sums of squares for all temperatures listed in the comparison file.
    :param obs_list_1: A list of observed temperatures.
    :param obs_list_2: A list of observed temperatures.
    :param sims_list_1: A list of simulated temperatures.
    :param sims_list_2: A list of simulated temperatures.
    :return: The result of the sums of square as a float.
    """
    sum = 0
    for x in range(len(obs_list_1)):
        sum += (float(obs_list_1[x]) - float(sims_list_1[x]))**2 + (float(obs_list_2[x]) - float(sims_list_2[x]))**2

    return sum

def root_mean_square(obs_list_1, obs_list_2, sims_list_1, sims_list_2):
    """
    Finds the root_mean_square for the temperatures listed in the comparison file.
    :param obs_list_1: A list of observed temperatures.
    :param obs_list_2: A list of observed temperatures.
    :param sims_list_1: A list of simulated temperatures.
    :param sims_list_2: A list of simulated temperatures.
    :return: The result of the root mean square as a float.
    """
    lenght = len(obs_list_1) + len(obs_list_2) + len(sims_list_1) + len(sims_list_2)
    return sqrt(sums_of_squares(obs_list_1, obs_list_2, sims_list_1, sims_list_2)/lenght)

def r_squared(dates, obs_list_1, obs_list_2, sims_list_1, sims_list_2):
    """
    Find the R squared for the simulations compared to the expected observations
    :param dates: the list of dates
    :param obs_list_1: A list of observed temperatures.
    :param obs_list_2: A list of observed temperatures.
    :param sims_list_1: A list of simulated temperatures.
    :param sims_list_2: A list of simulated temperatures.
    :return: r squared, as a float
    """
    linear = linear_model.LinearRegression()

    obs = obs_list_1
    sims = sims_list_1

    for i in range(len(obs_list_2)):
        obs.append(obs_list_2[i])
        sims.append(sims_list_2[i])

    x = []
    y = []
    date_index = 0

    for i in obs:
        if date_index == 0:
            x.append([float(dates[obs.index(i)]), float(i)])
            if obs.index(i) == dates.index(dates[-1]): date_index = 1
        else:
            x.append([float(dates[obs.index(i) - len(obs)//2]), float(i)])

    date_index = 0

    for i in sims:
        if date_index == 0:
            y.append([float(dates[sims.index(i)]), float(i)])
            if sims.index(i) == dates.index(dates[-1]): date_index = 1
        else:
            y.append([float(dates[sims.index(i) - len(sims)//2]), float(i)])

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

        for x in range(y): #find lenght of range
            run_myLake_ISIMIP.mylakepar(run_myLake_ISIMIP.get_longitude(lake_name, forcing_data_directory),
                                        run_myLake_ISIMIP.get_latitude(lake_name, forcing_data_directory),
                                        lake_name,outdir, param[x])

            run_myLake_ISIMIP.run_myLake(observation_path, input_directory, region, lake_name, modelid, scenarioid)

            temperatures_by_depth(observation_path, lake_name, outdir)

            make_comparison_file(outdir)

            performance_analysis(outdir)

    sys.stdout = orig_stdout


if __name__ == "__main__":
    #temperatures_by_depth("observations/NO_Lan", "Langtjern", "output/NO/Langtjern")
    #make_comparison_file("output/NO/Langtjern")
    performance_analysis("output/NO/Langtjern")