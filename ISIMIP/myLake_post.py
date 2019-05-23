import csv
import os
from math import sqrt
import numpy
import datetime
import netCDF4 as ncdf

"Post-processing script for myLake simulations. For ISIMIP."

def temperatures_by_depth(observation_folder, lakeName, output_folder):
    """
    Creates a new csv file with the observed temperatures separated in columns by depths.
    :param observation_folder:
    :param lakeName:
    :param output_folder:
    :return:
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

def sums_of_squares(obs_list_1, obs_list_2, sims_list_1, sims_list_2):
    sum = 0
    for x in range(len(obs_list_1)):
        sum += (float(obs_list_1[x]) - float(sims_list_1[x]))**2 + (float(obs_list_2[x]) - float(sims_list_2[x]))**2

    return sum

def root_mean_square(obs_list_1, obs_list_2, sims_list_1, sims_list_2):
    lenght = len(obs_list_1) + len(obs_list_2) + len(sims_list_1) + len(sims_list_2)
    return sqrt(sums_of_squares(obs_list_1, obs_list_2, sims_list_1, sims_list_2)/lenght)

if __name__ == "__main__":
    #temperatures_by_depth("observations/NO_Lan", "Langtjern", "output/NO/Langtjern")
    #make_comparison_file("output/NO/Langtjern")
    performance_analysis("output/NO/Langtjern")