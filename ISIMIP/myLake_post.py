import csv
import os
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
                if x <= 31:
                    date = int(str(start_year) + "01" + str(x))
                elif 31 < x <= 60:
                    date = int(str(start_year) + "02" + str(x))
                elif 60 < x <= 91:
                    date = int(str(start_year) + "03" + str(x))
                elif 91 < x <= 121:
                    date = int(str(start_year) + "04" + str(x))
                elif 121 < x <= 152:
                    date = int(str(start_year) + "05" + str(x))
                elif 152 < x <= 182:
                    date = int(str(start_year) + "06" + str(x))
                elif 182 < x <= 213:
                    date = int(str(start_year) + "07" + str(x))
                elif 213 < x <= 243:
                    date = int(str(start_year) + "08" + str(x))
                elif 243 < x <= 274:
                    date = int(str(start_year) + "09" + str(x))
                elif 274 < x <= 305:
                    date = int(str(start_year) + "10" + str(x))
                elif 305 < x <= 335:
                    date = int(str(start_year) + "11" + str(x))
                elif 335 < x <= 366:
                    date = int(str(start_year) + "12" + str(x))

                date_list.append(date)
        else:
            for x in range(1, 366):
                date = 0
                if x <= 31:
                    date = int(str(start_year) + "01" + str(x))
                elif 31 < x <= 59:
                    date = int(str(start_year) + "02" + str(x))
                elif 59 < x <= 90:
                    date = int(str(start_year) + "03" + str(x))
                elif 90 < x <= 120:
                    date = int(str(start_year) + "04" + str(x))
                elif 120 < x <= 151:
                    date = int(str(start_year) + "05" + str(x))
                elif 151 < x <= 181:
                    date = int(str(start_year) + "06" + str(x))
                elif 181 < x <= 212:
                    date = int(str(start_year) + "07" + str(x))
                elif 212 < x <= 242:
                    date = int(str(start_year) + "08" + str(x))
                elif 242 < x <= 273:
                    date = int(str(start_year) + "09" + str(x))
                elif 273 < x <= 304:
                    date = int(str(start_year) + "10" + str(x))
                elif 304 < x <= 334:
                    date = int(str(start_year) + "11" + str(x))
                elif 334 < x <= 365:
                    date = int(str(start_year) + "12" + str(x))

                date_list.append(date)

    return date_list


def make_comparison_file(output_folder):
    with open("{}/T_comparison.csv".format(output_folder), "w") as file:
        observation_dict = {}
        simulation_dict = {}
        depth_levels = []

        with open("{}/Observed_Temperatures.csv".format(output_folder), "r") as observation_file:
            reader = csv.reader(observation_file)
            depth_levels.append(reader[0][2])
            depth_levels.append(reader[0][-2])
            start_year = reader[0][2][:4]
            end_year = reader[-1][2][:4]

            for obs in reader[1:]:
                observation_dict[obs[0]] = [obs[2], obs[-2]]

        sims_dates = get_dates_of_simulation(start_year, end_year)

        with open("{}/Tzt.csv".format(output_folder), "r") as simulation_file:
            reader = csv.reader(simulation_file)
            for sim, date in reader, sims_dates:
                simulation_dict[date] = [sim[1], sim[-2]]

        csvFile = csv.reader(file)
        csvFile.writerow("Date, Observations:,  {}, {}, Simulations:, {}, {}".format(depth_levels[0], depth_levels[1], depth_levels[0], depth_levels[1]))
        for date in sims_dates:
            csvFile.writerow("{},, {}, {},, {}, {}".format(date, observation_dict[date][0], observation_dict[date]))


if __name__ == "__main__":
    temperatures_by_depth("observations/NO_Lan", "Langtjern", "output/NO/Langtjern")