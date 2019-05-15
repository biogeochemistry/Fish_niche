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


    with open("{}_temperature.csv".format(observation_path)) as obs_file:
        reader = list(csv.reader(obs_file))[1:]

        with open ("{}\\observed_temp.csv".format(output_folder), "w") as csvfile:
            out = csv.writer(csvfile)
            out.writerow(range(maxDepth + 1))
            row = {}

            for i in range(maxDepth + 1):
                row[i] = []

            for observation in reader:
                for i in range(maxDepth + 1):
                    if int(observation[3]) == i:
                        row[i].append(float(observation[4]))
                        break
                    elif int(observation[3]) > i:
                        row[i].append("")
                    elif int(observation[3]) < i:
                        row[i].append("")

            temp_list = []
            for i in range(len(row[0])):
                for x in range(maxDepth + 1):
                    temp_list.append(row[x])
                out.writerow(temp_list)
                temp_list.clear()


if __name__ == "__main__":
    temperatures_by_depth("observations/NO_Lan", "Langtjern", "output/NO/Langtjern")