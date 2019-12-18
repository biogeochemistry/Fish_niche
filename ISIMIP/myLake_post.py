import csv
import os
from math import sqrt
import sys
import numpy as np
import statistics
import netCDF4 as ncdf
import datetime
from multiprocessing import Pool
from sklearn.metrics import r2_score
from scipy.optimize import minimize, differential_evolution
import run_myLake_ISIMIP
import pandas as pd

"Post-processing script for myLake simulations. For ISIMIP."

test_cases = 10
parameters = {"Swa_b0" : [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
              "Swa_b1" : [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
              "C_shelter" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              "I_ScV" : [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8],
              "I_ScT" : [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8],
              "Alb_melt_ice" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              "Alb_melt_snow" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}




#full_lake_list = "Allequash"
regions = {"US": ["Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal", "CrystalBog", "Delavan",
                  "Falling_Creek", "Fish", "Great_Pond", "Green_lake", "Laramie", "Mendota", "Monona",
                  "Okauchee", "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik", "Trout", "TroutBog", "TwoSisters",
                  "Washington", "Wingra"],
           "CH": ["Biel", "LowerZurich", "Neuchatel"],
           "PT": ["Alqueva"],
           "FR": ["Annecy", "Bourget", "Geneva"],
           "AU": ["Argyle", "BurleyGriffin", "MtBold"],
           "CA": ["Dickie", "Eagle_lake", "Harp"],
           "SE": ["Ekoln", "Erken"],
           "UK": ["EsthwaiteWater", "Windermere"],
           "IE": ["Feeagh"],
           "FI": ["Kilpisjarvi", "Kuivajarvi", "Paajarvi"],
           "IL": ["Kinneret"],
           "RW": ["Kivu"],
           "CZ": ["Klicava", "Rimov", "Zlutice"],
           "NO": ["Langtjern"],
           "RU": ["Mozhaysk", "Vendyurskoe"],
           "DE": ["Muggelsee", "Rappbode", "Stechlin"],
           "CN": ["Ngoring"],
           "EE": ["NohipaloMustjarv", "NohipaloValgejarv", "Vortsjarv"],
           "ES": ["Sau"],
           "NZ": ["Rotorua", "Tarawera", "Taupo", "Waahi"]}
full_lake_list =["Argyle","Alqueva","Rotorua","Vendyurskoe","Muggelsee","EsthwaiteWater","Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal"
                , "Delavan","FallingCreek", "Fish", "Great_Pond",
                 "Green_lake", "Laramie", "Mendota", "Okauchee",
                 "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik",
                 "Trout", "TroutBog", "Two_Sisters","Washington", "Wingra",
                 "Biel", "LowerZurich", "Neuchatel","Annecy",
                 "Bourget" ,"BurleyGriffin","MtBold",
                 "Dickie","Eagle_lake" , "Harp","Ekoln", "Erken",
                 "EsthwaiteWater", "Windermere","Feeagh","Kilpisjarvi", "Kuivajarvi",
                 "Paajarvi","Kinneret","Kivu","Klicava", "Rimov",
                 "Zlutice","Langtjern","Mozhaysk","Rappbode", "Stechlin","Ngoring","NohipaloMustjarv", "NohipaloValgejarv","Vortsjarv","Sau", "Tarawera", "Taupo","Waahi"]
temp_by_lake = {"Argyle":[0.7,26.7],"Dickie":[1,9],"Harp":[1,27],"LowerZurich":[1,100],"Neuchatel":[0,100],
               "Rimov":[0.2,14.2],"Rappbode":[2.82,52.62],"Stechlin":[0,55],"Vortsjarv":[0.5,3],"Sau":[2,40],

               "Kilpisjarvi":[0.75,31],"Kuivajarvi":[0.2,12],"Paajarvi":[1,65],"Annecy":[10,61],"Feeagh":[0.9,42],
               "Kinneret":[2,34],"Langtjern":[1,8],"Rotorua":[2,16],"Tarawera":[2,80],"Sammamish":[1,20],

               "Kivu":[0,90],"Ekoln":[1,27.25], "Erken":[0.5,20],"EsthwaiteWater":[0.5,11.5],"Windermere":[1,35],
               "BlackOak":[0.9144,25.908],"Crystal":[1,17],"Laramie":[2,6],"Mendota":[0,20], "Okauchee":[1,25],

               "Sparkling":[0,17],"Sunapee":[1,9], "MtBold":[1,34],"NohipaloMustjarv":[0.5,6], "NohipaloValgejarv":[0.5,8],
                "BigMuskellunge":[1,18],"BurleyGriffin":[3,12],"Eagle_lake":[1,10], "Mozhaysk":[1,7],"Allequash":[1,7],

                "Biel":[5,70],"Muggelsee":[1,5],"Geneva":[3.6,263.6],"Bourget":[9.5,106],"Vendyurskoe":[4.58,11.3],
                "CrystalBog":[0.25,2],"Delavan":[1,15],"Fish":[1,17],"Green_lake":[1,66],"Toolik":[1,16],

                "Trout":[1,29],"TroutBog":[1,7],"Washington":[1,55],"Wingra":[1,3]}

def temperatures_by_depth(observation_folder, lakeName, output_folder,model,scenario):
    """
    Creates a new csv file with the observed temperatures separated in columns by depths.
    :param observation_folder: String
    :param lakeName: String
    :param output_folder: String
    :return: None
    """

    with open("{}/{}_hypsometry2.csv".format(observation_folder,lakeName)) as bathymetric_file:
        maxDepth = int(float(list(csv.reader(bathymetric_file))[-1][2]))
        depthLevels = list(range(maxDepth + 1))



    with open("{}/{}_temp_daily.csv".format(observation_folder,lakeName)) as obs_file:
        reader = list(csv.reader(obs_file))[1:]
        depthlist = []
        loop = 0
        for depth in reader:
            if float(depth[3]) not in depthlist:
                depthlist.append(float(depth[3]))
            elif float(depth[3]) == depthlist[0]:
                loop += 1
            if loop == 10:
                break
        outputdir2 = list(output_folder.split("/"))
        outputdir3 = 'output'
        for i in outputdir2:
            if i == 'output':
                outputdir3 = i
            else:
                if not os.path.exists(outputdir3):
                    os.mkdir(outputdir3)
                outputdir3 = os.path.join(outputdir3, i)
        if not os.path.exists(outputdir3):
            os.mkdir(outputdir3)



        with open ("{}/Observed_Temperatures.csv".format(output_folder), "w", newline = '') as csvfile:
            print("{}/Observed_Temperatures.csv".format(output_folder))
            header = "Date, {}\n".format(depthlist)
            csvfile.write(header.translate({ord(i): None for i in '[]'}))

            out = csv.writer(csvfile)
            rows = {}
            dates = []
            for i in depthlist:
                rows[i] = []

            for observation in reader:
                if int(observation[2]) not in dates:
                    dates.append(int(observation[2]))

            temp_list = []
            number = 0
            for date in dates:
                for observation in reader:
                    if int(observation[2]) == date:
                        temp_list.append(observation)

                for depth in depthlist:

                    missing_temp = True

                    for t in temp_list:

                        if float(t[3]) == depth:

                            if len(rows[depth]) <= number:
                                rows[depth].append(float(t[4]))
                                missing_temp = False


                    if missing_temp:
                        rows[depth].append("None")
                number += 1
                temp_list.clear()
            temp_list.clear()
            for date in dates:
                temp_list.append(date)
                for x in depthlist:
                     temp_list.append(rows[x][dates.index(date)])
                out.writerow(temp_list)
                temp_list.clear()
    print("observation done ... ... ... ... ")

def get_dates_of_simulation(start_year, stop_year):
    """
    Finds the dates for each day of simulation. The dates are found by beginning at the first of january of the given
    start year and adding a day until the 31st of december of the stop year, accounting for leap years.
    :param start_year: An integer, the chosen year for the beginning of the simulation.
    :param stop_year: An integer, the chosen year for the end of the simulation.
    :return: A list of all the dates of simulation, in order from first to last. Dates are integers in the form YYYYMMDD.
    """
    date_list = []
    year = start_year
    nb_year = stop_year - start_year

    for i in range(0, nb_year):
        if i % 400 == 0 or (i % 4 == 0 and i % 100 != 0):
            for x in range(1, 367):
                date = 0
                str_x = str(x)

                if x <= 31:
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "01" + str_x)
                elif 31 < x <= 60:
                    str_x = str(int(str_x) - 31)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "02" + str_x)
                elif 60 < x <= 91:
                    str_x = str(int(year) - 60)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "03" + str_x)
                elif 91 < x <= 121:
                    str_x = str(int(year) - 91)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "04" + str_x)
                elif 121 < x <= 152:
                    str_x = str(int(str_x) - 121)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "05" + str_x)
                elif 152 < x <= 182:
                    str_x = str(int(str_x) - 152)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "06" + str_x)
                elif 182 < x <= 213:
                    str_x = str(int(str_x) - 182)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "07" + str_x)
                elif 213 < x <= 243:
                    str_x = str(int(str_x) - 213)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "08" + str_x)
                elif 243 < x <= 274:
                    str_x = str(int(str_x) - 243)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "09" + str_x)
                elif 274 < x <= 305:
                    str_x = str(int(str_x) - 274)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "10" + str_x)
                elif 305 < x <= 335:
                    str_x = str(int(str_x) - 305)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "11" + str_x)
                elif 335 < x <= 366:
                    str_x = str(int(str_x) - 335)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "12" + str_x)

                date_list.append(date)
        else:
            for x in range(1, 366):
                date = 0
                str_x = str(x)

                if x <= 31:
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "01" + str_x)
                elif 31 < x <= 59:
                    str_x = str(int(str_x) - 31)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "02" + str_x)
                elif 59 < x <= 90:
                    str_x = str(int(str_x) - 59)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "03" + str_x)
                elif 90 < x <= 120:
                    str_x = str(int(str_x) - 90)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "04" + str_x)
                elif 120 < x <= 151:
                    str_x = str(int(str_x) - 120)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "05" + str_x)
                elif 151 < x <= 181:
                    str_x = str(int(str_x) - 151)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "06" + str_x)
                elif 181 < x <= 212:
                    str_x = str(int(str_x) - 181)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "07" + str_x)
                elif 212 < x <= 242:
                    str_x = str(int(str_x) - 212)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "08" + str_x)
                elif 242 < x <= 273:
                    str_x = str(int(str_x) - 242)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "09" + str_x)
                elif 273 < x <= 304:
                    str_x = str(int(str_x) - 273)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "10" + str_x)
                elif 304 < x <= 334:
                    str_x = str(int(str_x) - 304)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "11" + str_x)
                elif 334 < x <= 365:
                    str_x = str(int(str_x) - 334)
                    if len(str_x) == 1:
                        str_x = "0" + str_x
                    date = int(str(year) + "12" + str_x)

                date_list.append(date)
        year += 1
    return date_list


def findYPoint(xa,xb,ya,yb,xc):
    m = (float(ya) - float(yb)) / (float(xa) -float( xb))
    yc = (float(xc) - float(xb)) * m + float(yb)
    return yc
def make_comparison_file(output_folder,depths):
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
            for i in range(1,len(reader)):
                if float(reader[0][i]) == depths[0]:
                    depth_levels.append(reader[0][i])
                    col1= i
                elif float(reader[0][i]) == depths[1]:
                    depth_levels.append(reader[0][i])
                    col2=i
                if len(depth_levels) == 2:
                    break
            start_year = int(reader[1][0][:4])
            end_year = int(reader[-1][0][:4])

            for obs in reader[1:]:
                observation_dict[int(obs[0])] = [obs[col1], obs[col2]]

        sims_dates = get_dates_of_simulation(start_year, end_year)

        with open("{}/Tzt.csv".format(output_folder), "r") as simulation_file:
            reader = list(csv.reader(simulation_file))

            for sim in reader:
                try:

                    if int(float(depth_levels[0])) != float(depth_levels[0]):
                        xa = int(float(depth_levels[0]))
                        xb = xa + 1
                        if int(float(depth_levels[1])) != float(depth_levels[1]):
                            xa1 = int(float(depth_levels[1]))
                            xb1 = xa1 + 1
                            if sim[xa] != "None" and sim[xb] != "None":

                                valueyc = findYPoint(xa, xb, sim[xa], sim[xb], float(depth_levels[0]))
                                if sim[xa1] != "None" and sim[xb1] != "None":

                                    valueyc1 = findYPoint(xa1, xb1, sim[xa1], sim[xb1], float(depth_levels[1]))


                                    simulation_dict[sims_dates[reader.index(sim)]] = [valueyc, valueyc1]
                                else:
                                    simulation_dict[sims_dates[reader.index(sim)]] = [valueyc,"None"]

                            else:
                                if sim[xa1] != "None" and sim[xb1] != "None":
                                    valueyc1 = findYPoint(xa1, xb1, sim[xa1], sim[xb1], float(depth_levels[1]))

                                    simulation_dict[sims_dates[reader.index(sim)]] = ["None",valueyc1]
                                else:
                                    simulation_dict[sims_dates[reader.index(sim)]] = ["None", "None"]

                        else:
                            if sim[xa] != "None" and sim[xb] != "None":
                                valueyc = findYPoint(xa, xb, sim[xa], sim[xb], float(depth_levels[0]))

                                simulation_dict[sims_dates[reader.index(sim)]] = [valueyc, sim[col2]]
                            else:
                                simulation_dict[sims_dates[reader.index(sim)]] = ["None", sim[col2]]
                    else:
                        simulation_dict[sims_dates[reader.index(sim)]] = [sim[col1], sim[col2]]
                except IndexError:
                    continue

        csvFile = csv.writer(file)
        csvFile.writerow(["Date", "Observations",  depth_levels[0], depth_levels[1], "Simulations", depth_levels[0], depth_levels[1]])
        for date in sims_dates:
            if date in observation_dict.keys():
                try:

                    csvFile.writerow([date, '', observation_dict[date][0], observation_dict[date][1], '', simulation_dict[date][0], simulation_dict[date][1]])

                except KeyError:
                    continue
        return None

def make_comparison_file1(output_folder):
    """
    Search a given output folder for an observation file, containing mesured temperatures for a lake on a finite period,
    and a simulated temperatures file. Then writes corresponding observed and simulated temperatures to a csv file,
    where each column is a list of temperatures for a given depth.
    :param output_folder: A string containing the folder to search and write to.
    :return: None
    """
    with open("{}/T_comparison1.csv".format(output_folder), "w", newline="\n") as file:
        observation_dict,observation_dict1 = {},{}
        simulation_dict,simulation_dict1 = {},{}
        depth_levels = []

        with open("{}/Observed_Temperatures.csv".format(output_folder), "r") as observation_file:
            reader = list(csv.reader(observation_file))

            start_year = int(reader[1][0][:4])
            end_year = int(reader[-1][0][:4])
            sims_dates = get_dates_of_simulation(start_year, end_year)

            datacomparaison = False
            for j in range(2, len(reader[0])):
                for obs in reader[1:]:
                    observation_dict1[int(obs[0])] = [obs[j], obs[-1]]
                nbrNone = 0
                for date in sims_dates:
                    if date in observation_dict1.keys():
                        value = observation_dict1[date][0]
                        if observation_dict1[date][0] == "None":
                            nbrNone += 1

                            #if nbrNone > 150:
                                # break
                running = False
                if nbrNone/len(sims_dates) < 0.80:
                    depth_levels.append(reader[0][j])
                    depth_levels.append(reader[0][-1])
                    with open("{}/Tzt.csv".format(output_folder), "r") as simulation_file:
                        reader = list(csv.reader(simulation_file))
                    for sim in reader:
                        try:

                            if int(float(depth_levels[0])) != float(depth_levels[0]):
                                xa = int(float(depth_levels[0]))
                                xb = xa +1
                                if sim[xa] != "None" and sim[xb] != "None":
                                    valueyc = findYPoint(xa, xb, sim[xa], sim[xb], float(depth_levels[0]))
                                    simulation_dict1[sims_dates[reader.index(sim)]] = [valueyc, sim[-1]]
                                else:
                                    simulation_dict1[sims_dates[reader.index(sim)]] = ["None", sim[-1]]
                            else:
                                simulation_dict1[sims_dates[reader.index(sim)]] = [sim[j], sim[-1]]
                        except IndexError:
                            continue

                    csvFile = csv.writer(file)
                    csvFile.writerow(["Date", "Observations",  depth_levels[0], depth_levels[1], "Simulations", depth_levels[0], depth_levels[1]])
                    nbrNone= 0

                    running=True
                    for date in sims_dates:
                        if running == True:
                            if date in observation_dict1.keys():
                                try:
                                    csvFile.writerow([date, '', observation_dict1[date][0], observation_dict1[date][1], '', simulation_dict1[date][0], simulation_dict1[date][1]])

                                    if observation_dict1[date][0] == "None":
                                        nbrNone += 1

                                    if nbrNone > 150:
                                        nbrNone = 1/0

                                except KeyError:
                                    continue
                                except ZeroDivisionError:
                                     running = False
                        else:
                            break
                if running == True:
                    datacomparaison = True
                    break

        with open("{}/T_comparison.csv".format(output_folder), "w", newline="\n") as final:
            with open("{}/Observed_Temperatures.csv".format(output_folder), "r") as observation_file:
                reader = list(csv.reader(observation_file))

                start_year = int(reader[1][0][:4])
                end_year = int(reader[-1][0][:4])
                sims_dates = get_dates_of_simulation(start_year, end_year)

                if running == True and datacomparaison == True:
                    lenght=3
                    for obser in reader[1:]:
                        lenght = len(obser)
                        break
                    if (lenght - j)>2:
                        for i in range(2, (lenght-j)):
                            for obs in reader[1:]:
                                if float(obs[0]) == 0:

                                    observation_dict[0] = [obs[j], obs[-i]]
                                else:
                                    observation_dict[int(float(obs[0]))] = [obs[j], obs[-i]]
                            nbrNone = 0
                            for date in sims_dates:
                                if date in observation_dict.keys():
                                    value = observation_dict[date][1]
                                    if observation_dict[date][1] == "None":
                                        nbrNone += 1

                                    #if nbrNone > 150:
                                        # break
                            running = False
                            if nbrNone/len(sims_dates) < 0.50:
                                if reader[0][-i] > depth_levels[0]:
                                    depth_levels.append(reader[0][-i])
                                    with open("{}/Tzt.csv".format(output_folder), "r") as simulation_file:
                                        reader = list(csv.reader(simulation_file))
                                    for sim in reader:
                                        try:
                                            if int(float(depth_levels[2])) != float(depth_levels[2]):
                                                xa = int(float(depth_levels[2]))
                                                xb = xa + 1
                                                if sim[xa] != "None" and sim[xb] != "None":
                                                    valueyc = findYPoint(xa, xb, float(sim[xa]), float(sim[xb]), float(depth_levels[2]))
                                                    simulation_dict1[sims_dates[reader.index(sim)]] = [sim[2], valueyc]
                                                else:
                                                    simulation_dict1[sims_dates[reader.index(sim)]] = [sim[2], "None"]
                                            else:
                                                simulation_dict1[sims_dates[reader.index(sim)]] = [sim[2], sim[-1]]
                                        except IndexError:
                                            continue

                                    csvFile = csv.writer(final)
                                    csvFile.writerow(["Date", "Observations",  depth_levels[0], depth_levels[2], "Simulations", depth_levels[0], depth_levels[2]])
                                    nbrNone= 0
                                    running=True
                                    for date in sims_dates:
                                        if running == True:
                                            if date in observation_dict.keys():
                                                try:

                                                    csvFile.writerow([date, '', observation_dict1[date][0], observation_dict[date][1], '', simulation_dict1[date][0], simulation_dict1[date][1]])
                                                    if observation_dict[date][1] == "None":
                                                        nbrNone += 1
                                                    else:
                                                        nbrNone = 0
                                                    if nbrNone > 20:
                                                        nbrNone = 1/0

                                                except KeyError:
                                                    continue
                                                except ZeroDivisionError:
                                                    running = False
                                        else:
                                            break
                                if running == True:
                                    datacomparaison = True
                                    break
                    else:
                        for obs in reader[1:]:
                            if float(obs[0]) == 0:

                                observation_dict[0] = [obs[j], obs[-1]]
                            else:
                                observation_dict[int(float(obs[0]))] = [obs[j], obs[-1]]
                        nbrNone = 0
                        for date in sims_dates:
                            if date in observation_dict.keys():
                                value = observation_dict[date][1]
                                if observation_dict[date][1] == "None":
                                    nbrNone += 1

                                # if nbrNone > 150:
                                # break
                        running = False
                        if nbrNone / len(sims_dates) < 0.80:
                            if reader[0][-1] > depth_levels[0]:
                                depth_levels.append(reader[0][-1])
                                with open("{}/Tzt.csv".format(output_folder), "r") as simulation_file:
                                    reader = list(csv.reader(simulation_file))
                                for sim in reader:
                                    try:
                                        if int(float(depth_levels[2])) != float(depth_levels[2]):
                                            xa = int(float(depth_levels[2]))
                                            xb = xa + 1
                                            if sim[xa] != "None" and sim[xb] != "None":
                                                valueyc = findYPoint(xa, xb, float(sim[xa]), float(sim[xb]),
                                                                     float(depth_levels[2]))
                                                simulation_dict1[sims_dates[reader.index(sim)]] = [sim[2], valueyc]
                                            else:
                                                simulation_dict1[sims_dates[reader.index(sim)]] = [sim[2], "None"]
                                        else:
                                            simulation_dict1[sims_dates[reader.index(sim)]] = [sim[2], sim[-1]]
                                    except IndexError:
                                        continue

                                csvFile = csv.writer(final)
                                csvFile.writerow(["Date", "Observations", depth_levels[0], depth_levels[2], "Simulations",
                                                  depth_levels[0], depth_levels[2]])
                                nbrNone = 0
                                running = True
                                for date in sims_dates:
                                    if running == True:
                                        if date in observation_dict.keys():
                                            try:
                                                print(observation_dict1[date][0])
                                                print(observation_dict1[date][1])
                                                print(simulation_dict1[date][0])
                                                print(simulation_dict1[date][1])
                                                csvFile.writerow(
                                                    [date, '', observation_dict1[date][0], observation_dict[date][1], '',
                                                     simulation_dict1[date][0], simulation_dict1[date][1]])
                                                if observation_dict[date][1] == "None":
                                                    nbrNone += 1
                                                else:
                                                    nbrNone = 0
                                                if nbrNone > 20:
                                                    nbrNone = 1 / 0

                                            except KeyError:
                                                continue
                                            except ZeroDivisionError:
                                                running = False
                                    else:
                                        break
                            if running == True:
                                datacomparaison = True


    if datacomparaison == False:
        print("Unable to compare, no full observed data  found ")
        return False
    else:
        return True

def performance_analysis(lake_name, input_folder, output_folder):
    """
    Opens the comparison file created by make_comparison_file, and prints the results of analysis functions.

    :param
    :param
    :param output_folder: A string, containing the folder containing the comparison file.
    :return: Score, a float representing the overall performance of the current simulation.
    """
    if lake_name == "NohipaloMustjarv" or lake_name =="Nohipalo_Mustjarv":
        file = 'NoM'
    elif lake_name == "NohipaloValgejarv" or lake_name == "Nohipalo_Valgejarv":
        file = 'NoV'
    elif lake_name == "Crystal_lake" or lake_name == "Crystal":
        file = 'CRY'
    elif lake_name == "Crystal_Bog" or lake_name == "CrystalB":
        file = 'CrB'
    else:
        file = lake_name[:3]
    with open("{}/{}_par".format(input_folder, file), "r") as param_file:
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
    r_squ1 = r_squared(obs_list_1, sims_list_1)
    r_squ2 = r_squared(obs_list_2, sims_list_2)

    """if r_squ1 < 0:
        r_squ1_B = -r_squ1
    else: r_squ1_B = r_squ1
    if r_squ2 < 0:
        r_squ2_B = -r_squ2
    else: r_squ2_B = r_squ2"""

    score = (sos1 + sos2) #+ (rms1 + rms2) * 1000  + (1 - r_squ1_B) * 100 + (1 - r_squ2_B) * 100

    print("Analysis of {}.".format(output_folder[10:]))
    print("Sums of squares : {}, {}".format(sos1, sos2))
    print("RMSE : {}, {}".format(rms1, rms2))
    print("R squared : {}, {}".format(r_squ1, r_squ2))

    print("RMSE/SD : {}, {}".format(rmse_by_sd(obs_list_1, rms1), rmse_by_sd(obs_list_2, rms2)))
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
        if obs_list[x] == 'None':
            continue
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
    try:
        result = sqrt(sums_of_squares(obs_list, sims_list)/lenght)
    except ZeroDivisionError:
        result = 0
    return result


def r_squared(obs_list, sims_list):
    """
    Find the R squared for the simulations compared to the expected observations
    :param obs_list: A list of observed temperatures.
    :param sims_list: A list of simulated temperatures.
    :return: results of R squared, as a float
    """

    x = []
    y = []

    for i in obs_list:
        try:

            x.append(float(i))
            y.append(float(sims_list[obs_list.index(i)]))

        except ValueError: continue
        except IndexError: break


    return r2_score(x, y)

def standard_deviation(obs_list):
    """
    Find the standard deviation of the observations
    :param obs_list: Type list. The list of observed temperatures
    :return: The standard deviation of obs_list
    """
    observations = []
    for obs in obs_list:
        try:
            observations.append(float(obs))
        except ValueError: continue

    return statistics.stdev(observations)

def rmse_by_sd(obs_list, rmse):
    """
    Divides RMSE of the simulations by the SD of the observations
    :param obs_list: A list of observed temperatures.
    :param rmse: Float
    :return: A float, RMSE / SD
    """
    try:
        result = rmse/standard_deviation(obs_list)
    except ZeroDivisionError:
        result = "Error_Zero_Division"
    return result

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

                    temperatures_by_depth(observation_path, lake_name, outdir,modelid, scenarioid)

                    obs_data = make_comparison_file(outdir)

                    if obs_data is True:
                        performance_analysis(outdir)
                    else:
                        x = 1/0

                    print("\nEnd of case.\n\n")

                except KeyError:
                    print("\nCould not run MyLake simulation with these parameters.\nEnd of case.\n\n")
                except IndexError:
                    print("\nError in optimisation function.\nEnd of case.\n\n")
                except ZeroDivisionError:
                    print("\nNo obs data to compare with.\nEnd of case.\n\n")

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
        end_file.writelines(["Calibration results:", str(res)])

    return res



def run_optimization_Mylake_save(lake_name, input_directory,outdir,observation_path, region, modelid, scenarioid):
    """
    Intermediary function calling mylakepar function to generate new parameters, then running myLake with these parameters,
    and finally

    :param lake_name: Type string. The name of the lake to optimise
    :param observation_path: Type string. Observed temperatures file
    :param input_directory: Type string. The folder containing all input files (input, parameters, init)
    :param region: Type string. The abreviation for the lake's region (Ex: United-States = US, Norway = NO)
    :param outdir: Type string. The output folder
    :param modelid: Type string. Prediction model used.
    :param scenarioid: Type string. The prediction scenario. For optimization purpose this is always historical

    :return: performance analysis, which itself returns a score as float
    """



    depths = None
    for lake in temp_by_lake:
        if lake_name == lake:
            depths = temp_by_lake[lake_name]
            break

    if depths != None:
        run_myLake_ISIMIP.run_myLake(observation_path, input_directory, region, lake_name, modelid, scenarioid,
                                     flag="calibration")

        try:
            make_comparison_file(outdir,depths)

            score = performance_analysis(lake_name, input_directory, outdir)

            with open("{}/Calibration_Complete.txt".format(outdir), "w") as end_file:
                end_file.writelines(["Calibration results:", str(score)])


            return score


        except ValueError:
            print('error value')
            X = 1/0
            return -1
        except ZeroDivisionError:
            print('zero division')
            X = 1 / 0
            return -1
        except:
            print('error other')
            X = 1 / 0
            return -1



def run_optimization_Mylake(lake_name, observation_path, input_directory, region, outdir, modelid, scenarioid,params):
    """
    Intermediary function calling mylakepar function to generate new parameters, then running myLake with these parameters,
    and finally

    :param lake_name: Type string. The name of the lake to optimise
    :param observation_path: Type string. Observed temperatures file
    :param input_directory: Type string. The folder containing all input files (input, parameters, init)
    :param region: Type string. The abreviation for the lake's region (Ex: United-States = US, Norway = NO)
    :param outdir: Type string. The output folder
    :param modelid: Type string. Prediction model used.
    :param scenarioid: Type string. The prediction scenario. For optimization purpose this is always historical

    :return: performance analysis, which itself returns a score as float
    """

    # kz_N0, c_shelter, alb_melt_snow, alb_melt_ice, swa_b0, swa_b1 = params
    # i_scv = 1
    # i_sct = 0
    #


    depths = None
    for lake in temp_by_lake:
        if lake_name == lake:
            depths = temp_by_lake[lake_name]
            break

    if depths != None:
        # run_myLake_ISIMIP.mylakepar(retreive_longitude(lake_name), retreive_latitude(lake_name), lake_name,
        #                             input_directory,
        #                             kz_N0, c_shelter, alb_melt_ice, alb_melt_snow, i_scv, i_sct, swa_b0, swa_b1)

        run_myLake_ISIMIP.run_myLake(observation_path, input_directory, region, lake_name, modelid, scenarioid,
                                     flag="calibration")


        try:
            make_comparison_file(outdir,depths)

            score = performance_analysis(lake_name, input_directory, outdir)
            return score


        except ValueError:
            print('error value')
            X = 1/0
            return -1
        except ZeroDivisionError:
            print('zero division')
            X = 1 / 0
            return -1
        except:
            print('error other')
            X = 1 / 0
            return -1



def optimize_differential_evolution(lake_name, observation_path, input_directory, region, outdir, modelid, scenarioid):
    func = lambda params: run_optimization_Mylake(lake_name, observation_path, input_directory, region,
                                                  outdir, modelid, scenarioid, params)

    params_0 = np.array([0, 0.3, 0.55, 1, 0, 2.5, 1])
    bounds = [(0.001, 0.000001), (0, 1), (0.4, 1), (0.4, 1), (0.4, 4), (0.4, 2)]

    try:
        res = differential_evolution(func, bounds, tol=10, disp=True)
        print(res)

        success = float(res.get('fun'))

        if res == -1:
            print("Couldn't do {}".format(lake_name))
            x= 1/0
            return res
        else:
            params = tuple(res.get('x'[0]))
            orig_stdout = sys.stdout
            with open("{}/Calibration_Complete.txt".format(outdir), "w") as end_file:
                sys.stdout = end_file
                run_optimization_Mylake(lake_name, observation_path, input_directory, region, outdir, modelid, scenarioid,
                                        params)
                end_file.writelines(["Calibration results:", str(res)])
                sys.stdout = orig_stdout

            return res
    except:
        print('problem when doing calibration')
        outputdir2 = list(outdir.split("/"))
        outputdir3 = 'output'
        for i in outputdir2:
            if i == 'output':
                outputdir3 = i
            else:
                if not os.path.exists(outputdir3):
                    os.mkdir(outputdir3)
                outputdir3 = os.path.join(outputdir3, i)
        if not os.path.exists(outputdir3):
            os.mkdir(outputdir3)
        open("{}/Calibration_problem.txt".format(outdir), "w").close()
        print("problem with the calibration: lake %s"%(lake_name))
def retreive_longitude(lake):
    """Get longitude from param files

    """

    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    with open("observations/{}/{}/{}_hypsometry.csv".format(reg,lake, lake)) as obs:
        reader = list(csv.reader(obs))
        prefix = reader[1][0][3:]

    long = 0

    with open("input/{}/{}/{}_par".format(reg, prefix, prefix)) as par_file:
        for line in par_file.readlines():
            if "longitude" in line:
                for i in range(1, 30):
                    if line[i] == r"\t":
                        for j in range(i, 31):
                            if line[j] == r"\t":
                                long = float(line[i:j])
                                break
                            else:
                                continue
                        break
                    else:
                        continue

                break

    return long

def retreive_latitude(lake):
    """Get latitude from param files"""
    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    with open("observations/{}/{}/{}_hypsometry.csv".format(reg,lake, lake)) as obs:
        reader = list(csv.reader(obs))
        prefix = reader[1][0][3:]

    lat = 0
    with open("input/{}/{}/{}_par".format(reg, prefix, prefix)) as par_file:
        for line in par_file.readlines():
            if "latitude" in line:
                for i in range(1, 30):
                    if line[i] == r"\t":
                        for j in range(i, 31):
                            if line[j] == r"\t":
                                long = float(line[i:j])
                                break
                            else:
                                continue
                        break
                    else:
                        continue

                break

    return lat

"""
#test functions for the algorithm

def optimize_test():
    func = lambda params : test_function(params)
    params_0 = np.array([10, 5])
    res = minimize(func, params_0, method="nelder-mead", options={'xtol': 0, 'disp': True})

    print(str(res))
    return res


def test_function(params):
    x, y = params
    var1 = x**2
    var2 = y**3
    calcul = 3*var1/var2
    if calcul > 10:
        score = calcul - 10
    else:
        score = 10 - calcul

    return score
"""

if __name__ == "__main__":
    #temperatures_by_depth("observations/Langtjern", "Langtjern", "output/NO/Langtjern/GFDL-ESM2M/rcp26")
    #make_comparison_file("output/NO/Langtjern/GFDL-ESM2M/rcp26")
    #performance_analysis("Langtjern", "input/NO/Lan", "output/NO/Langtjern/GFDL-ESM2M/rcp26")
    #optimise_lake("Langtjern", "observations/Langtjern", "input/NO/Lan", "NO", "forcing_data/Langtjern", "output/NO/Langtjern", "GFDL-ESM2M", "historical")
    #find_best_parameters("output/NO/Langtjern/optimisation_log.txt")
    #optimize_Nelder_Meald("Langtjern", "observations/Langtjern", "input/NO/Lan", "NO", "forcing_data/Langtjern", "output/NO/Langtjern/GFDL-ESM2M/historical", "GFDL-ESM2M", "historical")
    #optimize_differential_evolution("Kilpisjarvi", "observations/Kilpisjarvi", "input/FI/Kil", "CA", "output/CA/Kilpisjarvi/GFDL-ESM2M/rcp26", "GFDL-ESM2M", "rcp26")
    #optimize_test()
    #print(retreive_latitude("BurleyGriffin"))
    datafolder = r'C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\output'
    alldf = pd.read_csv(os.path.join(datafolder, "comparison_all_lake2.csv"))
    obs_list_1 = alldf['obsT']
    sims_list_1 = alldf['modelT']
    print(1)
    sos1 = sums_of_squares(obs_list_1, sims_list_1)
    print(sos1)
    rms1 = root_mean_square(obs_list_1, sims_list_1)
    print(rms1)
    r_squ1 = r_squared(obs_list_1, sims_list_1)
    print(sos1,rms1,r_squ1)




    #temperatures_by_depth(r'C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations\NZ\Rotorua', 'Rotorua', r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\output\NZ\Rotorua\EWEMBI\historical")