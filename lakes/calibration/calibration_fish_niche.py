#!/usr/bin/env python

""" Script for MyLake - ISIMIP - Calibration
By using the algorithm differential evolution for the calibration,
the script searches for the best values for the parameters by comparing the observed and simulated temperatures.
The score used by the algorithm is calculating by the addition the sum of squares of the upper and lower layers.
"""

__author__ = "Julien Bellavance and Marianne Cote"

from lake_information as LakeInfo
import pandas as pd
import os
import sys
import csv
import statistics
from sklearn.metrics import r2_score
from math import sqrt, floor
from scipy.optimize import differential_evolution

temp_by_lake = {'Annie': [1, 17], 'Allequash_Lake': [1, 6], 'Alqueva': [1, 60], 'Annecy': [10, 61],
                "Argyle": [0.7, 26.7], 'Biel': [5, 70], 'Big_Muskellunge_Lake': [1, 18],
                'Black_Oak_Lake': [0.9144, 24.384], 'Bourget': [9.5, 106], 'Burley_Griffin': [3, 12],
                'Crystal_Bog': [0.5, 1.5], 'Crystal_Lake': [1, 17], 'Delavan': [1, 15], 'Dickie_Lake': [1, 9],
                'Eagle_Lake': [1, 10], 'Ekoln_basin_of_Malaren': [1, 27.25], 'Erken': [0.5, 20],
                'Esthwaite_Water': [0.5, 11.5], 'Falling_Creek_Reservoir': [1, 8], 'Feeagh': [0.9, 42],
                'Fish_Lake': [1, 17], 'Geneva': [3.6, 263.6], 'Great_Pond': [1, 19], 'Green_Lake': [1, 66],
                'Harp_Lake': [1, 27], 'Kilpisjarvi': [0.75, 31], 'Kinneret': [2, 34], 'Kivu': [0, 90],
                'Kuivajarvi': [0.5, 8], 'Langtjern': [1, 8], 'Laramie_Lake': [0.8, 6], 'Lower_Zurich': [1, 100],
                'Mendota': [0, 20], 'Monona': [1, 18], 'Mozaisk': [1, 7], 'Mt_Bold': [1, 34], 'Mueggelsee': [1, 5],
                'Neuchatel': [0, 100], 'Ngoring': [4.768982031, 19.76898203], 'Nohipalo_Mustjarv': [0.5, 5],
                'Nohipalo_Valgejarv': [0.5, 7], 'Okauchee_Lake': [1, 25], 'Paajarvi': [1, 60],
                'Rappbode_Reservoir': [2.82, 52.62], 'Rimov': [0.2, 14.2], 'Rotorua': [2, 16], 'Sammamish': [1, 20],
                'Sau_Reservoir': [2, 40], 'Sparkling_Lake': [0, 17], 'Stechlin': [0, 55], 'Sunapee': [1, 9],
                'Tahoe': [1.5, 168.1], 'Tarawera': [2, 80], 'Toolik_Lake': [1, 16], 'Trout_Bog': [1, 7],
                'Trout_Lake': [1, 29], 'Two_Sisters_Lake': [4.572, 15.24], 'Vendyurskoe': [4.58, 11.3],
                'Vortsjarv': [0.5, 3], 'Washington': [1, 55], 'Windermere': [1, 35], 'Wingra': [1, 3]}


def run_optimization_Mylake(lake_name, params, modelid, scenarioid):
    """
    Intermediary function calling initiate_par.mylakepar function to generate new parameters, then running MyLake model
    with these parameters, and finally create file comparing the observed and simulated temperatures.
    :param lake_name: Name of the lake calibrating
    :param params: List of value giving for each parameter
    :param modelid: Type string. The climatic model used.
    :param scenarioid: Type string. The prediction scenario. For optimization purposes, this is always historical.
    :return: performance analysis, which itself returns the score, rmse, and r_square as a list.
    """

    print("params ", len(params))
    kz_n0, c_shelter, alb_melt_snow, alb_melt_ice, swa_b0, swa_b1 = params
    i_scv = 1
    i_sct = 0

    lake = CalibrationInfo(lake_name, kz_n0=kz_n0, c_shelter=c_shelter,
                           alb_melt_ice=alb_melt_ice, alb_melt_snow=alb_melt_snow, i_sct=i_sct,
                           i_scv=i_scv, swa_b0=swa_b0, swa_b1=swa_b1)
    outdir = os.path.join(lake.output_path, modelid, scenarioid)

    if lake.selected_depth is not None:

        lake.initiate_par(kz_n0=kz_n0, c_shelter=c_shelter,
                          alb_melt_ice=alb_melt_ice, alb_melt_snow=alb_melt_snow,
                          swa_b0=swa_b0, swa_b1=swa_b1).mylakepar()

        lake.run_mylake(modelid, scenarioid, "calibration")

        if not os.path.exists("{}/Observed_Temperatures.csv".format(outdir)):
            lake.temperatures_by_depth()

        try:
            lake.make_comparison_file()

            score, rmse, rsquare = lake.performance_analysis()

            with open("{}/Calibration_Complete.txt".format(outdir), "w") as end_file:
                end_file.writelines(["Calibration results:", str(score)])

            return score

        except ValueError:
            print('error value')
            x = 1 / 0
            print(x)
            return -1

        except ZeroDivisionError:
            print('zero division')
            x = 1 / 0
            print(x)
            return -1
        except:
            print('error other')
            x = 1 / 0
            print(x)
            return -1
    else:
        return -1


class CalibrationInfo(LakeInfo):
    """
    Class contains all the attributes needed to run the calibration.
    """

    def __init__(self, lake_name, kz_n0=0.00007, c_shelter="NaN", alb_melt_ice=0.6, alb_melt_snow=0.9, i_scv=1.15,
                 i_sct=0, swa_b0=2.5, swa_b1=1, k_bod=0.1, k_sod=500, i_sc_doc=1):
        """
        Initiates the class. CalibrationInfo inherites from the class LakeInfo all the attributes and functions.
        :param lake_name: Type string. Long lake's name.
        :param kz_n0: Type float: by default 0.00007. Min. stability frequency (s-2)
        :param c_shelter: Type float: by default empty. Wind shelter parameter (-)
        :param alb_melt_ice: Type float: by default 0.6. Albedo of melting ice (-)
        :param alb_melt_snow: Type float: by default 0.9. Albedo of melting ice (-)
        :param i_scv: Type float: by default 1.15. Scaling factor for inflow volume (-)
        :param i_sct: Type float: by default 0. Scaling coefficient for inflow temperature (-)
        :param swa_b0: Type float: by default 2.5. Non-PAR light atteneuation coefficient (m-1).
        :param swa_b1: Type float: by default 1. PAR light atteneuation coefficient (m-1).
        :param k_bod: Type float: by default 0.1. Organic decomposition rate (1/d)
        :param k_sod: Type float: by default 500. Sedimentary oxygen demand (mg m-2 d-1).
        :param i_sc_doc: Type float: by default 1. Scaling factor for inflow concentration of DOC  (-)
        """
        super().__init__(lake_name)
        self.selected_depth = temp_by_lake.get(lake_name, None)
        self.kz_n0 = kz_n0
        self.c_shelter = c_shelter
        self.alb_melt_ice = alb_melt_ice
        self.alb_melt_snow = alb_melt_snow
        self.i_scv = i_scv
        self.i_sct = i_sct
        self.swa_b0 = swa_b0
        self.swa_b1 = swa_b1
        self.k_bod = k_bod
        self.k_sod = k_sod
        self.I_sc_doc = i_sc_doc

    def optimize_differential_evolution(self, modelid="EWEMBI", scenarioid="historical"):
        """
        Function prepares and launches the calibration by using the function differential_evolution from scipy.
        :param modelid: Climatic model used for the calibration.
        :param scenarioid: Climatic scenario used for the calibration
        :return: class CalibrationResult: class containing the parameters values resulting from the calibration and
        the metric measures at those values.
        """
        func = lambda parameters: run_optimization_Mylake(self.lake_name, parameters, modelid, scenarioid)

        # Initial values: [0, 0.3, 0.55, 1, 0, 2.5, 1]
        bounds = [(0.001, 0.000001), (0, 1), (0.4, 1), (0.4, 1), (0.4, 4), (0.4, 2)]
        try:
            print("bounds ", len(bounds))
            res = differential_evolution(func, bounds)
            print(res)

            if res == -1:
                print("Couldn't do {}".format(self.lake_name))
                return res
            else:
                params = tuple(res.get('x'[0]))
                orig_stdout = sys.stdout
                with open("{}/Calibration_Complete.txt".format(self.output_path), "w") as end_file:
                    sys.stdout = end_file
                    score, rmse, r_square = run_optimization_Mylake(self.lake_name, params, modelid, scenarioid)
                    end_file.writelines(["Calibration results:", str(res)])
                    sys.stdout = orig_stdout

                return CalibrationResult(self.lake_name, res, score, rmse, r_square)
        except:
            print('problem when doing calibration')
            outputdir2 = list(self.output_path.split("/"))
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
            open("{}/{}/{}/Calibration_problem.txt".format(self.output_path, "EWEMBI", "historical"), "w").close()
            print("problem with the calibration: lake %s" % self.lake_name)

    def variables_by_depth(self):
        """
        Creates a new csv file with the observed temperatures separated in columns by depths.
        :return: None
        """
        obs_file = pd.read_excel(self.observation_file,self.lake_id)
        axisx = obs_file[0].unique()
        axisy = obs_file[1].unique()
        datasetT = []
        datasetO = []
        datasetS = []
        for date in axisx:
            valueatdepthT = []
            valueatdepthO = []
            valueatdepthS = []
            for depth in axisy:
                row = obs_file.loc[(obs_file['date'] >= date) & (obs_file['depth(max)'] <= depth)]
                valueatdepthT.append(row.loc['Water temp (°C)'])
                valueatdepthO.append('O2(mg/l)')
                valueatdepthS.append('Siktdjup (m)')
            datasetT.append(valueatdepthT)
            datasetO.append(valueatdepthO)
            datasetS.append(valueatdepthS)

        temperature = pd.DataFrame(index=axisx,columns=axisy, data=datasetT)
        temparature = temperature.dropna(subset=axisy, how='all')
        temperature.to_csv(os.path.join(self.calibration_path, "Observed_Temperature.csv"))

        oxygen = pd.DataFrame(index=axisx, columns=axisy, data=datasetT)
        oxygen = oxygen.dropna(subset=axisy, how='all')
        oxygen.to_csv(os.path.join(self.calibration_path, "Observed_Oxygen.csv"))

        secchi = pd.DataFrame(index=axisx, columns=axisy, data=datasetT)
        secchi = secchi.dropna(subset=axisy, how='all')
        secchi.to_csv(os.path.join(self.calibration_path, "Observed_Secchi.csv"))


        print("observation done ... ... ... ... ")

    def make_comparison_file(self):
        """
        Searches in a given output folder an observation file containing measured temperatures for a lake on a finite
        period and the simulated temperatures file. Then writes corresponding observed and simulated temperatures
        to a csv file, where each column is a list of temperatures for a given depth.
        :return: None
        """

        with open("{}/{}/{}/T_comparison.csv".format(self.output_path, "EWEMBI", "historical"), "w",
                  newline="\n") as file:
            observation_dict = {}
            simulation_dict = {}
            depth_levels = []

            with open("{}/{}/{}/Observed_Temperatures.csv".format(self.output_path, "EWEMBI", "historical"),
                      "r") as observation_file:
                reader = list(csv.reader(observation_file))
                for i in range(1, len(reader[0])):

                    if float(reader[0][i]) == self.selected_depth[0]:
                        depth_levels.append(reader[0][i])
                        col1 = i
                    elif float(reader[0][i]) == self.selected_depth[1]:
                        depth_levels.append(reader[0][i])
                        col2 = i
                    if len(depth_levels) == 2:
                        break

                for obs in reader[1:]:
                    observation_dict[int(obs[0])] = [obs[col1], obs[col2]]

            sims_dates = get_dates_of_simulation(self.start_year, self.end_year)

            with open("{}/{}/{}/Tzt.csv".format(self.output_path, "EWEMBI", "historical"), "r") as simulation_file:
                reader = list(csv.reader(simulation_file))

                for sim in reader:
                    try:

                        if (floor(float(depth_levels[0])) + 0.5) != float(depth_levels[0]):
                            xa = (floor(float(depth_levels[0])))
                            xb = xa + 1
                            if xa >= float(depth_levels[0]):
                                xa = xa - 1
                                xb = xa + 1
                            if floor(float(depth_levels[1])) + 0.5 != float(depth_levels[1]):
                                xa1 = (floor(float(depth_levels[1])))
                                xb1 = xa1 + 1
                                if (xa1 + 0.5) > float(depth_levels[1]):
                                    xa1 = (floor(float(depth_levels[1]))) - 1
                                    xb1 = xa1 + 1
                                if sim[xa] != "None" and sim[xb] != "None":

                                    valueyc = findYPoint(xa, xb, sim[xa], sim[xb], float(depth_levels[0]))
                                    if sim[xa1] != "None" and sim[xb1] != "None":

                                        valueyc1 = findYPoint(xa1, xb1, sim[xa1], sim[xb1], float(depth_levels[1]))
                                        simulation_dict[sims_dates[reader.index(sim)]] = [valueyc, valueyc1]

                                    else:
                                        simulation_dict[sims_dates[reader.index(sim)]] = [valueyc, "None"]

                                else:
                                    if sim[xa1] != "None" and sim[xb1] != "None":
                                        valueyc1 = findYPoint(xa1, xb1, sim[xa1], sim[xb1], float(depth_levels[1]))

                                        simulation_dict[sims_dates[reader.index(sim)]] = ["None", valueyc1]
                                    else:
                                        simulation_dict[sims_dates[reader.index(sim)]] = ["None", "None"]

                            else:
                                if sim[xa] != "None" and sim[xb] != "None":
                                    valueyc = findYPoint(xa, xb, sim[xa], sim[xb], float(depth_levels[0]))

                                    simulation_dict[sims_dates[reader.index(sim)]] = [valueyc, sim[col2]]
                                else:
                                    simulation_dict[sims_dates[reader.index(sim)]] = ["None", sim[col2]]
                        else:
                            if floor(float(depth_levels[1])) + 0.5 != float(depth_levels[1]):
                                xa1 = floor(float(depth_levels[1]))
                                xb1 = xa1 + 1
                                if sim[xa1] != "None" and sim[xb1] != "None":
                                    valueyc1 = findYPoint(xa1, xb1, sim[xa1], sim[xb1], float(depth_levels[1]))

                                    simulation_dict[sims_dates[reader.index(sim)]] = [
                                        float(sim[int(float(depth_levels[0]) - 0.5)]), valueyc1]

                                else:
                                    simulation_dict[sims_dates[reader.index(sim)]] = [
                                        float(sim[int(float(depth_levels[0]) - 0.5)]), "None"]
                            else:
                                simulation_dict[sims_dates[reader.index(sim)]] = [
                                    float(sim[int(float(depth_levels[0]) - 0.5)]),
                                    float(sim[int(float(depth_levels[0]) - 0.5)])]
                    except IndexError:
                        continue

            csvfile = csv.writer(file)
            csvfile.writerow(["Date", "Observations", depth_levels[0], depth_levels[1], "Simulations", depth_levels[0],
                              depth_levels[1]])
            for date in sims_dates:
                if date in observation_dict.keys():
                    try:
                        csvfile.writerow([date, '', observation_dict[date][0], observation_dict[date][1], '',
                                          simulation_dict[date][0], simulation_dict[date][1]])

                    except KeyError:
                        continue
            return None

    def performance_analysis(self):
        """
        Opens the comparison file created by make_comparison_file, and prints the results of analysis functions.
        :return: Score, a float representing the overall performance of the current simulation.
        """

        file = self.prefix
        with open("{}/{}_par".format(self.input_path, file), "r") as param_file:
            for line in param_file.readlines():
                if "C_shelter" in line:
                    if float(line.split('\t')[1]) < 0:
                        return 10000
                elif "swa_b0" in line:
                    if float(line.split('\t')[1]) < 0:
                        return 10000
                elif "swa_b01" in line:
                    if float(line.split('\t')[1]) < 0:
                        return 10000
                elif "I_scV" in line:
                    if float(line.split('\t')[1]) < 0:
                        return 10000
                elif "I_scT" in line:
                    if float(line.split('\t')[1]) < 0:
                        return 10000
                elif "alb_melt_ice" in line:
                    if float(line.split('\t')[1]) < 0:
                        return 10000
                elif "alb_melt_snow" in line:
                    if float(line.split('\t')[1]) < 0:
                        return 10000

        with open("{}/{}/{}/T_comparison.csv".format(self.output_path, "EWEMBI", "historical"), "r") as file:
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

        score = (sos1 + sos2)  # + (rms1 + rms2) * 1000  + (1 - r_squ1_B) * 100 + (1 - r_squ2_B) * 100

        print("Analysis of {}.".format(self.lake_name))
        print("Sums of squares : {}, {}".format(sos1, sos2))
        print("RMSE : {}, {}".format(rms1, rms2))
        print("R squared : {}, {}".format(r_squ1, r_squ2))

        print("RMSE/SD : {}, {}".format(rmse_by_sd(obs_list_1, rms1), rmse_by_sd(obs_list_2, rms2)))
        print("Score : {}".format(score))

        return score, (rms1 + rms2), (r_squ1 + r_squ2)


class CalibrationResult:
    """
    Class containing the result of the calibration.
    """

    def __init__(self, lake_name, res, score, rmse, r_square):
        self.lake_name = lake_name
        self.params = tuple(res.get('x'[0]))
        self.score = score
        self.rmse = rmse
        self.r_square = r_square


def findYPoint(xa, xb, ya, yb, xc):
    """
    Function used to calculate the temperature at a depth non simulated by the model. MyLake simulates the temperature
     at each meter (atarting at 0.5) and this function permit to comparer the temperature at the same depth that it has
      been measured.
    :param xa: Closest depth simulated below the wanted depth.
    :param xb: Closest depth simulated over the wanted depth.
    :param ya: Temperature at the depth xa.
    :param yb: Temperature at the depth xb.
    :param xc: Depth at which the temparature is wanted
    :return: Temperature at the depth yc.
    """
    m = (float(ya) - float(yb)) / (float(xa) - float(xb))
    yc = (float(xc) - (float(xb) + 0.5)) * m + float(yb)
    return yc


def get_dates_of_simulation(start_year, stop_year):
    """
    Finds the dates for each day of simulation. The dates are found by beginning on the first of January of the given
    start year and adding a day until the 31st of December of the stop year, accounting for leap years.
    :param start_year: An integer, the chosen year for the beginning of the simulation.
    :param stop_year: An integer, the chosen year for the end of the simulation.
    :return: A list of all the dates of simulation, in order from first to last.
            Dates are integers in the form YYYYMMDD.
    """
    date_list = []
    year = start_year
    nb_year = stop_year - start_year
    if nb_year == 0:
        nb_year = 1
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


def sums_of_squares(obs_list, sims_list):
    """
    Finds the sums of squares for all temperatures listed in the comparison file.
    :param obs_list: A list of observed temperatures.
    :param sims_list: A list of simulated temperatures.
    :return: The result of the sums of square as a float.
    """
    sums = 0
    for x in range(len(obs_list)):
        if obs_list[x] == 'None':
            continue
        sums += (float(obs_list[x]) - float(sims_list[x])) ** 2

    return sums


def root_mean_square(obs_list, sims_list):
    """
    Finds the root_mean_square for the temperatures listed in the comparison file.
    :param obs_list: A list of observed temperatures.
    :param sims_list: A list of simulated temperatures.
    :return: The result of the root mean square as a float.
    """
    lenght = len(obs_list) + len(sims_list)
    try:
        results = sqrt(sums_of_squares(obs_list, sims_list) / lenght)
    except ZeroDivisionError:
        results = 0
    return results


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

        except ValueError:
            continue
        except IndexError:
            break

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
        except ValueError:
            continue

    return statistics.stdev(observations)


def rmse_by_sd(obs_list, rmse):
    """
    Divides RMSE of the simulations by the SD of the observations
    :param obs_list: A list of observed temperatures.
    :param rmse: Float
    :return: A float, RMSE / SD
    """
    try:
        results = rmse / standard_deviation(obs_list)
    except ZeroDivisionError:
        results = "Error_Zero_Division"
    return results


if __name__ == "__main__":
    calibration_info = CalibrationInfo('Langtjern')
    result = calibration_info.optimize_differential_evolution()
    print(result.lake_name, result.score, result.rmse, result.r_square)
