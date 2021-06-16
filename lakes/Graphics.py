#!/usr/bin/env python

""" Script for MyLake
script specific to the visualisation of the data.
"""

__author__ = "Marianne Cote"

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import linregress
import seaborn as sns
import os
import time
import math
import skill_metrics as sm
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import statsmodels.api as smodels
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import scipy.stats as stats
import pandas as pd
import numpy as np
from math import sqrt, floor, log10, log
from sklearn.metrics import r2_score, mean_squared_error
import statistics
from itertools import product
import csv
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

    d = pd.DataFrame(list(zip(obs_list, sims_list)),
                     columns=['obs', 'sim'])

    d["obs"] = d["obs"].astype(float)
    d["sim"] = d["sim"].astype(float)

    try:
        results = mean_squared_error(d["obs"], d["sim"])
        results = sqrt(results)
        resultsnormalise = sqrt(mean_squared_error(d["obs"], d["sim"])) / (max(d["obs"]) - min(d["obs"]))
    except:
        results = np.nan
        resultsnormalise = np.nan
    return results, resultsnormalise

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
    try:
        rsquare = r2_score(x, y)
    except:
        rsquare = np.nan
    return rsquare

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

def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

def Area_at_depth():
    lakes = pd.read_csv("2017SwedenList_only_validation_12lakes.csv", encoding='ISO-8859-1')
    lakes_small = lakes[lakes["volume"]< 1.0e7]['lake_id'].tolist()
    lakes_medium = lakes[lakes["volume"]>= 1.0e7 ]
    lakes_medium = lakes_medium[lakes_medium['volume'] <= 5.0e9]['lake_id'].tolist()
    lakes_large = lakes[lakes["volume"] >  5.0e9]['lake_id'].tolist()

    T_list = [2, 4, 6, 8, 10, 12, 13, 14, 15]
    light_list = [0.5, 1, 2, 4, 8, 16, 24, 32, 48]
    AreaDays=[]
    Areaday=[]
    NTG=[]
    for taille in ['small','medium','large']:
        for x in T_list:
            AreaDaysT = []
            AreadayT = []
            NTGT = []
            for y in light_list:
                if y == 0.5:
                    y = '0.50'
                df = pd.read_csv(os.path.join("F:\output", "fish_niche_Area_Light%s_T%s_2001-2010.csv"%(y,x)))
                if taille == 'small':
                    dataset = df[df['LakeId'].isin(lakes_small)].mean()
                elif taille== 'medium':
                    dataset = df[df['LakeId'].isin(lakes_medium)].mean()
                else:
                    dataset = df[df['LakeId'].isin(lakes_large)].mean()
                AreaDaysT.append(dataset['AreaDays'])
                AreadayT.append(dataset['Areaday'])
                NTGT.append(dataset['NTGdays'])
            AreaDays.append(AreaDaysT)
            Areaday.append(AreadayT)
            NTG.append(NTGT)

        x = np.array(T_list)
        y = np.array(light_list)

        #x = np.linspace(0, 5, 50)
        #y = np.linspace(0, 5, 40)
        X, Y = np.meshgrid(x, y)
        #Z = f(X, Y)

        ZAreaday = np.array(Areaday)

        ZAreadays = np.array(AreaDays)
        ZNTG = np.array(NTG)


        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.grid(False)

        plt.imshow(ZAreaday, extent=[2, 15, 0.5, 48], origin='lower', cmap='RdGy', interpolation='nearest', aspect='auto')
        plt.colorbar()

        plt.savefig("Areaday_%s.png" % taille)
        plt.close()
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.grid(False)
        plt.imshow(ZAreadays, extent=[2, 15, 0.5, 48], origin='lower', cmap='ocean', interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.savefig("Areadays_%s.png" % taille)
        plt.close()
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.grid(False)
        plt.imshow(ZNTG, extent=[2, 15, 0.5, 48], origin='lower', cmap='bone', interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.axis(aspect='image')
        plt.savefig("NTG_%s.png" % taille)
        plt.close()

def timeline_plot(modeleddata: list, modeleddates: list, observeddata: list = None, observeddates: list = None, ax=None,
                  line_kwargs: dict = {},
                  sct_kwargs: dict = {}):
    """
    plot timeline with modeled data (line) and observed data (dot)

    :param modeleddata: modeled data for each day included in years presenting observed data (at same depth)
    :param modeleddates: all days included in the period modeled
    :param observeddata: Observed data to be plot
    :param observeddates: Dates Where observed data have been measured
    :param ax: Axe where you want to plot. If None, look for last axe used in current figure
    :param line_kwargs: Others arguments given to the lineplot (measures' plot)
    :param sct_kwargs: Others arguments given to the scatterplot (Observations' plot)
    :return: None
    """
    if ax is None:
        ax = plt.gca()
    if modeleddates is None or modeleddata is None:
        raise TypeError

    if observeddata is None:
        sns.scatterplot(x=observeddates, y=observeddata, ax=ax, **sct_kwargs)

    sns.lineplot(x=modeleddates, y=modeleddata, ax=ax, **line_kwargs)
    ax.set_xlabel("Dates")
    ax.set_xlim(min(modeleddates), max(modeleddates))
    ax.set_ylim(-0.5, 30)

def target_diagram_background_custom(crmsd, bias, max_value_overall, min_value_overall, ax=None, sct_args={},
                              cicle_max_args = {}, cicle_min_args = {}, cicle_1_args = {}, other_circle_args = {},
                              space_between_circles = 0.5, max_number_cicle = 3, add_circles = True,color=['k','k','k'],label=['','',''],size = [10,10,10]):
    if ax is None:
        ax = plt.gca()

    if min_value_overall > max_value_overall:
        max_value = min_value_overall
        min_value = max_value_overall
    else:
        max_value = max_value_overall
        min_value = min_value_overall

    if add_circles:
        ax.add_patch(plt.Circle((0, 0), max_value, **cicle_max_args))
        ax.add_patch(plt.Circle((0, 0), 1, **cicle_1_args))
        ax.add_patch(plt.Circle((0, 0), min_value, **cicle_min_args))

        if min_value < 0.2:
            new_circle = math.floor(min_value)
        else:
            new_circle = math.ceil(min_value)
        for i in range(max_number_cicle-1):
            if (max_value-new_circle) > space_between_circles:
                if (new_circle+space_between_circles) != 1:
                    ax.add_patch(plt.Circle((0, 0), new_circle, **other_circle_args))
                    new_circle += space_between_circles
            else:
                break

    for i in range(len(crmsd)):
        sct_args['color'] = color[i]
        sct_args['s'] = size[i]
        sct_args['label'] = label[i]
        plt.scatter(y=bias[i], x=crmsd[i], **sct_args)

    if max_value < 1:
        limit = 1 + space_between_circles
    else:
        limit = max_value + space_between_circles

    plt.ylim(-1 * limit, limit)
    plt.xlim(-1 * limit, limit)
    sns.despine()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    plt.xlabel('Normalized\n cRMSE', horizontalalignment='right', x=1)
    # #ax.set_xlabel('Normalized cRMSE', loc='right')
    plt.ylabel('Normalized Bias', verticalalignment="top", y=1)
    # ax.set_ylabel('Normalized Bias')
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.set_aspect(1.0)

def target_diagram_background(crmsd, bias, max_value_overall, min_value_overall, ax=None, sct_args={},
                              cicle_max_args = {}, cicle_min_args = {}, cicle_1_args = {}, other_circle_args = {},
                              space_between_circles = 0.2,max_number_cicle = 3, add_circles = True,color=['k','k','k'],label=['','',''],size = [10,10,10]):
    if ax is None:
        ax = plt.gca()

    if min_value_overall > max_value_overall:
        max_value = min_value_overall
        min_value = max_value_overall
    else:
        max_value = max_value_overall
        min_value = min_value_overall

    if add_circles:
        if max_value > 1.5:
            limit = 4
            ax.add_patch(plt.Circle((0, 0), 3.5, **cicle_max_args))
            ax.add_patch(plt.Circle((0, 0), 3, **other_circle_args))
            ax.add_patch(plt.Circle((0, 0), 2, **other_circle_args))
            ax.add_patch(plt.Circle((0, 0), 1.5, **other_circle_args))
            ax.add_patch(plt.Circle((0, 0), 1, **cicle_1_args))
            ax.add_patch(plt.Circle((0, 0), 0.5, **cicle_min_args))
        else:
            limit = 2
            ax.add_patch(plt.Circle((0, 0), 1.5, **cicle_max_args))
            ax.add_patch(plt.Circle((0, 0), 1, **cicle_1_args))
            ax.add_patch(plt.Circle((0, 0), 0.5, **cicle_min_args))
    else:
        if max_value < 1:
            limit = 1 + space_between_circles
        else:
            limit = max_value + space_between_circles

        # if min_value < 0.2:
        #     new_circle = math.floor(min_value)
        # else:
        #     new_circle = math.ceil(min_value)
        # for i in range(max_number_cicle-1):
        #     if (max_value-new_circle) > space_between_circles:
        #         if (new_circle+space_between_circles) != 1:
        #             ax.add_patch(plt.Circle((0, 0), new_circle, **other_circle_args))
        #             new_circle += space_between_circles
        #     else:
        #         break

    for i in range(len(crmsd)):
        sct_args['color'] = color[i]
        sct_args['s'] = size[i]
        sct_args['label'] = label[i]
        plt.scatter(y=bias[i], x=crmsd[i], **sct_args)




    plt.ylim(-1 * limit, limit)
    plt.xlim(-1 * limit, limit)
    sns.despine()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    plt.xlabel('Normalized\n cRMSE', horizontalalignment='right', x=1)
    # #ax.set_xlabel('Normalized cRMSE', loc='right')
    plt.ylabel('Normalized Bias', verticalalignment="top", y=1)
    # ax.set_ylabel('Normalized Bias')
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.set_aspect(1.0)

def line_plot(lineStart=None,lineEnd=None, ax=None,linearg={}):
    if ax is None:
        ax = plt.gca()
    if lineStart is None:
        if x.min < y.min():
            lineStart = np.floor(x.min())
        else:
            lineStart = np.floor(y.min())
        lineStart = 0
    if lineEnd is None:
        if x.max() > y.max():
            lineEnd = np.ceil(x.max())
        else:
            lineEnd = np.ceil(y.max())

    plt.plot([lineStart, lineEnd], [lineStart, lineEnd],ax=ax, **linearg)
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)

def linear_regression_plot(x,y,ax=None,linearregressionarg={}, confidentintervalarg = {}, predictionintervalarg = {}):
    if ax is None:
        ax = plt.gca()


    x = smodels.add_constant(x)  # constant intercept term
    # Model: y ~ x + c
    model = smodels.OLS(y, x)
    fitted = model.fit()
    x_pred = np.linspace(x.min(), x.max(), 50)
    x_pred2 = smodels.add_constant(x_pred)
    y_pred = fitted.predict(x_pred2)

    ax.plot(x_pred, y_pred, **linearregressionarg)

    print(fitted.params)  # the estimated parameters for the regression line
    print(fitted.summary())  # summary statistics for the regression

    y_hat = fitted.predict(x)  # x is an array from line 12 above
    y_err = y - y_hat
    mean_x = x.T[1].mean()
    n = len(x)
    dof = n - fitted.df_model - 1

    t = stats.t.ppf(1 - 0.025, df=dof)
    s_err = np.sum(np.power(y_err, 2))
    conf = t * np.sqrt((s_err / (n - 2)) * (1.0 / n + (
            np.power((x_pred - mean_x), 2) / ((np.sum(np.power(x_pred, 2))) - n * (np.power(mean_x, 2))))))
    upper = y_pred + abs(conf)
    lower = y_pred - abs(conf)

    #Confidence Interval
    ax.fill_between(x_pred, lower, upper, **confidentintervalarg)

    #Prediction Interval
    sdev, lower, upper = wls_prediction_std(fitted, exog=x_pred2, alpha=0.025)
    ax.fill_between(x_pred, lower, upper, **predictionintervalarg)

def error_bar_plot(x,y,xerr,yerr,ax=None, errorbararg = {},markerwidth = 4):
    if ax is None:
        ax = plt.gca()

    (_, caps, _) = plt.errorbar(x, y, xerr=xerr, yerr=yerr,ax=ax, **errorbararg)
    for cap in caps:
        cap.set_markeredgewidth(markerwidth)

def base_plot_comparison(x,y, lineStart=None,lineEnd= None, ax=None,linecolor = "k"):
    if ax is None:
        ax = plt.gca()
    linearg = {'fmt': 'k-', 'color': linecolor, 'label': "y= x", 'linewidth': 4}
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    rmse, nrmse = root_mean_square(x, y)
    linearregressionarg = {'ftm': '-', 'color': 'k', 'linewidth': 4,
                        "label": "linear regression (y = %0.3f x + %0.3f) \n R\u00b2 : %0.3f RMSE: %0.3f" % (slope, intercept, r_value, rmse)}
    confidenceintervalarg = {'color': '#888888', 'alpha': 0.4, 'label': "Confidence interval"}
    predictionintervalarg = {'color': '#888888', 'alpha': 0.1, 'label': "Prediction interval"}
    line_plot(lineStart=lineStart,lineEnd=lineEnd,ax=ax, linearg=linearg)
    linear_regression_plot(x, y, ax, linearregressionarg, confidenceintervalarg, predictionintervalarg)

def base_contourplot(X,Y,z_variable_data, z_variablelist,lake,number_of_plot_by_row,vmin,vmax,ax = None, colorlinecontour = "white",individual=False):

    if type(z_variable_data) is dict:
        list_z = range(0, len(z_variablelist))
        for z in list_z:
            Z = z_variable_data["%s" % z_variablelist[z]]
            if ax is None:
                axs = plt.gca()
            else:
                try:
                    axis_x = int(np.floor(z / number_of_plot_by_row))
                    axis_y = int(z- (number_of_plot_by_row * axis_x))
                    axs = ax[axis_x][axis_y]
                except:
                    axs = ax

            try:
                cp1 = axs.contourf(X, Y, Z, vmin=vmin, vmax=vmax)
                cp = axs.contour(X, Y, Z, colors=colorlinecontour)

                plt.clabel(cp, inline=True, fmt='%.2e')
                axs.title.set_text("O2 threshold: %s mg/l" % z_variablelist[z])
            except:
                print("issue: possible number for subplot higher than number for subplot present")


        if len(list_z) < 16 and len(list_z) != 1:
            axs[3][3].set_visible(False)

        return cp1
    else:
        if individual:
            if ax is None:
                axs = plt.gca()
            else:
                axs = ax
            cm = axs.contourf(X, Y, z_variable_data, vmin=vmin, vmax=vmax)
            cp = axs.contour(X, Y, z_variable_data, colors=colorlinecontour)
            plt.colorbar(cm)  # Add a colorbar to a plot
            plt.clabel(cp, inline=True, fmt='%.2e')
            ax.title.set_text("O2 threshold: %s mg/l" % z_variablelist[0])
        return None



def create_dictionnary_of_z_value_by_lake(data,z_list,list_of_parameter_measured,lake_id_list,new_data):
    vmin,vmax = {},{}
    for z_value in [str(x) for x in z_list]:
        for column in [str(x) for x in list_of_parameter_measured]:
            for lake_id in [str(x) for x in lake_id_list]:
                if not column in new_data:
                    new_data[column]= {lake_id: {z_value:data[z_value][column][lake_id]}}
                else:
                    if not lake_id in new_data[column]:
                        new_data[column][lake_id] = {z_value: data[z_value][column][lake_id]}
                    else:
                        if not z_value in new_data[column][lake_id]:
                            new_data[column][lake_id][z_value]= data[z_value][column][lake_id]
                        else:
                            new_data[column][lake_id][z_value].append(data[z_value][column][lake_id])
    for column in [str(x) for x in list_of_parameter_measured]:
        for lake_id in [str(x) for x in lake_id_list]:
            vminlakemethod = []
            vmaxlakemethod = []
            if not column in vmin:
                for z_value in [str(x) for x in z_list]:
                    vminlakemethod.append(min([min(x) for x in new_data[column][lake_id][z_value]]))
                    vmaxlakemethod.append(max([max(x) for x in new_data[column][lake_id][z_value]]))

                vmin[column] = {lake_id: min(vminlakemethod)}
                vmax[column] = {lake_id: max(vmaxlakemethod)}
            else:
                for z_value in [str(x) for x in z_list]:
                    vminlakemethod.append(min([min(x) for x in new_data[column][lake_id][z_value]]))
                    vmaxlakemethod.append(max([max(x) for x in new_data[column][lake_id][z_value]]))

                vmin[column][lake_id]= min(vminlakemethod)
                vmax[column][lake_id]= max(vmaxlakemethod)


    return new_data,vmin,vmax

def get_file_by_value_of_xyz(x_value,y_value,z_value,dict_raw_data_by_lake,variables_list_in_order,directory,lake_id_list):
    #file format:
    #Need to be change if other file naming is use

    variables_in_order_for_file = ['Light', 'Temperature', 'Oxygen']
    general_name_file = "fish_niche_Area"
    simulated_period = "2001-2010"

    order_for_file = variables_in_order_for_file
    order_for_file[variables_in_order_for_file.index(variables_list_in_order[0])] = x_value
    order_for_file[variables_in_order_for_file.index(variables_list_in_order[1])] = y_value
    order_for_file[variables_in_order_for_file.index(variables_list_in_order[2])] = z_value
    variables_in_order_for_file = ['Light', 'Temperature', 'Oxygen']

    filename = "%s_%s%s_%s%s_%s%s_%s.csv"%(general_name_file,variables_in_order_for_file[0],order_for_file[0],variables_in_order_for_file[1],order_for_file[1],variables_in_order_for_file[2],order_for_file[2],simulated_period)


    # Open and extract data from file
    with open('%s/%s' % (directory, filename), newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    list_of_parameter_measured = data[0][1:]
    for lake in range(0,len(lake_id_list)):
        lake += 1
        if not "%s" % data[lake][0] in dict_raw_data_by_lake:
            try:
                dict_raw_data_by_lake["%s" % data[lake][0]] = [[float(data[lake][x]) for x in range(1,len(data[lake]))]]
            except ValueError:
                dict_raw_data_by_lake["%s" % data[lake][0]] = [[data[lake][x] for x in range(1,len(data[lake]))]]
        else:
            try:
                dict_raw_data_by_lake["%s" % data[lake][0]].append([float(data[lake][x]) for x in range(1,len(data[lake]))])
            except ValueError:
                dict_raw_data_by_lake["%s" % data[lake][0]].append([data[lake][x] for x in range(1,len(data[lake]))])

    return dict_raw_data_by_lake, list_of_parameter_measured

class Graphics:
    """
    Main class that regroup all function to plot modeled data
    """

    def __init__(self, outputfolder):
        self.output_folder = outputfolder
        self.time = time.strftime("%Y%m%d-%H%M%S")

    def comparison_obs_sims_plot(self, variable_analized, calibration_methods, modeldata, obsdata, depthlayers,
                                 ice_cover, icecovertarea=True):
        """

        :type obsdata: pd.DataFrame
        :param variable_analized:
        :param calibration_methods:
        :param modeldata:
        :param obsdata:
        :param depthlayers:
        :param ice_cover:
        :param icecovertarea:
        :return:
        """

        # Figures aesthetics
        colorsSchemeByWaterLevel = {"surface": ["black", "#B30000", "#FF9785"],
                                    "deepwater": ["black", '#14218F', "#0AEFFF"]}
        subplotAxeByWaterLevel = {"surface": 0, "deepwater": 1}
        markerStyleByWaterLevel = {"surface": "o", "deepwater": "s"}
        markerColorByWaterLevel = {"surface": colorsSchemeByWaterLevel['surface'][1],
                                   "deepwater": colorsSchemeByWaterLevel['deepwater'][1]}
        lineStyleByWaterLevel = {"new_calibration": "-", "old_calibration": "--", "estimation": "-."}
        sns.set_style("ticks", {"xtick.major.size": 100, "ytick.major.size": 100})
        plt.xticks(rotation=15)
        linewidthByMethod = {"new_calibration": 1.5, "old_calibration": 2, "estimation": 1}
        orderPlotPresentedByMethod = {"Observation": 100, "new_calibration": 10, "old_calibration": 50,
                                      "estimation": 80}
        scatterPlotDotSize = 50
        transparenceLineplot = 0.8
        iceCovertAreaColor = "grey"
        iceCovertAreaAlpha = 0.2
        legendNames = ["Second GA Calibration", "First GA Calibration", "Stewise Regression", "Observations"]

        fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 1]})

        for depth_level in ["surface", "deepwater"]:
            axe_level = subplotAxeByWaterLevel[depth_level]
            depthlayer = depthlayers[depth_level]

            if not obsdata.loc[obsdata['Depth'] == depthlayer]["Observations"].empty:

                # Ice Covert Area
                if icecovertarea:
                    secondaryAxe = axs[axe_level].twinx()
                    secondaryAxe.fill_between(ice_cover.iloc[:, 1].tolist(), ice_cover.iloc[:, 0].tolist(),
                                              color=iceCovertAreaColor, alpha=iceCovertAreaAlpha, zorder=-10)
                    secondaryAxe.set_ylim(0, 1)
                    secondaryAxe.set_yticklabels([])
                    secondaryAxe.yaxis.set_visible(False)

                # Plot each method and observations
                for method in calibration_methods:
                    lineplotstyle = {"linewidth": linewidthByMethod[method],
                                     "color": colorsSchemeByWaterLevel[depth_level][calibration_methods.index(method)],
                                     "zorder": orderPlotPresentedByMethod[method],
                                     "alpha": transparenceLineplot}

                    timeline_plot(modeleddata=modeldata["%s_Model_%s" % (method, depth_level)],
                                  modeleddates=modeldata['Dates'], ax=axs[axe_level], line_kwargs=lineplotstyle)
                    axs[axe_level].line[calibration_methods.index(method)].set_linestyle(lineStyleByWaterLevel[method])

                    if method == "new_calibration":
                        scatterplotstyle = {'marker': markerStyleByWaterLevel[depth_level],
                                            's': scatterPlotDotSize, "edgecolor": 'k',
                                            "facecolors": markerColorByWaterLevel[depth_level],
                                            "linewidth": linewidthByMethod[method],
                                            "linestyle": lineStyleByWaterLevel[method],
                                            "zorder": orderPlotPresentedByMethod["Observation"]}

                        timeline_plot(modeleddata=modeldata["%s_Model_%s" % (method, depth_level)],
                                      modeleddates=modeldata['Dates'],
                                      observeddata=obsdata.loc[obsdata['Depth'] == depthlayer]["Observations"],
                                      observeddates=obsdata.loc[obsdata['Depth'] == depthlayer]["Dates"],
                                      ax=axs[axe_level], line_kwargs=scatterplotstyle)

                # set y axis title to subplot in function of the variable analysed and the depth 
                if variable_analized == "Temperature":
                    unity = "°C"
                else:
                    unity = "mg*$10^-1$"
                axs[axe_level].set_ylabel("%s at %s m (%s)" % (variable_analized, depthlayer, unity))

                # legend
                axs[axe_level].legend(legendNames)

        # Add title to figure
        # plt.title('timeline_lake_%s_%s'%(self.lake_name,variable_analized)))

        # Save figure
        plt.savefig(
            os.path.join(self.output_folder,
                         "Comparative_timeline_%s_%s_%s" % (self.lake_name, variable_analized, self.time)))
        plt.close('all')

    def taylor_target_plot(self,all_data_from_model, all_data_from_observation, label_method, variable, information,
                           label_taylor):
        """

        :param all_data_from_model:
        :param all_data_from_observation:
        :param label_method:
        :param variable:
        :return:
        """
        # take list of lists
        # [[[]*3]*12]

        colorpalette = ['#7A0014', '#E7FFD6', '#4A7FBF']  # sns.color_palette('Accent_r', 3)
        color_circle = "k"
        markers = ["o", "v", "^", "s", "P", "*", ">", "X", "D", "<", "p", "d"]
        sizes = [200, 150, 100]
        circle_argsmin = {"color": color_circle, "fill": False, "ls": '--', 'zorder': 0}
        circle_argsmax = {"color": color_circle, "fill": False, "ls": '-.', 'zorder': 0}
        circle_args = {"color": color_circle, "fill": False, "ls": '-', 'zorder': 0}
        other = {"color": "grey", "fill": False, "ls": ':', 'zorder': 0}

        biaslist, crmsdlist, rmsdlist = [None] * (len(all_data_from_model)), [None] * (len(all_data_from_model)), [
            None] * (len(all_data_from_model))
        sdevlist, crmsdtaylist, ccoeflist = [None] * (len(all_data_from_model)), [None] * (len(all_data_from_model)), [
            None] * (len(all_data_from_model))
        fig1, ax1 = plt.subplots(figsize=(10, 10), dpi=100)
        print(len(all_data_from_model))
        for lake in range(0, len(all_data_from_model)):
            observations = all_data_from_observation[lake]
            models = all_data_from_model[lake]
            print(len(models))
            statistic_target_lake = [None] * (len(models))
            statistic_taylor_lake = [None] * (len(models) + 1)
            bias, crmsd, rmsd = [None] * (len(models)), [None] * (len(models)), [None] * (len(models))
            sdev, crmsdtay, ccoef = [None] * (len(models) + 1), [None] * (len(models) + 1), [None] * (len(models) + 1)
            for method in range(0, len(models)):
                stats_target = sm.target_statistics(models[method], observations[method], norm=True)
                stats_taylor = sm.taylor_statistics(models[method], observations[method])

                # Calculate statistics for target and taylor diagram
                bias[method] = stats_target['bias']
                crmsd[method] = stats_target['crmsd']
                rmsd[method] = stats_target['rmsd']

                if method == 0:
                    crmsdtay[method] = stats_taylor['crmsd'][0]
                    ccoef[method] = stats_taylor['ccoef'][0]
                    sdev[method] = stats_taylor['sdev'][0] / np.mean(observations[method])

                crmsdtay[method + 1] = stats_taylor['crmsd'][1]
                ccoef[method + 1] = stats_taylor['ccoef'][1]
                sdev[method + 1] = stats_taylor['sdev'][1] / np.mean(models[method])

                statistic_target_lake[method] = stats_target
                statistic_taylor_lake[method] = stats_taylor


            try:
                maxv = abs(round_decimals_up(max([max(bias, key=abs),max(crmsd, key=abs)])))
            except:
                maxv = 1
            try:
                minv = abs(round_decimals_down(min([min(bias, key=abs), min(crmsd, key=abs)])))
            except:
                minv = 0
            if not information == "all_lakes":
                sct_args = {'color': colorpalette[method], 'marker': markers[lake],
                            'label': label_method[method], 'zorder': 100, 'alpha': 0.8, 's': sizes[method], 'edgecolors': 'k'}
                target_diagram_background(crmsd=crmsd,bias=bias, max_value_overall=maxv, min_value_overall=minv, ax=ax1,max_number_cicle = 5,
                                    add_circles=True, sct_args=sct_args,cicle_max_args = circle_argsmax,
                                    cicle_min_args = circle_argsmin, cicle_1_args = circle_args, other_circle_args = other,color=colorpalette,label=label_method,size = sizes)

            else:
                sct_args = {'color': colorpalette[method], 'marker': markers[lake],
                            'label': label_method[method], 'zorder': 100, 'alpha': 0.8, 's': sizes[method],
                            'edgecolors': 'k'}
                biaslist[lake], crmsdlist[lake], rmsdlist[lake] = bias, crmsd, rmsd
                sdevlist[lake], crmsdtaylist[lake], ccoeflist[lake] = sdev, crmsdtay, ccoef

                print(len(all_data_from_model))
                if lake !=(len(all_data_from_model)-1):
                    target_diagram_background(crmsd=crmsd, bias=bias, max_value_overall=maxv,min_value_overall=minv, ax=ax1,max_number_cicle = 5, add_circles=False,sct_args=sct_args,cicle_max_args = circle_argsmax,
                                    cicle_min_args = circle_argsmin, cicle_1_args = circle_args, other_circle_args = other,color=colorpalette,label=label_method,size = sizes)
                else:
                    t1 = [max(i) for i in biaslist]
                    t2 = max(t1)
                    t3 = max([t2, max([max(i) for i in crmsdlist])])
                    try:
                        # maxv = abs(
                        # round_decimals_up(max([max([max(i) for i in biaslist]), max([max(i) for i in crmsdlist])])))
                        maxbias = [max(bias, key=abs)for bias in biaslist]
                        maxbiass = max(maxbias, key=abs)
                        maxcrmsd = [max(crmsd, key=abs) for crmsd in crmsdlist]
                        maxcrmsds = max(maxcrmsd, key=abs)
                        maxx = max([maxbiass,maxcrmsds],key=abs)
                        maxr = round_decimals_up(maxx)
                        maxv = abs(maxr)
                        # maxv = abs(round_decimals_up( max([max([max(bias, key=abs)for bias in biaslist], key=abs),max([max(crmsd, key=abs) for crmsd in crmsdlist], key=abs)], key=abs)))
                    except:
                        maxv = 1
                    try:
                        minbias = [min(bias, key=abs) for bias in biaslist]
                        minbiass = min(minbias, key=abs)
                        mincrmsd = [min(crmsd, key=abs) for crmsd in crmsdlist]
                        mincrmsds = min(mincrmsd, key=abs)
                        minx = min([minbiass, mincrmsds], key=abs)
                        minr = round_decimals_down(minx)
                        minv = abs(minr)
                        # minv = abs(
                        #     round_decimals_down(
                        #         min([min([min(i) for i in biaslist]), min([min(i) for i in crmsdlist])])))
                    except:
                        minv = 0

                    sct_args = {'color': colorpalette[method], 'marker': markers[lake],
                            'label': label_method[method], 'zorder': 100, 'alpha': 0.8, 's': sizes[method], 'edgecolors': 'k'}
                    target_diagram_background( crmsd=crmsd, bias=bias, max_value_overall=maxv, min_value_overall=minv,
                                        ax=ax1, max_number_cicle = 5,add_circles=True,sct_args=sct_args,cicle_max_args = circle_argsmax,
                                        cicle_min_args = circle_argsmin, cicle_1_args = circle_args,
                                        other_circle_args = other,color=colorpalette,label=label_method,size = sizes)
            if lake == 0:
                plt.legend()
                if information == "all_lakes":
                    biaslist[lake], crmsdlist[lake], rmsdlist[lake] = bias, crmsd, rmsd
                    sdevlist[lake], crmsdtaylist[lake], ccoeflist[lake] = sdev, crmsdtay, ccoef

        # plt.show()

        plt.savefig(
            os.path.join(self.output_folder, "Comparative_target_%s_%s_%s" % (variable, information, self.time)))
        plt.close()

    def graphique_secchi(self,x, y, xerr, yerr, calibration=False,old=False):


        # Esthetic Parameters
        colorpalette = sns.color_palette("dark", 10)
        linecolor = colorpalette[9]
        lineStart = 0
        lineEnd = 20
        errorbararg = {'fmt': 'o', 'color': colorpalette[3], 'markersize': 8, 'capsize': 20,
                       'linewidth': 4, 'elinewidth': 4}


        sns.set(font_scale=2)
        sns.set_style("ticks")
        plt.grid(False)


        #Figure
        fig, ax = plt.subplots(figsize=(15.0, 14.5))
        base_plot_comparison(x,y,ax=ax,linecolor=linecolor)
        error_bar_plot(x,y,xerr,yerr,ax,errorbararg)

        plt.xlabel("Average Observed Secchi_Depth (m)")
        plt.ylabel("Average Modeled Secchi Depth (m)")
        plt.xlim(0, 15)
        plt.ylim(0, 20)
        fig.suptitle("")
        fig.tight_layout(pad=2)
        ax.legend(loc='best')

        if calibration:
            fig.savefig('Secchi_mean_comparison_calibration_old_%s_%s.png' % (old,self.time), dpi=125)
        else:
            fig.savefig('Secchi_mean_comparison_old_%s_%s.png' % (old,self.time), dpi=125)
        plt.close()

    def graphiqueTO(self,x, y, z, symbol, variable, calibration=False,old=False, lakeid="",
                    outputfolder=r'F:\output'):

        # Esthetic Parameters:
        sns.set(font_scale=2)
        sns.set_style("ticks")
        plt.grid(False)
        colorpalette = sns.color_palette("dark", 10)

        print(len(x), len(y), len(z))


        #Arrange data for multiple lakes analysed
        if lakeid == "":
            xall, yall = [item for sublist in x for item in sublist], [item for sublist in y for item in sublist]
        else:
            xall,yall  = x,y


        lineStart = -1
        lineEnd = 28

        #Figure
        fig, ax = plt.subplots(figsize=(15.0, 12))

        if variable == "Temperature (°C)":
            linearg={'fmt':'k-', 'color':colorpalette[0], 'label':"y= x", 'linewidth':4}
            base_plot_comparison(xall, yall,lineStart=lineStart,lineEnd=lineEnd, ax=None, linecolor="k")
            line_plot(lineStart=lineStart,lineEnd=lineEnd,ax=ax,linearg=linearg)
            ccmap = 'Blues'
        else:
            linearg = {'fmt':'k-', 'color':colorpalette[3], 'label':"y= x", 'linewidth':4}
            base_plot_comparison(xall, yall, lineStart=lineStart, lineEnd=lineEnd, ax=None, linecolor="k")
            line_plot(lineStart=lineStart, lineEnd=lineEnd, ax=ax, linearg=linearg)
            ccmap = 'Reds'


        markers = ["o", "v", "^", "s", "P", "*", ">", "X", "D", "<", "p", "d"]
        if lakeid == "":
            for i, c in enumerate(np.unique(symbol)):
                try:
                    cs = plt.scatter(x[i], y[i], c=z[i], marker=markers[c], s=90, cmap=ccmap, linewidths=1, edgecolors='k',
                                     alpha=0.8)
                except:
                    print("error in")
        else:
            try:
                i = int(symbol[0])
            except:
                i = 1
            print(i)
            if i > 11:
                print("here")
            print(markers[i])
            cs = plt.scatter(xall, yall, c=z, marker=markers[i], s=90, cmap=ccmap, linewidths=1, edgecolors='k',
                             alpha=0.8)

        cb = plt.colorbar(cs)
        plt.clim(0.0, 1.0)
        cb.ax.tick_params(labelsize=14)

        cb.ax.invert_yaxis()

        fig.suptitle("")
        fig.tight_layout(pad=2)

        plt.xlabel("Average Observed %s" % variable)
        plt.ylabel("Average Modeled %s" % variable)
        plt.ylim()
        plt.xlim()

        ax.legend(loc='best')  # 'upper left')
        if calibration:
            if variable == "Temperature (°C)":
                fig.savefig(os.path.join(outputfolder, 'Temperature_comparison_calibrated_old_%s_%s_%s.png' % (old,lakeid,self.time)), dpi=125)

            else:
                fig.savefig(os.path.join(outputfolder, 'Oxygen_comparison_calibrated_old_%s_%s_%s.png' % (old,lakeid, self.time)),
                            dpi=125)
        else:
            if variable == "Temperature (°C)":
                fig.savefig(os.path.join(outputfolder, 'Temperature_comparison_old_%s_%s_%s.png' % (old,lakeid,self.time)),
                            dpi=125)
            else:
                fig.savefig(os.path.join(outputfolder, 'Oxygen_comparison_old_%s_%s_%s.png' % (old,lakeid, self.time)),
                            dpi=125)
        plt.close()

    def contourplot_temp_vs_light_oxy(self,x_list,y_list,z_list,variables_list_in_order,label_axis_in_order,
                                      subfolder = "T_L_O_matrices",
                                      lakes_list=r'C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList'
                                                 r'_only_validation_12lakes.csv',individual=False):
        self.time = 2
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16
        plt.style.context('seaborn-paper')
        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
        plt.rcParams['axes.facecolor'] = 'silver'
        left = 0.10  # the left side of the subplots of the figure
        right = 0.80  # the right side of the subplots of the figure
        bottom = 0.03  # the bottom of the subplots of the figure
        top = 0.95  # the top of the subplots of the figure
        wspace = 0.2  # the amount of width reserved for blank space between subplots
        hspace = 0.5  # the amount of height reserved for white space between subplots

        # raw data and lake information
        lakes_data = pd.read_csv(lakes_list, encoding='ISO-8859-1')
        lake_id_list = list(lakes_data['lake_id'])


        [X, Y] = np.meshgrid(x_list, y_list)
        x_list = [float("{:.1f}".format(x_value)) for x_value in x_list]
        y_list = [float("{:.1f}".format(y_value)) for y_value in y_list]
        z_list = [float("{:.1f}".format(z_value)) for z_value in z_list]
        dict_all_z_value_by_lake_and_by_column = {}
        dict_all_z_value_by_lake = {}
        dict_z_value_by_lake, vmin_z_value_by_lake, vmax_z_value_by_lake = {}, {}, {}
        for z_value in z_list:
            dict_y_value_by_lake = {}
            for y_value in y_list:
                dict_raw_data_by_lake = {}
                for x_value in x_list:
                    dict_raw_data_by_lake,list_of_parameter_measured = get_file_by_value_of_xyz(x_value, y_value, z_value, dict_raw_data_by_lake,
                                                                variables_list_in_order, os.path.join(self.output_folder, subfolder), lake_id_list)

                for lake in lake_id_list:
                    for column in range(0, len(list_of_parameter_measured)):
                        if not list_of_parameter_measured[column] in dict_y_value_by_lake:
                            dict_y_value_by_lake[list_of_parameter_measured[column]]={"%s" % lake:[[item[column] for item in  dict_raw_data_by_lake["%s" % lake]]]}
                        else:
                            if not "%s"%lake in dict_y_value_by_lake[list_of_parameter_measured[column]]:
                                dict_y_value_by_lake[list_of_parameter_measured[column]]["%s" % lake] =  [[item[column] for item in dict_raw_data_by_lake["%s" % lake]]]

                            else:
                                dict_y_value_by_lake[list_of_parameter_measured[column]]["%s" % lake].append([item[column] for item in dict_raw_data_by_lake["%s" % lake]])




            dict_all_z_value_by_lake_and_by_column['%s'%z_value] = dict_y_value_by_lake



        dict_z_value_by_lake,vmin,vmax = create_dictionnary_of_z_value_by_lake(dict_all_z_value_by_lake_and_by_column,z_list,list_of_parameter_measured,lake_id_list,dict_z_value_by_lake)


        for column in list_of_parameter_measured:
            for lake in lake_id_list:
                if individual:
                    for z_value in z_list:
                        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                        base_contourplot(X,Y,dict_z_value_by_lake[column]['%s'%lake]['%s'%z_value], [z_value],lake, 4,
                                         vmin[column]['%s'%lake],vmax[column]['%s'%lake],ax,individual=individual)
                        ax.set_ylabel(label_axis_in_order['ylabel'],fontsize=BIGGER_SIZE)
                        ax.set_xlabel(label_axis_in_order['xlabel'],fontsize=BIGGER_SIZE)
                        plt.savefig("%s/habitat_treshold/contourplot_%s_lake_%s_%s_%s_time%s.png" % (
                            self.output_folder, column, lake,variables_list_in_order[2],z_value, self.time))

                fig, axs = plt.subplots(4, 4, figsize=(15, 15))
                print(vmax[column]['%s'%lake])
                contour = base_contourplot(X, Y, dict_z_value_by_lake[column]['%s'%lake], z_list, lake, 4,
                                 vmin[column]['%s'%lake], vmax[column]['%s'%lake], axs)

                # axs[3][3].set_visible(False)
                plt.subplots_adjust(right=right, left=left,top=top, wspace=wspace, hspace=hspace)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                print(np.linspace(vmin[column]['%s'%lake], vmax[column]['%s'%lake], 10))
                clb = plt.colorbar(contour,  cax=cbar_ax,boundaries=np.linspace(vmin[column]['%s'%lake], vmax[column]['%s'%lake], 10))
                # clb = plt.colorbar(contour, cax=cbar_ax)
                clb.set_label(column)
                # plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
                fig.text(0.5, 0.02, label_axis_in_order['xlabel'], ha='center', va='center',fontsize=BIGGER_SIZE)
                fig.text(0.05, 0.5, label_axis_in_order['ylabel'], ha='center', va='center', rotation='vertical',fontsize=BIGGER_SIZE)
                print("%s/habitat_treshold/contourplot_%s_lake_%s_%s_time%s.png" % (
                            self.output_folder, column, lake,variables_list_in_order[2], self.time))
                plt.savefig("%s/habitat_treshold/contourplot_%s_lake_%s_%s_time%s.png" % (
                            self.output_folder, column, lake,variables_list_in_order[2], self.time))
                plt.close('all')





    # def contourplot_temp_vs_oxy_light_old_version(self,temp_list,light_list,oxygen_list, subfolder = "T_L_O_matrices", lakes_list=r'C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList_only_validation_12lakes.csv',individual=False):
    #     #self.time = 2
    #     SMALL_SIZE = 12
    #     MEDIUM_SIZE = 14
    #     BIGGER_SIZE = 16
    #     plt.style.context('seaborn-paper')
    #     plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    #     plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    #     plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    #     plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    #     plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    #     plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    #     plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    #
    #     lakes_data = pd.read_csv(lakes_list, encoding='ISO-8859-1')
    #     plt.rcParams['axes.facecolor'] = 'silver'
    #     AreaDays_o2, Areaday_o2, NTGdays_o2 = {}, {}, {}
    #     first3 = 0
    #     vminAD, vmaxAD = {},{}
    #     vminAd, vmaxAd ={},{}
    #     vminNTG, vmaxNTG = {},{}
    #     for light in light_list:
    #         if light != int(light):
    #             light = "%s0" % light
    #
    #         [X, Y] = np.meshgrid(oxygen_list, temp_list)
    #         first2 = 0
    #         AreaDays, Areaday, NTGdays = {}, {}, {}
    #         for temp in temp_list:
    #             temperatures = {}
    #             first = 0
    #             for oxygen in oxygen_list:
    #                 if oxygen != str(oxygen):
    #                     if oxygen != int(oxygen):
    #                         oxygen = "%s0" % oxygen
    #                 with open('%s/%s/fish_niche_Area_Light%s_T%s_O%s_2001-2010.csv' % (
    #                 self.output_folder, subfolder, light, temp, oxygen), newline='') as f:
    #                     reader = csv.reader(f)
    #                     data = list(reader)
    #
    #                 for lake in range(0, len(lakes_data["lake_id"])):
    #                     lake += 1
    #                     if first == 0:
    #                         if data[lake][1] == 'AreaDays':
    #                             temperatures["%s" % data[lake][0]] = [[data[lake][1], data[lake][2], data[lake][3]]]
    #                         else:
    #                             temperatures["%s" % data[lake][0]] = [
    #                                 [float(data[lake][1]), float(data[lake][2]), float(data[lake][3])]]
    #
    #
    #                     else:
    #                         if data[lake][1] == 'AreaDays':
    #                             temperatures["%s" % data[lake][0]].append([data[lake][1], data[lake][2], data[lake][3]])
    #                         else:
    #                             temperatures["%s" % data[lake][0]].append(
    #                                 [float(data[lake][1]), float(data[lake][2]), float(data[lake][3])])
    #
    #                 first = 1
    #
    #             for lake in range(0, len(lakes_data["lake_id"])):
    #                 lake += 1
    #                 if first2 == 0:
    #                     print(temperatures["%s" % data[lake][0]][0])
    #                     AreaDays["%s" % data[lake][0]] = [[item[0] for item in temperatures["%s" % data[lake][0]]]]
    #                     Areaday["%s" % data[lake][0]] = [[item[1] for item in temperatures["%s" % data[lake][0]]]]
    #                     NTGdays["%s" % data[lake][0]] = [[item[2] for item in temperatures["%s" % data[lake][0]]]]
    #
    #                     vminAD["%s" % data[lake][0]] = min([item[0] for item in temperatures["%s" % data[lake][0]]])
    #                     vmaxAD["%s" % data[lake][0]] = max([item[0] for item in temperatures["%s" % data[lake][0]]])
    #                     vminAd["%s" % data[lake][0]] = min([item[1] for item in temperatures["%s" % data[lake][0]]])
    #                     vmaxAd["%s" % data[lake][0]] = max([item[1] for item in temperatures["%s" % data[lake][0]]])
    #                     vminNTG["%s" % data[lake][0]]= min([item[2] for item in temperatures["%s" % data[lake][0]]])
    #                     vmaxNTG["%s" % data[lake][0]] = max([item[2] for item in temperatures["%s" % data[lake][0]]])
    #
    #                 else:
    #                     AreaDays["%s" % data[lake][0]].append([item[0] for item in temperatures["%s" % data[lake][0]]])
    #                     Areaday["%s" % data[lake][0]].append([item[1] for item in temperatures["%s" % data[lake][0]]])
    #                     NTGdays["%s" % data[lake][0]].append([item[2] for item in temperatures["%s" % data[lake][0]]])
    #                     if min([item[0] for item in temperatures["%s" % data[lake][0]]]) < vminAD["%s" % data[lake][0]]:
    #                         vminAD["%s" % data[lake][0]] = min([item[0] for item in temperatures["%s" % data[lake][0]]])
    #                     if max([item[0] for item in temperatures["%s" % data[lake][0]]]) > vmaxAd["%s" % data[lake][0]]:
    #                         vmaxAD["%s" % data[lake][0]] = max([item[0] for item in temperatures["%s" % data[lake][0]]])
    #                     if min([item[1] for item in temperatures["%s" % data[lake][0]]]) < vminAd["%s" % data[lake][0]]:
    #                         vminAd["%s" % data[lake][0]] = min([item[1] for item in temperatures["%s" % data[lake][0]]])
    #                     if max([item[1] for item in temperatures["%s" % data[lake][0]]]) > vmaxAd["%s" % data[lake][0]]:
    #                         vmaxAd["%s" % data[lake][0]] = max([item[1] for item in temperatures["%s" % data[lake][0]]])
    #                     if min([item[2] for item in temperatures["%s" % data[lake][0]]]) < vminNTG["%s" % data[lake][0]]:
    #                         vminNTG["%s" % data[lake][0]] = min([item[2] for item in temperatures["%s" % data[lake][0]]])
    #                     if max([item[2] for item in temperatures["%s" % data[lake][0]]]) > vmaxNTG["%s" % data[lake][0]]:
    #                         vmaxNTG["%s" % data[lake][0]] = max([item[2] for item in temperatures["%s" % data[lake][0]]])
    #             first2 = 1
    #
    #         if first3 == 0:
    #             AreaDays_o2["%s" % light] = AreaDays
    #             Areaday_o2["%s" % light] = Areaday
    #             NTGdays_o2["%s" % light] = NTGdays
    #
    #
    #         if individual:
    #             for lake in lakes_data["lake_id"]:
    #                 Z = AreaDays["%s" % lake]
    #                 fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    #
    #                 cp1 =ax.contourf(X, Y, Z,vmin=vminAD["%s" % lake],vmax=vmaxAD["%s" % lake])
    #                 # fig.colorbar(cp)  # Add a colorbar to a plot
    #                 cp = plt.contour(X, Y, Z, colors='white')
    #                 fig.colorbar(cp1)
    #                 plt.clabel(cp, inline=True, fmt='%.2e')
    #                 plt.title("light threshold of %s µmol/m2s" % light)
    #                 # divider = make_axes_locatable(ax)
    #                 # cax = divider.append_axes("top", size="8%", pad=0)
    #                 # cax.get_xaxis().set_visible(False)
    #                 # cax.get_yaxis().set_visible(False)
    #                 # cax.set_facecolor('#e0e0e0')
    #                 #
    #                 # at = AnchoredText(("Oxygen threshold of %s mg/l")%oxygen, loc=10,
    #                 #                   prop=dict(backgroundcolor='#e0e0e0',color='#e0e0e0',size=12))
    #                 #
    #                 # cax.add_artist(at)
    #                 ax.set_ylabel('Temperature threshold (°C)')
    #                 ax.set_xlabel('Oxygen threshold (mg/L)')
    #                 plt.savefig("%s/habitat_treshold/contourplot_AreaDays_lake_%s_light_%s_time%s.png" % (
    #                 self.output_folder, lake, light, self.time))
    #
    #                 Z = Areaday["%s" % lake]
    #                 fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    #                 cp1 = ax.contourf(X, Y, Z,vmin=vminAd["%s" % lake],vmax=vmaxAd["%s" % lake])
    #                 cp = plt.contour(X, Y, Z, colors='white')
    #                 fig.colorbar(cp1)
    #                 plt.clabel(cp, inline=True)
    #                 plt.title("light threshold of %s µmol/m2s" % light)
    #                 ax.set_ylabel('Temperature threshold (°C)')
    #                 ax.set_xlabel('Oxygen threshold (mg/L)')
    #                 plt.savefig("%s/habitat_treshold/contourplot_Areaday_lake_%s_light_%s_time%s.png" % (
    #                 self.output_folder, lake, light, self.time))
    #
    #                 Z = NTGdays["%s" % lake]
    #                 fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    #                 cp1 = ax.contourf(X, Y, Z,vmin=vminNTG["%s" % lake],vmax=vmaxNTG["%s" % lake])
    #                 cp = plt.contour(X, Y, Z, colors='white')
    #                 fig.colorbar(cp1)
    #                 plt.clabel(cp, inline=True, fmt='%.2e')
    #                 plt.title("light threshold of %s µmol/m2s" % light)
    #                 ax.set_ylabel('Temperature threshold (°C)')
    #                 ax.set_xlabel('Oxygen threshold (mg/L)')
    #                 plt.savefig("%s/habitat_treshold/contourplot_NTGdays_lake_%s_light_%s_time%s.png" % (
    #                 self.output_folder, lake, float(light), self.time))
    #                 plt.close('all')
    #
    #
    #     final4 = True
    #     for lake in lakes_data["lake_id"]:
    #         fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    #         for light in range(0, len(light_list)):
    #             if final4:
    #                 if light_list[light] != int(light_list[light]):
    #                     light_list[light] = "%s0" % light_list[light]
    #             Z = AreaDays_o2["%s" % light_list[light]]["%s" % lake]
    #             axis_x = int(np.floor(light / 4))
    #             axis_y = int(light - (4 * axis_x))
    #             print(axs)
    #             ax = axs[axis_x][axis_y]
    #             cp1 = ax.contourf(X, Y, Z,vmin=vminAD["%s" % lake],vmax=vmaxAD["%s" % lake])
    #             # fig.colorbar(cp)  # Add a colorbar to a plot
    #             cp = ax.contour(X, Y, Z, colors='white')
    #             plt.clabel(cp, inline=True, fmt='%.2e')
    #             ax.title.set_text("Light threshold: %s µmol/m2s" % light_list[light])
    #
    #         final4 = False
    #
    #         left = 0.15  # the left side of the subplots of the figure
    #         right = 0.9  # the right side of the subplots of the figure
    #         bottom = 0.05  # the bottom of the subplots of the figure
    #         top = 0.8  # the top of the subplots of the figure
    #         wspace = 0.4 # the amount of width reserved for blank space between subplots
    #         hspace = 0.5  # the amount of height reserved for white space between subplots
    #
    #         #axs[3][3].set_visible(False)
    #         plt.subplots_adjust(right=0.85, left=left, wspace=wspace, hspace=hspace)
    #         cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    #         plt.colorbar(cp1, cax=cbar_ax)
    #         # plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    #         fig.text(0.5, 0.01, 'Oxygen threshold mg/L', ha='center', va='center')
    #         fig.text(0.01, 0.5, 'Temperature threshold (°C)', ha='center', va='center', rotation='vertical')
    #         plt.savefig(
    #             "%s/habitat_treshold/contourplot_AreaDays_lake_%s_light_time%s.png" % (self.output_folder, lake, self.time))
    #         plt.close('all')
    #
    #         fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    #         for light in range(0, len(light_list)):
    #             Z = Areaday_o2["%s" % light_list[light]]["%s" % lake]
    #             axis_x = int(np.floor(light / 4))
    #             axis_y = int(light - (4 * axis_x))
    #             print(axs)
    #             ax = axs[axis_x][axis_y]
    #             ax.contourf(X, Y, Z,vmin=vminAd["%s" % lake],vmax=vmaxAd["%s" % lake])
    #             # fig.colorbar(cp)  # Add a colorbar to a plot
    #             cp = ax.contour(X, Y, Z, colors='white')
    #             plt.clabel(cp, inline=True, fmt='%.2e')
    #             ax.title.set_text("Light threshold: %s µmol/m2s" % light_list[light])
    #
    #         # axs[3][3].set_visible(False)
    #
    #         plt.subplots_adjust(right=0.9)
    #         cbar_ax = fig.add_axes([0.95, 0.05, 0.05, 0.7])
    #         plt.colorbar(cp1, cax=cbar_ax)
    #         #plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    #         fig.text(0.5, 0.01, 'Oxygen threshold mg/L', ha='center', va='center')
    #         fig.text(0.01, 0.5, 'Temperature threshold (°C)', ha='center', va='center', rotation='vertical')
    #         plt.savefig(
    #             "%s/habitat_treshold/contourplot_Areaday_lake_%s_light_time%s.png" % (self.output_folder, lake, self.time))
    #         plt.close('all')
    #
    #         fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    #         for light in range(0, len(light_list)):
    #             Z = NTGdays_o2["%s" % light_list[light]]["%s" % lake]
    #             axis_x = int(np.floor(light / 4))
    #             axis_y = int(light - (4 * axis_x))
    #             print(axs)
    #             ax = axs[axis_x][axis_y]
    #             ax.contourf(X, Y, Z,vmin=vminNTG["%s" % lake],vmax=vmaxNTG["%s" % lake])
    #             # fig.colorbar(cp)  # Add a colorbar to a plot
    #             cp = ax.contour(X, Y, Z, colors='white')
    #             plt.clabel(cp, inline=True, fmt='%.2e')
    #             ax.title.set_text("Light threshold: %s µmol/m2s" % light_list[light])
    #
    #         # axs[3][3].set_visible(False)
    #
    #         plt.subplots_adjust(right=0.8,left=left,bottom=bottom,top=top)
    #         cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    #         plt.colorbar(cp1, cax=cbar_ax)
    #         #plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    #         fig.text(0.5, 0.01, 'Oxygen threshold (mg/L)', ha='center', va='center')
    #         fig.text(0.01, 0.5, 'Temperature threshold (°C)', ha='center', va='center', rotation='vertical')
    #         plt.savefig(
    #             "%s/habitat_treshold/contourplot_NTGdays_lake_%s_light_time%s.png" % (self.output_folder, lake, self.time))
    #         plt.close('all')

    def density_plot(self,summary,calibration,old):
        colorpalette = ["orange", "red", "forestgreen"]

        # sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")

        plt.grid(False)
        sns.distplot(summary["rmseT"], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     color=colorpalette[0], label=None)
        plt.ylabel("Density")
        plt.xlabel("RMSE (Celcius)")
        plt.savefig("densityrmseTc%so%s.png" % (calibration, old))
        plt.close()
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")

        plt.grid(False)
        sns.distplot(summary["rmseO"], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     color=colorpalette[1], label=None)
        plt.ylabel("Density")
        plt.xlabel("RMSE (mg*L-1)")
        plt.savefig("densityrmseOc%so%s.png" % (calibration, old))
        plt.close()
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")

        plt.grid(False)
        sns.distplot(summary["rmseS"], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     color=colorpalette[2], label=None)
        plt.ylabel("Density")
        plt.xlabel("RMSE (m)")
        plt.savefig("densityrmseSc%so%s.png" % (calibration, old))
        plt.close()
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.grid(False)
        sns.distplot(summary["nrmseT"], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     color=colorpalette[0], label="Temperature")
        sns.distplot(summary["nrmseO"], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     color=colorpalette[1], label="Oxygen")
        sns.distplot(summary["nrmseS"], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     color=colorpalette[2], label="Secchi depth")
        plt.ylabel("Density")
        plt.xlabel("Normalised RMSE")
        plt.savefig("densityc%so%s.png" % (calibration, old))
        plt.close()

        dataall = pd.read_csv("2017SwedenList.csv", encoding='ISO-8859-1')
        dataall["area"] = dataall["area"]
        dataall["volume"] = dataall["volume"] * 1e-9
        dataall["sedimentArea"] = dataall["sedimentArea"] * 1e-18
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.grid(False)
        sns.distplot(dataall["area"], color="black", hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     label=None, norm_hist=True)
        plt.xscale('log')

        plt.ylabel("Density")
        plt.xlabel("Area ( km2)")
        plt.savefig("area.png")
        plt.close()
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.grid(False)
        sns.distplot(dataall["depth"], color="black", hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     label=None)
        plt.ylabel("Density")
        plt.xlabel("Depth (m)")
        plt.savefig("depth.png")
        plt.close()
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.grid(False)
        sns.distplot(dataall["longitude"], color="black", hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     label=None)
        plt.ylabel("Density")
        plt.xlabel("Longitude")
        plt.savefig("lon.png")
        plt.close()
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.grid(False)
        sns.distplot(dataall["latitude"], color="black", hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     label=None)
        plt.ylabel("Density")
        plt.xlabel("Latitude")
        plt.savefig("latitude.png")
        plt.close()
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.grid(False)
        sns.distplot(dataall["volume"], color="black", hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     label=None)
        plt.ylabel("Density")
        plt.xlabel("Volume (km3)")
        plt.savefig("vol.png")
        plt.close()
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.grid(False)
        mean = dataall["Mean"].dropna()
        sns.distplot(mean, color="black", hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label=None)
        plt.ylabel("Density")
        plt.xlabel("Mean depth (m)")
        plt.savefig("mean.png")
        plt.close()
        sns.set(font_scale=2)
        plt.figure(figsize=(10, 8))
        sns.set_style("ticks")
        plt.grid(False)
        sns.distplot(dataall["sedimentArea"], color="black", hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3}, label=None)
        plt.ylabel("Density")
        plt.xlabel("sedimentArea (x10^12 km\u00b2) ")
        plt.savefig("sediment.png")
        plt.close()

    def observation_plot(self,finalT,T_final):
        fig1 = plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
        fig1 = sns.scatterplot(data=finalT, x="dates", y="obsT", hue="lakepath", palette="dark")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)
        plt.xlim([finalT['dates'].min(), finalT['dates'].max()])
        plt.savefig("T_obs.png")
        finalT = T_final.dropna(subset=['obsO'])
        fig2 = plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
        fig2 = sns.scatterplot(data=T_final, x="dates", y="obsO", hue="lakepath", palette="dark")
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)
        plt.xlim([finalT['dates'].min(), finalT['dates'].max()])
        plt.savefig("O2_obs.png")


def violin_parallel():
    """
    Simple function to call a parallel calibration of all lakes.
    :return: None
    """
    print(num_cores)
    # for lake in lakes_list:
    #     run_calibrations(lake)
    Parallel(n_jobs=num_cores, verbose=10)(delayed(violin_plot)(lakenumber) for lakenumber in range(0, len(lakes_list)))


def violin_plot(rcp="85", lakes_list1="2017SwedenList.csv", output_path=r"F:\output"):
    # lake = lakes_list[lake_number]
    lakes = pd.read_csv(lakes_list1, encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    sns.set_color_codes("colorblind")
    sns.set_context("paper", 2.5)
    variables = [["Tzt.csv", "Change in Surface Temperature ($^\circ$C)"],
                 ["O2zt.csv", "Change in Bottom Oxygen\n Concentration (mg m-2)"],
                 ["His.csv", "Change in Ice Cover Duration (day)"]]
    model_data = [["model", "lake", "volume", "depth", "scenario", variables[0][1], variables[1][1], variables[2][1]]]
    lakesss_data = pd.DataFrame(columns=["lake", "model", "volume", "depth",
                                         "dateM", "dateD", "historicalT", "rcp45T", "rcp85T", "diff45T", "diff85T",
                                         "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                                         "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
    # kernel = [["model", "lake", "scenario", variables[0][1], variables[1][1], variables[2][1]]]
    aaaa = 0
    for modelid in [2]:
        m1, m2 = models[modelid]
        # if 1==1:
        try:

            n = 1
            if aaaa == 0:
                lakess_data = pd.DataFrame(
                    columns=["lake", "model", "volume", "depth",
                             "dateM", "dateD", "historicalT", "rcp45T", "rcp85T",
                             "diff45T", "diff85T",
                             "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                             "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
            else:
                lakesss_data = lakesss_data.append(lakess_data, ignore_index=True)
            aaaa = 0
            lake_number = -1
            # if 1==1:
            for lake in lakes_list:
                lake_number += 1
                lake_data = pd.DataFrame(
                    columns=["lake", "model", "volume", "depth",
                             "dateM", "dateD", "historicalT", "rcp45T", "rcp85T",
                             "diff45T", "diff85T",
                             "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                             "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
                print(m2, lake, n, lake_number)
                n += 1

                volume = lakes_data.get('volume').get(lake)
                depth = lakes_data.get('depth').get(lake)
                eh = lakes_data.get('ebhex').get(lake)
                eh = eh[2:] if eh[:2] == '0x' else eh
                while len(eh) < 6:
                    eh = '0' + eh
                d1, d2, d3 = eh[:2], eh[:4], eh[:6]

                if rcp == "85":
                    sce = 8
                else:
                    sce = 5
                for scenarioid in [1, sce]:
                    exA, y1A, exB, y1B = scenarios[scenarioid]
                    # y2A = y1A + 4
                    y2B = y1B + 4
                    outdir = os.path.join(output_path, d1, d2, d3,
                                          'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (m1, exA, exB, m2, y1A, y2B))

                    lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                        list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                        list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                        list(lakes["Mean"])[lake_number],
                                        list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                        list(lakes["volume"])[lake_number], scenarioid, scenarioid=scenarioid,
                                        modelid=modelid)

                    # lake.variables_by_depth()
                    # lakeinfo.runlake(modelid,scenarioid)

                    if scenarioid == 1:
                        dstart = date(y1A, 1, 1)
                        dend = date(y2B, 12, 31)

                        # this will give you a list containing all of the dates
                        dd = [dstart + timedelta(days=x) for x in range((dend - dstart).days + 1)]

                        date_stringsM = [d.strftime('%m') for d in dd]
                        date_stringsD = [d.strftime('%d') for d in dd]
                        lake_data["dateM"] = date_stringsM
                        lake_data["dateD"] = date_stringsD
                        lake_data["lake"] = lake
                        lake_data["model"] = modelid
                        lake_data["volume"] = volume
                        lake_data["depth"] = depth

                    for variable in [0, 1, 2]:
                        data = pd.read_csv(os.path.join(outdir, variables[variable][0]), header=None)

                        if variable == 0:
                            lake_data["%sT" % exA] = data[0]
                        elif variable == 1:
                            lake_data["%sO" % exA] = data.iloc[:, -1] * 0.001
                        else:
                            icecoverduration = (data[6].sum()) / 10
                            lake_data["%sI" % exA] = icecoverduration

                data_summary = lake_data.mean()
                for rcp in [rcp]:
                    for letter in ["T", "O", "I"]:
                        data_summary["diff%s%s" % (rcp, letter)] = data_summary["rcp%s%s" % (rcp, letter)] - \
                                                                   data_summary[
                                                                       "historical%s" % letter]
                if aaaa == 0:
                    lakess_data = lake_data
                    aaaa += 1
                else:
                    lakess_data = lakess_data.append(lake_data, ignore_index=True)

                for rcp in [rcp]:
                    model_code = {1: 'KNM',
                                  2: 'DMI',
                                  3: 'MPI',
                                  4: 'MOH',
                                  5: 'IPS',
                                  6: 'CNR'}
                    model_data.append(
                        [model_code.get(modelid), lake, volume, depth, "rcp%s" % rcp, data_summary["diff%sT" % rcp],
                         data_summary["diff%sO" % rcp], data_summary["diff%sI" % rcp]])

        #
        except:
            print("model %s doesnt exist" % (m1 + m2))

    import seaborn as sns
    from matplotlib.cm import ScalarMappable
    import matplotlib.pyplot as plt

    headers = model_data.pop(0)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    final_data = pd.DataFrame(model_data, columns=headers)
    lakesss_data.to_csv("annually_average_T_Ice_cover_Oxygen_originall_%s.csv" % rcp)
    final_data.to_csv("annually_average_T_Ice_cover_Oxygen_%s.csv" % rcp)
    plotT = sns.catplot(x="model", y=variables[0][1], col="scenario", data=final_data, kind="violin", color="orange",
                        split=True,
                        height=8, aspect=.8)
    plotT.savefig(os.path.join(output_path, "violinT1_%s_%s.png" % (rcp, timestr)))
    plt.close(fig=plotT)
    plotI2 = sns.violinplot(x="model", y=variables[0][1], col="scenario", data=final_data, inner=None, color=".8")
    ploT2 = sns.stripplot(x="model", y=variables[0][1], data=final_data, hue="volume", palette="viridis")
    # Colormap for comparison
    # Data
    y_min = min(final_data["volume"])
    y_max = max(final_data["volume"])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(y_min, y_max)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2)
    plotT2.savefig(os.path.join(output_path, "violinT1_%s_%s.png" % (rcp, timestr)))

    print("save T")
    plotO = sns.catplot(x="model", y=variables[1][1], col="scenario", data=final_data, kind="violin", color="red",
                        split=True,
                        height=8, aspect=.9)
    plotO.savefig(os.path.join(output_path, "violinO1_%s_%s.png" % (rcp, timestr)))
    print("save O")
    plotI = sns.catplot(x="model", y=variables[2][1], col="scenario", data=final_data, kind="violin",
                        color="forestgreen", split=True,
                        height=8, aspect=.7)
    plotI.savefig(os.path.join(output_path, "violinI1_%s_%s.png" % (rcp, timestr)))
    print("save Ice")
    print("end violin")


def violin_plot45(lakes_list1="2017SwedenList.csv", output_path=r"F:\output"):
    # lake = lakes_list[lake_number]
    lakes = pd.read_csv(lakes_list1, encoding='ISO-8859-1')
    lakes_data = lakes.set_index("lake_id").to_dict()
    lakes_list = list(lakes_data.get("name").keys())
    sns.set_color_codes("colorblind")
    sns.set_context("paper", 2.5)
    variables = [["Tzt.csv", "Change in Surface Temperature ($^\circ$C)"],
                 ["O2zt.csv", "Change in Bottom Oxygen\n Concentration (mg m-2)"],
                 ["His.csv", "Change in Ice Cover Duration (day)"]]
    model_data = [["model", "lake", "volume", "depth", "scenario", variables[0][1], variables[1][1], variables[2][1]]]
    lakesss_data = pd.DataFrame(columns=["lake", "model", "volume", "depth",
                                         "dateM", "dateD", "historicalT", "rcp45T", "rcp85T", "diff45T", "diff85T",
                                         "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                                         "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
    # kernel = [["model", "lake", "scenario", variables[0][1], variables[1][1], variables[2][1]]]
    aaaa = 0
    for modelid in [2]:
        m1, m2 = models[modelid]
        # if 1==1:
        try:

            n = 1
            if aaaa == 0:
                lakess_data = pd.DataFrame(
                    columns=["lake", "model", "volume", "depth",
                             "dateM", "dateD", "historicalT", "rcp45T", "rcp85T",
                             "diff45T", "diff85T",
                             "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                             "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
            else:
                lakesss_data = lakesss_data.append(lakess_data, ignore_index=True)
            aaaa = 0
            lake_number = -1
            # if 1==1:
            for lake in lakes_list:
                lake_number += 1
                lake_data = pd.DataFrame(
                    columns=["lake", "model", "volume", "depth",
                             "dateM", "dateD", "historicalT", "rcp45T", "rcp85T",
                             "diff45T", "diff85T",
                             "historicalO", "rcp45O", "rcp85O", "diff45O", "diff85O",
                             "historicalI", "rcp45I", "rcp85I", "diff45I", "diff85I"])
                print(m2, lake, n, lake_number)
                n += 1

                volume = lakes_data.get('volume').get(lake)
                depth = lakes_data.get('depth').get(lake)
                eh = lakes_data.get('ebhex').get(lake)
                eh = eh[2:] if eh[:2] == '0x' else eh
                while len(eh) < 6:
                    eh = '0' + eh
                d1, d2, d3 = eh[:2], eh[:4], eh[:6]

                for scenarioid in [1, 5, 8]:
                    exA, y1A, exB, y1B = scenarios[scenarioid]
                    # y2A = y1A + 4
                    y2B = y1B + 4
                    outdir = os.path.join(output_path, d1, d2, d3,
                                          'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (m1, exA, exB, m2, y1A, y2B))

                    lakeinfo = LakeInfo(lakes_list[lake_number], list(lakes["lake_id"])[lake_number],
                                        list(lakes["subid"])[lake_number], list(lakes["ebhex"])[lake_number],
                                        list(lakes["area"])[lake_number], list(lakes["depth"])[lake_number],
                                        list(lakes["Mean"])[lake_number],
                                        list(lakes["longitude"])[lake_number], list(lakes["latitude"])[lake_number],
                                        list(lakes["volume"])[lake_number], scenarioid, scenarioid=scenarioid,
                                        modelid=modelid)

                    # lake.variables_by_depth()
                    # lakeinfo.runlake(modelid,scenarioid)

                    if scenarioid == 1:
                        dstart = date(y1A, 1, 1)
                        dend = date(y2B, 12, 31)

                        # this will give you a list containing all of the dates
                        dd = [dstart + timedelta(days=x) for x in range((dend - dstart).days + 1)]

                        date_stringsM = [d.strftime('%m') for d in dd]
                        date_stringsD = [d.strftime('%d') for d in dd]
                        lake_data["dateM"] = date_stringsM
                        lake_data["dateD"] = date_stringsD
                        lake_data["lake"] = lake
                        lake_data["model"] = modelid
                        lake_data["volume"] = volume
                        lake_data["depth"] = depth

                    for variable in [0, 1, 2]:
                        data = pd.read_csv(os.path.join(outdir, variables[variable][0]), header=None)

                        if variable == 0:
                            lake_data["%sT" % exA] = data[0]
                        elif variable == 1:
                            lake_data["%sO" % exA] = data.iloc[:, -1] * 0.001
                        else:
                            icecoverduration = (data[6].sum()) / 10
                            lake_data["%sI" % exA] = icecoverduration

                data_summary = lake_data.mean()
                for rcp in ["45", "85"]:
                    for letter in ["T", "O", "I"]:
                        data_summary["diff%s%s" % (rcp, letter)] = data_summary["rcp%s%s" % (rcp, letter)] - \
                                                                   data_summary[
                                                                       "historical%s" % letter]
                        if letter == "I":
                            data_summary["ice%s%s" % (rcp, letter)] = data_summary["rcp%s%s" % (rcp, letter)]
                            data_summary["icehisto%s" % (letter)] = data_summary["historical%s" % letter]
                if aaaa == 0:
                    lakess_data = lake_data
                    aaaa += 1
                else:
                    lakess_data = lakess_data.append(lake_data, ignore_index=True)

                for rcp in ["45", "85"]:
                    model_code = {1: 'KNM',
                                  2: 'DMI',
                                  3: 'MPI',
                                  4: 'MOH',
                                  5: 'IPS',
                                  6: 'CNR'}
                    model_data.append(
                        [model_code.get(modelid), lake, volume, depth, "rcp%s" % rcp, data_summary["diff%sT" % rcp],
                         data_summary["diff%sO" % rcp], data_summary["diff%sI" % rcp]])

        #
        except:
            print("model %s doesnt exist" % (m1 + m2))
    headers = model_data.pop(0)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    final_data = pd.DataFrame(model_data, columns=headers)
    lakesss_data.to_csv("annually_average_T_Ice_cover_Oxygen_originall.csv")
    final_data.to_csv("annually_average_T_Ice_cover_Oxygen.csv")
    plotT = sns.catplot(x="model", y=variables[0][1], col="scenario", data=final_data, kind="violin", color="orange",
                        split=True,
                        height=8, aspect=.8)
    plotT.savefig(os.path.join(output_path, "violinT4589_%s.png" % (timestr)))
    print("save T")
    plotO = sns.catplot(x="model", y=variables[1][1], col="scenario", data=final_data, kind="violin", color="red",
                        split=True,
                        height=8, aspect=.9)
    plotO.savefig(os.path.join(output_path, "violinO4589_%s.png" % (timestr)))
    print("save O")
    plotI = sns.catplot(x="model", y=variables[2][1], col="scenario", data=final_data, kind="violin",
                        color="forestgreen", split=True,
                        height=8, aspect=.7)
    plotI.savefig(os.path.join(output_path, "violinI4589_%s.png" % (timestr)))
    print("save Ice")
    print("end violin")

def exploratory(lakelist="2017SwedenList_only_validation_12lakes.csv"):
    df = pd.read_csv(lakelist, encoding='ISO-8859-1')
    df['Lake_group'] = 2
    df.loc[df['volume'] < 1.0e7, 'Lake_group'] = 1
    df.loc[df['volume'] > 5.0e9, 'Lake_group'] = 3
    df[['area','depth','longitude','latitude','Mean','volume','Turnover',
        'sedimentArea','Subcatchment_Area','Catchment_Area','C.L','SC.L','Lake_group']].hist(bins=20,figsize=(15, 15))
    #plt.show()
    plt.savefig("explo.png")

    continous_var = ['area','depth','longitude','latitude','Mean','volume','Turnover',
        'sedimentArea','Subcatchment_Area','Catchment_Area','C.L','SC.L']
    continous_var_label = ['Area', 'Max_Depth', 'Longitude', 'Latitude', 'Mean_depth', 'Volume', 'Residence_Time',
                           'Sediment_Area', 'Subcatchment_Area', 'Catchment_Area', 'Catchment_Lake_ratio',
                           'Subcatchment_Lake_ratio']
    df2 = df[continous_var].describe()
    df2.to_csv("describe.csv")

    df['Lake_group'] = 2
    df.loc[df['volume'] < 1.0e7, 'Lake_group'] = 1
    df.loc[df['volume'] > 5.0e9, 'Lake_group'] = 3
   # sns.scatterplot(x="area", y="depth", data=df, hue="Lake_group")
    #plt.savefig("explo2.png")
    sns.pairplot(df[['area','depth','longitude','latitude','Mean','volume','Turnover',
        'sedimentArea','Subcatchment_Area','Catchment_Area','C.L','SC.L','Lake_group']], hue="Lake_group")

    plt.savefig("explo2.png")

    plt.figure(figsize=(32, 50))

    for i, col in enumerate(continous_var):
        plt.subplot(12, 4, i * 2 + 1)
        #plt.subplots_adjust(hspace=.25, wspace=.3)

        plt.grid(True)
        plt.title(col)
        sns.kdeplot(df.loc[df["Lake_group"] == 1, col], label="Small", color="green", shade=True, kernel='gau', cut=0)
        sns.kdeplot(df.loc[df["Lake_group"] == 2, col], label="Medium", color="red", shade=True, kernel='gau', cut=0)
        sns.kdeplot(df.loc[df["Lake_group"] == 3, col], label="Large", color="blue", shade=True, kernel='gau', cut=0)
        plt.subplot(12, 4, i * 2 + 2)
        sns.boxplot(y=col, data=df, x="Lake_group", palette=["green", "red","blue"])
    plt.savefig("explo3.png")

    plt.figure(figsize=(32, 50))

    continous_var = ['area', 'depth','longitude','latitude', 'Mean', 'volume', 'Turnover',
                     'sedimentArea', 'Subcatchment_Area', 'Catchment_Area', 'C.L', 'SC.L']
    continous_var_label = ['Area', 'Max_Depth', 'Longitude', 'Latitude', 'Mean_depth', 'Volume', 'Residence_Time',
                     'Sediment_Area', 'Subcatchment_Area', 'Catchment_Area', 'Catchment_Lake_ratio', 'Subcatchment_Lake_ratio']
    palette = iter(sns.husl_palette(len(continous_var)))
    for i, col in enumerate(continous_var):
        plt.subplot(21, 2, i  + 1)
        # plt.subplots_adjust(hspace=.25, wspace=.3)

        plt.grid(True)
        plt.title(col)
        data = df[col]
        #sns.kdeplot(data, color=palette[i], shade=True, kernel='gau', cut=0)
        sns.distplot(data, hist=True, kde=True,
             bins=int(180/5), color = palette[i],
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
        plt.set_title(continous_var_label[i])

    plt.savefig("explo4.png")

if __name__ == "__main__":
    print("hello")
    temp_list = [1,2, 4, 6, 8, 10, 12, 14, 15]


    oxygen_list = [3,   3.5,    4,  4.5,    5,  5.5,     6,  6.5,    7, 7.5,   8,   8.5,    9,  9.5,    10, 10.5]
    #Graphics(r"F:\output").contourplot_temp_vs_light_oxy( temp_list, light_list, oxygen_list)

    light_list = [0.5, 1, 2, 4, 8, 12, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    Graphics(r"F:\output").contourplot_temp_vs_light_oxy(oxygen_list,temp_list,light_list,
                                                         ['Oxygen','Temperature','Light'],
                                                         {'xlabel':'Oxygen threshold (mg/L)','ylabel':'Temperature threshold (°C)','zlabel':'Light threshold (µmol/m2s)'},individual=True)

    light_list = [0.5, 1, 2, 4, 8, 12, 16, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    Graphics(r"F:\output").contourplot_temp_vs_light_oxy(light_list, temp_list, oxygen_list,
                                                         ['Light', 'Temperature', 'Oxygen'],
                                                         {'xlabel': 'Light threshold (µmol/m2s)',
                                                          'ylabel': 'Temperature threshold (°C)',
                                                          'zlabel': 'Oxygen threshold (mg/L)'},individual=True)


    #
    # Graphics(r"F:\output").contourplot_temp_vs_oxy_light(temp_list, light_list, oxygen_list)

# label_axis_in_order = {'xlabel':'Light threshold (µmol/(m2s))', 'ylabel':'Temperature threshold (°C)','zlabel':''}
#                     variables_list_in_order = {'x':'Light','y':'Temperature','z':'Oxygen'}