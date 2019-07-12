# !/usr/bin/env python3
"""
    File name: FishNiche_output_analysis_scripts2.py
    Author: Marianne Cote
    Date created: 2018-08-30
    Python Version: 3.6

    Contains all functions used to produce the figures and files.

"""

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.api import add_constant, OLS
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib.patches import Patch
import statistics as st
import matplotlib.lines as mlines
import sklearn.metrics
from math import sqrt, isnan, ceil
import matplotlib.pyplot as plt
import os
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
# from pandas.tseries.offsets import MonthBegin,MonthEnd
# import matplotlib.dates as dates
import scipy.stats as stats
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter, DayLocator, WeekdayLocator, MONDAY
# import matplotlib as mpl
from datetime import date, datetime
from os import path

from csv import reader
# from function_runlakesGoran_par_test import runlakesGoran_par
import pandas.plotting._converter

pandas.plotting._converter.register()
plt.style.use('seaborn-poster')

outputfolderdata = r'..\figure_calibration\Figure_test_oxygen'
datafolder = r'C:\Users\macot620\Documents\GitHub\Fish_niche\output'
outputfolder = r'C:\Users\macot620\Documents\GitHub\Fish_niche\output'
models = {1:('ICHEC-EC-EARTH', 'r1i1p1_KNMI-RACMO22E_v1_day'),
          2:('ICHEC-EC-EARTH', 'r3i1p1_DMI-HIRHAM5_v1_day'),
          3:('MPI-M-MPI-ESM-LR', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day'),
          4:('MOHC-HadGEM2-ES', 'r1i1p1_SMHI-RCA4_v1_day'),
          5:('IPSL-IPSL-CM5A-MR', 'r1i1p1_IPSL-INERIS-WRF331F_v1_day'),
          6:('CNRM-CERFACS-CNRM-CM5', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day')}
scenarios = {1:('historical', 1971, 'historical', 1976),
             2:('historical', 2001, 'rcp45', 2006),
             3:('rcp45', 2031, 'rcp45', 2036),
             4:('rcp45', 2061, 'rcp45', 2066),
             5:('rcp45', 2091, 'rcp45', 2096),
             6:('rcp85', 2031, 'rcp85', 2036),
             7:('rcp85', 2061, 'rcp85', 2066),
             8:('rcp85', 2091, 'rcp85', 2096)}

years = YearLocator() # every year
# months = MonthLocator() # every month
days = DayLocator()
yearsFmt = DateFormatter('%Y')
# monthsFmt=DateFormatter('%M')
months = MonthLocator()
mondays = WeekdayLocator(MONDAY) # major ticks on the mondays
alldays = DayLocator() # minor ticks on the days
weekFormatter = DateFormatter('%b') # e.g., Jan 12
dayFormatter = DateFormatter('%d')
# format the coords message box

def graphique(x, y, xerr, yerr, coeff, r_value, variable_test, slope, intercept, variable_analyzed):
    """
        create figure for regression(specific function).
        :param x: x
        :param y: y
        :param xerr: xerr
        :param yerr: yerr
        :param coeff: coeff
        :param r_value: r_value
        :param variable_test: tested variable
        :param slope: slope of regression
        :param intercept: intercept of regression
        :param variable_analyzed: analysed variable

        :return: png file with figure
    """
    # 2018-08-30 MC

    plt.style.use('seaborn-poster')
    params = {
        'axes.labelsize': 30,
        'text.fontsize': 30,
        'legend.fontsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'text.usetex': False,
        'figure.figsize': [14, 10]  # instead of 4.5, 4.5
    }

    plt.rcParams.update(params)
    line_start = 0
    if variable_analyzed == 'SECCHI':
        lineEnd = 14
    else:
        lineEnd = 14000
    fig, ax = plt.subplots(figsize=(18.0, 10.0))
    plt.plot([line_start, lineEnd], [line_start, lineEnd], 'k--', color='b', label="1x+0", zorder=0)\

    (_, caps, _)= plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color="k", elinewidth=2, markersize=1, capsize=5,
                                  zorder=10)
    for cap in caps:
        cap.set_markeredgewidth(2)

    fig.suptitle("")
    fig.tight_layout(pad=2)
    ax.grid(True)
    fig.savefig('filename1.png', dpi=125)

    x = add_constant(x) # constant intercept term
    # Model: y ~ x + c
    model = OLS(y, x)
    fitted = model.fit()
    # rsquared = fitted.rsquared()
    x_pred = np.linspace(x.min(), x.max(), 50)
    x_pred2 = add_constant(x_pred)
    y_pred = fitted.predict(x_pred2)

    ax.plot(x_pred, y_pred, '-', color='r', linewidth=2,
              label="linear regression(%0.3f x + %0.3f)" %(slope, intercept))
    fig.savefig('filename2.png', dpi=125)

    print(fitted.params) # the estimated parameters for the regression line
    print(fitted.summary()) # summary statistics for the regression

    y_hat = fitted.predict(x) # x is an array from line 12 above
    y_err = y - y_hat
    mean_x = x.T[1].mean()
    n = len(x)
    dof = n - fitted.df_model - 1

    t = stats.t.ppf(1 - 0.025, df=dof)
    s_err = np.sum(np.power(y_err, 2))
    conf = t * np.sqrt((s_err /(n - 2))*(1.0 / n +(np.power((x_pred - mean_x), 2)/
                                                        ((np.sum(np.power(x_pred, 2)))- n *(
                                                         np.power(mean_x, 2))))))
    upper = y_pred + abs(conf)
    lower = y_pred - abs(conf)
    ax.fill_between(x_pred, lower, upper, color='k', alpha=0.5, label="Confidence interval")
    fig.savefig('filename3_%s.png' % coeff, dpi=125)

    sdev, lower, upper = wls_prediction_std(fitted, exog=x_pred2, alpha=0.025)
    ax.fill_between(x_pred, lower, upper, color='k', alpha=0.25, label="Prediction interval")

    plt.xlabel("Observed Secchi depth(m)")
    plt.ylabel("Modeled Secchi depth(m)")

    plt.plot([], [], color='w', label="$R^2$ : 0.999")
    ax.legend(loc='lower right', prop={'size': 20})
    plt.tight_layout()
    fig.savefig('Mean_regression_%s_%s_%s.png' %(variable_analyzed, variable_test, coeff), dpi=125)
    return fitted.summary()

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax1=None):
    """
    create figure for regression(specific function).
        :param x: x
        :param y: y
        :param xerr: xerr
        :param yerr: yerr
        :param coeff: coeff
        :param r_value: r_value
        :param variable_test: tested variable
        :param slope: slope of regression
        :param intercept: intercept of regression
        :param variable_analyzed: analysed variable

        :return: png file with figure
    """
    ax = ax1 if ax1 is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr)or len(yerr)== len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr)== 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def price(x):
    return '$%1.2f' % x

def area_at_depth(surface_area, depth_pos, max_depth):
    """
    calcul the Area at a certain depth(and volume for a depth of 1m).
    using a cone model, calculate the proportional area at the depth_pos.
   (use thales property on surface area)
    after, consider a cylinder with a depth of 1 meter(area*depth=1)to conclude that cylinder area = volume
    :param surface_area: surface_area of the lake
    :type surface_area: float
    :param depth_pos: specific depth for the
    """
    depth = depth_pos
    result =(surface_area *(depth - max_depth)** 2)/(max_depth ** 2)
    return result

def FishNiche_csv_result(scenarioid, modelid, lakelistfile, k_SOD, k_BOD, m, n, i, j, csvp, csvr):
    global fig1
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2B = y1B + 4
    timeaxis = pd.date_range("%s-01-01" % y1A, "%s-12-31" % y2B, freq="D")
    timestep = len(timeaxis)
    fig1, fig2, fig3, fig4 = plt.figure(), plt.figure(), plt.figure(), plt.figure()

    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)

    # lakeswehavedatafor = [698]
    lakeswehavedatafor = [14939, 67035, 16765, 698, 33494, 33590, 30704, 19167, 31895, 310, 32276, 99045, 99516,
                          33590, 33494]
    # lakeswehavedatafor = [698, 67035, 31895, 310, 32276]

    for lakenum in np.arange(1, nlakes):
        lake_id, subid, name, eh, area, depth, longitude, latitude, volume \
            = lakes[lakenum].strip().split(',')
        if int(lake_id)in lakeswehavedatafor:

            # getOutputPathFromEbHex
            eh = eh[2:] if eh[:2] == '0x' else eh
            while len(eh)< 6:
                eh = '0' + eh
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]

            outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
            print(outdir)

            O2model = pd.read_csv(path.join(outdir, 'O2zt.csv'), header=None)
            Tmodel = pd.read_csv(path.join(outdir, 'Tzt.csv'), header=None)

            filename = '../sweden_inflow_data/Validation_data_for_lookup.xlsx'
            # need to be change if the file change
            worksheet = pd.read_excel(filename, sheet_name=lake_id)
            dates = worksheet['date']
            depthdata = worksheet['depth(max)']
            O2raw = worksheet['O2(mg/l)'] * 1000
            Traw = worksheet['Water temp(°C)']
            anydata_t, anydata_o = False, False
            T_model_samples, T_data_samples, T_depths = [], [], []
            O2_model_samples, O2_data_samples, O2_depths = [], [], []
            crashed = False
            try:
                for ii in np.arange(0, len(dates)):
                    dnum = datetime.strptime(dates[ii], "%Y-%m-%d").date()
                    start = timeaxis[0].date()
                    dateindex =(dnum - start).days
                    if dateindex >= 1 and dateindex <= timestep:
                        if lake_id == '698':
                            if depthdata[ii] <= 2:
                                depth = ceil(depthdata[ii])
                                T_data = Traw[ii]
                                T_model = Tmodel.loc[dateindex, depth - 1]

                                if not isnan(T_data):
                                    T_data_samples.append(T_data)
                                    T_model_samples.append(T_model)
                                    T_depths.append(depth)
                                    anydata_t = True

                            depthO = ceil(depthdata[ii])
                            O2_data = O2raw[ii]
                            O2_model = O2model.loc[dateindex, depthO - 1]
                            if not isnan(O2_data):
                                O2_data_samples.append(O2_data)
                                O2_model_samples.append(O2_model)
                                O2_depths.append(depthO)
                                anydata_o = True

                        elif lake_id == '67035':
                            if depthdata[ii] <= 1 or depthdata[ii] >= 60:
                                depth = ceil(depthdata[ii])
                                T_data = Traw[ii]
                                T_model = Tmodel.loc[dateindex, depth - 1]

                                if not isnan(T_data):
                                    T_data_samples.append(T_data)
                                    T_model_samples.append(T_model)
                                    T_depths.append(depth)
                                    anydata_t = True

                            depthO = ceil(depthdata[ii])
                            O2_data = O2raw[ii]
                            O2_model = O2model.loc[dateindex, depthO - 1]
                            if not isnan(O2_data):
                                O2_data_samples.append(O2_data)
                                O2_model_samples.append(O2_model)
                                O2_depths.append(depthO)
                                anydata_o = True

                        else:
                            depth = ceil(depthdata[ii])
                            O2_data = O2raw[ii]
                            T_data = Traw[ii]
                            O2_model = O2model.loc[dateindex, depth - 1]
                            T_model = Tmodel.loc[dateindex, depth - 1]

                            if not isnan(T_data):
                                T_data_samples.append(T_data)
                                T_model_samples.append(T_model)
                                T_depths.append(depth)
                                anydata_t = True

                            if not isnan(O2_data):
                                O2_data_samples.append(O2_data)
                                O2_model_samples.append(O2_model)
                                O2_depths.append(depth)
                                anydata_o = True
            except:
                crashed = True
                print("Error when handling lake %s %s" %(name, lake_id))
            label_o = "%s %s" %(name, lake_id)
            if crashed:
                print("%s - Error in program,unreliable" % label_o)
            else:
                if not anydata_o:
                    print('missing_data_%s' % name)
                else:
                    RMSE_O = round(sqrt(sklearn.metrics.mean_squared_error(O2_data_samples, O2_model_samples)),
                                     2)
                    corr_O = scipy.stats.pearsonr(O2_data_samples, O2_model_samples)
                    pearsono = '%.3f' %(round(corr_O[0], 3))
                    # i = [10e-08, 10e-07, 10e-06, 10e-05, 10e-04, 10e-03, 10e-02, 10e-01, 10e+00, 10e+01, 10e+02, 10e+03]
                    # j = [10e-07, 50e-07, 10e-06, 50e-06, 10e-05, 50e-05, 10e-04, 50e-04, 10e-03, 50e-03, 10e-02]
                    if path.exists(path.join('%s_%s.csv' %(csvr, lake_id))):
                        data = pd.read_csv('%s_%s.csv' %(csvr, lake_id), index_col=0)
                        data.iloc[n, m] = RMSE_O
                        data.to_csv('%s_%s.csv' %(csvr, lake_id))

                    else:
                        data = pd.DataFrame(index=j, columns=i)
                        data.iloc[n, m] = RMSE_O
                        data.to_csv('%s_%s.csv' %(csvr, lake_id))

                    if path.exists(path.join('%s_%s.csv' %(csvp, lake_id))):
                        data = pd.read_csv('%s_%s.csv' %(csvp, lake_id), index_col=0)
                        data.iloc[n, m] = pearsono
                        data.to_csv('%s_%s.csv' %(csvp, lake_id))
                    else:
                        data = pd.DataFrame(index=j, columns=i)
                        data.iloc[n, m] = pearsono
                        data.to_csv('%s_%s.csv' %(csvp, lake_id))

def FishNiche_csv_results_revise(scenarioid, modelid, lakelistfile, calivari, k_BOD):
    global fig1
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2B = y1B + 4
    timeaxis = pd.date_range("%s-01-01" % y1A, "%s-12-31" % y2B, freq="D")
    timestep = len(timeaxis)
    fig1, fig2, fig3, fig4 = plt.figure(), plt.figure(), plt.figure(), plt.figure()

    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)

    # lakeswehavedatafor = [698]
    lakeswehavedatafor = [14939, 67035, 16765, 698, 33494, 33590, 99045, 30704, 19167, 31895, 310, 32276, 99045, 99516,
                          6950, 33590, 33494]
    # lakeswehavedatafor = [698, 67035, 31895, 310, 32276]
    calibrationdata = pd.DataFrame(index=np.arange(0, len(lakeswehavedatafor)), columns=['lake_id', calivari])
    place = -1
    numberlakes = 0
    for lakenum in np.arange(1, nlakes):
        lake_id, subid, name, eh, area, depth, longitude, latitude, volume \
            = lakes[lakenum].strip().split(',')
        if int(lake_id)in lakeswehavedatafor:
            numberlakes += 1
            maxdepth = float(depth)
            # getOutputPathFromEbHex
            eh = eh[2:] if eh[:2] == '0x' else eh
            while len(eh)< 6:
                eh = '0' + eh
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]

            outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
            print(outdir)

            O2model = pd.read_csv(path.join(outdir, 'O2zt.csv'), header=None)
            Tmodel = pd.read_csv(path.join(outdir, 'Tzt.csv'), header=None)

            filename = '../sweden_inflow_data/Validation_data_for_lookup.xlsx'
            # need to be change if the file change
            worksheet = pd.read_excel(filename, sheet_name=lake_id)
            dates = worksheet['date']
            depthdata = worksheet['depth(max)']
            O2raw = worksheet['O2(mg/l)'] * 1000
            Traw = worksheet['Water temp(°C)']
            anydata_t, anydata_o = False, False
            T_model_samples, T_data_samples, T_depths = [], [], []
            O2_model_samples, O2_data_samples, O2_depths = [], [], []
            crashed = False

            try:
                for ii in np.arange(0, len(dates)):
                    dnum = datetime.strptime(dates[ii], "%Y-%m-%d").date()
                    start = timeaxis[0].date()
                    dateindex =(dnum - start).days
                    if dateindex >= 1 and dateindex <= timestep:
                        if lake_id == '698':
                            if depthdata[ii] <= 2:
                                depth = ceil(depthdata[ii])
                                T_data = Traw[ii]
                                T_model = Tmodel.loc[dateindex, depth - 1]

                                if not isnan(T_data):
                                    T_data_samples.append(T_data)
                                    T_model_samples.append(T_model)
                                    T_depths.append(depth)
                                    anydata_t = True

                            depthO = ceil(depthdata[ii])
                            O2_data = O2raw[ii]
                            O2_model = O2model.loc[dateindex, depthO - 1]
                            if not isnan(O2_data):
                                O2_data_samples.append(O2_data)
                                O2_model_samples.append(O2_model)
                                O2_depths.append(depthO)
                                anydata_o = True

                        elif lake_id == '67035':
                            if depthdata[ii] <= 1 or depthdata[ii] >= 60:
                                depth = ceil(depthdata[ii])
                                T_data = Traw[ii]
                                T_model = Tmodel.loc[dateindex, depth - 1]

                                if not isnan(T_data):
                                    T_data_samples.append(T_data)
                                    T_model_samples.append(T_model)
                                    T_depths.append(depth)
                                    anydata_t = True

                            depthO = ceil(depthdata[ii])
                            O2_data = O2raw[ii]
                            O2_model = O2model.loc[dateindex, depthO - 1]
                            if not isnan(O2_data):
                                O2_data_samples.append(O2_data)
                                O2_model_samples.append(O2_model)
                                O2_depths.append(depthO)
                                anydata_o = True

                        else:
                            depth = ceil(depthdata[ii])
                            O2_data = O2raw[ii]
                            T_data = Traw[ii]
                            O2_model = O2model.loc[dateindex, depth - 1]
                            T_model = Tmodel.loc[dateindex, depth - 1]

                            if not isnan(T_data):
                                T_data_samples.append(T_data)
                                T_model_samples.append(T_model)
                                T_depths.append(depth)
                                anydata_t = True

                            if not isnan(O2_data):
                                O2_data_samples.append(O2_data)
                                O2_model_samples.append(O2_model)
                                O2_depths.append(depth)
                                anydata_o = True
            except:
                crashed = True
                print("Error when handling lake %s %s \n" %(name, lake_id))
            label_t = "%s %s" %(name, lake_id)
            label_o = "%s %s" %(name, lake_id)
            if crashed:
                label_t =("%s - Error in program,unreliable" % label_t)
                label_o =("%s - Error in program,unreliable" % label_o)
            else:
                if not anydata_t:
                    label_t = "%s(No data in timespan)" % label_t
                else:
                    RMSE_T = round(sqrt(sklearn.metrics.mean_squared_error(T_data_samples, T_model_samples)),
                                     2)
                    corr_T = scipy.stats.pearsonr(T_data_samples, T_model_samples)
                    pearsont = '%.3f' %(round(corr_T[0], 3))
                    label_t = "%s\n(RMSE: %s Pearson: %s)" %(label_t, RMSE_T, pearsont)
                if not anydata_o:
                    label_o = "%s(No data in timespan)" % label_o
                else:
                    RMSE_O = round(sqrt(sklearn.metrics.mean_squared_error(O2_data_samples, O2_model_samples)),
                                     2)
                    corr_O = scipy.stats.pearsonr(O2_data_samples, O2_model_samples)
                    pearsono = '%.3f' %(round(corr_O[0], 3))

                    label_o = "%s\n(RMSE: %s Pearson: %s)" %(label_o, RMSE_O, pearsono)

            lineStart = 0
            lineEnd = 25
            rows = ceil(sqrt(nlakes - 1))
            cols = ceil((nlakes - 1)/ rows)

            plt.rc('axes', linewidth=2)
            fontsize = 30
            lw = 0.5
            markers = ['o', 'v', '+', '*', '8', '^', 'D', 's', 'd', 'p', 'x']
            markers = ['d', 'v', '+', '*', '8', 's', 'D', '^', 'o']
            s = [100, 100 * 1.5, 100, 100 * 2, 100, 100 * 1.5, 100, 100, 100 * 1.5, 100, 100]
            i = 100 * 1.5
            s = [i * 1.5, i * 2, i, i * 2.5, i, i * 1.5, i, i * 2, i]
            if lakenum < len(markers):
                mark = markers[lakenum - 1]
                size = s[lakenum - 1]
            else:
                mark = markers[0]
                size = s[0]
            alpha = 0.8
            params = {
                'axes.labelsize': 30,
                'text.fontsize': 30,
                'legend.fontsize': 30,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'text.usetex': False,
                'figure.figsize': [14, 10]  # instead of 4.5, 4.5
            }

            plt.rcParams.update(params)
            colorset = ['k', 'cyan', 'maroon', 'C1', 'C6', 'C2', 'C3', 'maroon', 'gold',
                        'C8']  # ['C1','C3','y','k','w','C7','C0','C2','C9']
            place += 1
            if anydata_t:
                fig1 = plt.figure(k_BOD * 100) # ,figsize=(3.0, 3.0))
                ax = plt.gca()
                color = [1 -(i / maxdepth)for i in T_depths]
                points = plt.scatter(T_data_samples, T_model_samples, label=label_t, s=size, c=color, edgecolors='k',
                                       linewidths=lw, cmap='Blues_r', marker=mark, alpha=alpha)
                # for i,txt in enumerate(T_depths):
                # plt.annotate(txt,(T_data_samples[i],T_model_samples[i]))
                plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', linewidth=1.0)
                plt.tick_params(axis='both', which='major') # , labelsize=14)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(fontsize)
                    # tick.label1.set_fontweight('bold')
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(fontsize)
                    # tick.label1.set_fontweight('bold')
                plt.xlabel('Observed Temperature($\degree$C)') # ,fontweight='bold')
                plt.ylabel('Modeled Temperature($\degree$C)') # ,fontweight='bold')
                # plt.legend(loc=4)
                if numberlakes == len(lakeswehavedatafor):
                    cb = plt.colorbar(points, ticks=np.linspace(1.1, -0.1, 12, endpoint=True))
                    # cb.set_label(weight='bold')
                    cb.ax.set_yticklabels(['surface', '', '', '', '', '', '', '', 'max depth']) # ,weight='bold')
                plt.tight_layout()

                fig2 = plt.figure((k_BOD + 1)* 100) # ,figsize=(18.0, 10.0))
                ax = plt.gca()

                plt.subplot(rows, cols, lakenum)
                points = plt.scatter(T_data_samples, T_model_samples, c=T_depths, edgecolors='k', label=label_t, s=20,
                                       cmap='bone_r') # T_depths
                plt.colorbar(points)
                plt.clim([0, maxdepth])
                plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
                plt.title(label_t, fontsize=14, y=1.02)
                plt.subplots_adjust(hspace=0.45, wspace=0.2)
                plt.tick_params(axis='both', which='major') # , labelsize=14)
                plt.yticks(np.arange(0, 35, 5))
                fig2.text(0.5, 0.02, 'Modeled Temperature($\degree$C)', ha='center')
                fig2.text(0.07, 0.5, 'Observed Temperature($\degree$C)', va='center', rotation='vertical')
                fig2.text(0.92, 0.5, 'depth(m)', va='center', rotation='vertical')

            lineStart = 0
            lineEnd = 20000
            if anydata_o:
                fig3 = plt.figure((k_BOD + 3)* 100) # ,figsize=(18.0, 10.0))
                ax = plt.gca()
                color = [1 -(i / maxdepth)for i in O2_depths]
                points = plt.scatter([i / 1000 for i in O2_data_samples], [i / 1000 for i in O2_model_samples],
                                       label=label_o, alpha=alpha, s=size, linewidths=lw, c=color, edgecolors='k',
                                       cmap='Reds_r', marker=mark)

                # for i,txt in enumerate(O2_depths):
                # plt.annotate(txt,(O2_data_samples[i],O2_model_samples[i]))
                plt.plot([lineStart, lineEnd / 10 ** 3], [lineStart, lineEnd / 10 ** 3], 'k-', linewidth=1.0)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(fontsize)
                # tick.label1.set_fontweight('bold')
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(fontsize)
                    # tick.label1.set_fontweight('bold')
                plt.tick_params(axis='both', which='major') # , labelsize=14)

                plt.xlabel(r'Observed Oxygen(mg/L)') # ,fontweight='bold' )
                plt.ylabel(r'Modeled Oxygen(mg/L)') # ,fontweight='bold' )
                # plt.legend(loc=4)
                # plt.ylim(0,20000,2)
                plt.yticks([0, 5, 10, 15, 20])
                plt.xticks([0, 5, 10, 15, 20])
                if numberlakes == len(lakeswehavedatafor):
                    cb = plt.colorbar(points, ticks=np.linspace(1.1, -0.1, 12, endpoint=True))
                    # cb.set_label(weight='bold')
                    cb.ax.set_yticklabels(['surface', '', '', '', '', '', '', '', 'max depth']) # ,weight='bold')

                fig4 = plt.figure((k_BOD + 4)* 100) # ,figsize=(18.0, 10.0))
                ax = plt.gca()
                plt.subplot(rows, cols, lakenum)
                points = plt.scatter([i / 10 ** 3 for i in O2_data_samples], [j / 10 ** 3 for j in O2_model_samples],
                                       c=O2_depths, edgecolors='k', label=label_o, s=50, cmap='bone_r')
                cb = plt.colorbar(points)
                plt.clim([0, maxdepth])
                cb.ax.tick_params(labelsize=14)
                plt.plot([lineStart, lineEnd / 10 ** 3], [lineStart, lineEnd / 10 ** 3], 'k-')
                plt.title(label_o, fontsize=14, y=1.02)
                plt.subplots_adjust(hspace=0.45, wspace=0.2)
                plt.tick_params(axis='both', which='major') # , labelsize=14)
                plt.yticks(np.arange(0, 24, 4))
                fig4.text(0.5, 0.02, 'Modeled Oxygen(mg/L)', ha='center')
                fig4.text(0.07, 0.5, 'Observed Oxygen(mg/L)', va='center', rotation='vertical')
                fig4.text(0.92, 0.5, 'depth(m)', va='center', rotation='vertical')



                # calibration = abs(sum(T_model_samples)-sum(T_data_samples))**2
    # calibrationdata.to_csv(os.path.join(outputfolderdata, 'calibration,csv'))
    # df_csv = pd.read_csv(os.path.join(outputfolderdata, 'calibration.csv'))
    # result = pd.merge(df_csv, calibrationdata, on='lake_id')
    # result.to_csv(os.path.join(outputfolderdata, 'calibration.csv'),index=False)

    fig1.savefig(path.join(datafolder, "Figure_1_temp_test%s.png" % calivari))
    # plt.gcf().clear(fig1)
    fig2.savefig(path.join(datafolder, "Figure_2_temp alltest_%s.png" % calivari))
    # plt.gcf().clear(fig2)
    fig3.savefig(path.join(datafolder, "Figure_1_Oxygen_test%s.png" % calivari))
    # plt.gcf().clear(fig3)
    fig4.savefig(path.join(datafolder, "Figure_2_Oxygen alltest_%s.png" % calivari))

    # return calibrationdata

def FishNiche_export_timeseries(dataframe, scenarioid, filename):
    exA, y1A, exB, y1B = scenarios[scenarioid]
    y2B = y1B + 4
    date_start = date(y1A, 1, 1)
    date_end = date(y2B, 12, 31)
    timeseries = [
        ['lakeid', 'Name', 'Date', 'Average O2 above max T gradient', 'Average O2 below max T gradient',
         'Average T above max T gradient', 'Average T below max T gradient',
         'Volume with T < 15 C', 'Volume with O2 > 3000', 'Volume with Attn > 1% of surface Attn',
         'Volume satisfying all three previous', 'Depth of maximal T gradient']]

    nlakes, columns = dataframe.shape
    lakeid = [x for x in dataframe["lakeid"]]
    Names = [x for x in dataframe["Name"]]

    for lakenum in np.arange(1, nlakes + 1):
        Dates = pd.date_range(date_start, date_end, freq='D')
        rows_before_lake =(len(Dates))*(lakenum - 1)
        average_o2_above = dataframe.loc[lakenum, "Average O2 above max T gradient"]
        average_o2_below = dataframe.loc[lakenum, "Average O2 below max T gradient"]
        average_t_above = dataframe.loc[lakenum, "Average T above max T gradient"]
        average_t_below = dataframe.loc[lakenum, "Average T below max T gradient"]
        volume_t_below = dataframe.loc[lakenum, "Volume with T < 15 C"]
        volume_o2_above = dataframe.loc[lakenum, "Volume with O2 > 3000"]
        volume_attn_above = dataframe.loc[lakenum, "Volume with Attn > 1% of surface Attn"]
        volume_3_param = dataframe.loc[lakenum, "Volume satisfying all three previous"]
        depth_max = dataframe.loc[lakenum, "Depth of maximal T gradient"]
        for time in np.arange(1, len(Dates)+ 1):
            day, month, year = Dates[time - 1].day, Dates[time - 1].month, Dates[time - 1].year
            if day < 10:
                day = str("0%s" % day)
            if month < 10:
                month = str("0%s" % month)
            timeseries.append([])
            timeseries[time + rows_before_lake].append(lakeid[lakenum - 1])
            timeseries[time + rows_before_lake].append(Names[lakenum - 1])
            timeseries[time + rows_before_lake].append("%s-%s-%s" %(year, month, day))
            timeseries[time + rows_before_lake].append(average_o2_above[time - 1])
            timeseries[time + rows_before_lake].append(average_o2_below[time - 1])
            timeseries[time + rows_before_lake].append(average_t_above[time - 1])
            timeseries[time + rows_before_lake].append(average_t_below[time - 1])
            timeseries[time + rows_before_lake].append(volume_t_below[time - 1])
            timeseries[time + rows_before_lake].append(volume_o2_above[time - 1])
            timeseries[time + rows_before_lake].append(volume_attn_above[time - 1])
            timeseries[time + rows_before_lake].append(volume_3_param[time - 1])
            timeseries[time + rows_before_lake].append(depth_max[time - 1])

    if path.isfile(filename):
        with open(filename, "a")as f:
            data = pd.DataFrame(timeseries[1:])
            data.to_csv(f, index=False, header=False)
    else:
        with open(filename, "w")as f:
            data = pd.DataFrame(timeseries[1:], columns=timeseries[0])
            data.to_csv(f, index=False, header=True)

def FishNiche_generate_timeseries(filename, scenarioid, modelid):
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2B = y1B + 4

    with open(filename, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)

    timeseries_records = pd.DataFrame(index=np.arange(1, nlakes), columns=(
        'lakeid', "Name", 'Average O2 above max T gradient', 'Average O2 below max T gradient',
        'Average T above max T gradient', 'Average T below max T gradient',
        'Volume with T < 15 C', 'Volume with O2 > 3000', 'Volume with Attn > 1% of surface Attn',
        'Volume satisfying all three previous', 'Depth of maximal T gradient'))
    for lakenum in np.arange(1, nlakes):
        lake_id, subid, name, eh, area, depth, longitude, latitude \
            = lakes[lakenum].strip().split(',')
        surface_area = float(area)
        max_depth = float(depth)

        # getOutputPathFromEbHex
        eh = eh[2:] if eh[:2] == '0x' else eh
        while len(eh)< 6:
            eh = '0' + eh
        d1, d2, d3 = eh[:2], eh[:4], eh[:6]
        outdir = path.join(outputfolder, d1, d2, d3,
                             'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
        print(outdir)

        # read *.cvs
        o2 = pd.read_csv(path.join(outdir, 'O2zt.csv'), header=None)
        T = pd.read_csv(path.join(outdir, 'Tzt.csv'), header=None)
        Attn = pd.read_csv(path.join(outdir, 'Attn_zt.csv'), header=None)

        # set variable
        Tt = T.transpose()
        T_grad = np.diff(Tt, axis=0).transpose()
        maxgrad_depth =(np.absolute(T_grad.transpose())).argmax(0)
        maxgrad_depth = maxgrad_depth.transpose()

        tlen, zlen = T.shape
        average_o2_above, average_T_below, average_o2_below, average_T_above = [0] * tlen, [0] * tlen, [0] * tlen, [
            0] * tlen
        volume_o2_above, volume_T_below, volume_Attn_above, volume_all_parameters = [0] * tlen, [0] * tlen, [
            0] * tlen, [0] * tlen

        timeseries_records.loc[lakenum, "lakeid"] = lake_id
        timeseries_records.loc[lakenum, "Name"] = name

        for time in np.arange(0, tlen):
            average_o2_above[time] = st.mean(o2.loc[time, 0:maxgrad_depth[time]])
            average_o2_below[time] = st.mean(o2.loc[time, maxgrad_depth[time] + 1:])
            average_T_above[time] = st.mean(T.loc[time, 0:maxgrad_depth[time]])
            average_T_below[time] = st.mean(T.loc[time, maxgrad_depth[time] + 1:])

            surface_attn = Attn.loc[time, 0]  # MC 05-28-2018 change 1 for 0
            for depth in np.arange(0, zlen):
                if o2.loc[time, depth] > 3000:
                    volume_o2_above[time] = volume_o2_above[time] + area_at_depth(surface_area, depth, max_depth)
                if T.loc[time, depth] < 15:
                    volume_T_below[time] = volume_T_below[time] + area_at_depth(surface_area, depth, max_depth)
                if Attn.loc[time, depth] >= 0.01 * surface_attn:
                    volume_Attn_above[time] = volume_Attn_above[time] + area_at_depth(surface_area, depth, max_depth)
                if o2.loc[time, depth] > 3000 and T.loc[time, depth] < 15 and Attn.loc[
                    time, depth] >= 0.01 * surface_attn:
                    volume_all_parameters[time] = volume_all_parameters[time] + area_at_depth(surface_area, depth,
                                                                                                max_depth)

        # add to list

        timeseries_records.loc[lakenum, 'Average O2 above max T gradient'] = average_o2_above
        timeseries_records.loc[lakenum, 'Average O2 below max T gradient'] = average_o2_below
        timeseries_records.loc[lakenum, 'Average T above max T gradient'] = average_T_above
        timeseries_records.loc[lakenum, 'Average T below max T gradient'] = average_T_below
        timeseries_records.loc[lakenum, 'Volume with O2 > 3000'] = volume_o2_above
        timeseries_records.loc[lakenum, 'Volume with T < 15 C'] = volume_T_below
        timeseries_records.loc[lakenum, 'Volume with Attn > 1% of surface Attn'] = volume_Attn_above
        timeseries_records.loc[lakenum, 'Volume satisfying all three previous'] = volume_all_parameters
        timeseries_records.loc[lakenum, 'Depth of maximal T gradient'] = maxgrad_depth + 1
    FishNiche_export_timeseries(timeseries_records, scenarioid, "Fish_Niche_Export.csv")
    return timeseries_records

def FishNiche_plot_timeseries(lakelistfile, scenarioid, modelid, calivari, outputfile):
    y1A, y1B = scenarios[scenarioid][1], scenarios[scenarioid][3]
    y2B = y1B + 4
    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)
    timeaxis = pd.date_range("%s-01-01" % y1A, "%s-12-31" % y2B, freq="D")
    timeseries = pd.read_csv('fish_niche_export2.csv', encoding="ISO-8859-1")

    for lakenum in np.arange(1, nlakes):
        lake_id, subid, name, eh, area, depth, longitude, latitude \
            = lakes[lakenum].strip().split(',')
        rows = ceil(sqrt(nlakes - 1))
        cols = ceil((nlakes - 1)/ rows)
        data = timeseries.loc[timeseries["lakeid"] == int(lake_id), "Volume with T < 15 C"]
        data2 = timeseries.loc[timeseries["lakeid"] == int(lake_id), "Volume with O2 > 3000"]
        time = timeseries.loc[timeseries["lakeid"] == int(lake_id), "Date"]
        time = pd.to_datetime(time,
                                format='%d.%m.%Y') # MC 2018-08-15,format='%Y-%m-%d')with . = format produce by matlab, - = python
        ts1 = pd.concat([time, data, data2], axis=1)
        ts1 = ts1.set_index('Date')

        fig3 = plt.figure(3 + int(lake_id))
        ax1 = ts1["Volume with T < 15 C"].plot()

        fig2 = plt.figure(2 + int(lake_id))
        i, ii, j, jj = data.max(), data.min(), 0, 0
        print(i)
        while i > 10:
            j += 1
            i = i / 10
        if ceil(i)== 10:
            j += 1
            i = i / 10
        if data.max()/ 10 ** j > 5:
            max = 10
        else:
            max = 5
            if data.max()/ 10 ** j < 2:
                max = 2

        ax = plt.subplot()
        ax.plot_date(time, ts1["Volume with T < 15 C"] / 10 ** j, 'C0o-', lw=2, ms=3)
        ax.plot_date(time, ts1["Volume with O2 > 3000"] / 10 ** j, 'C3o-', lw=2, ms=3)

        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        ax.autoscale_view()

        ax.fmt_xdata = DateFormatter('%Y-%m-%d')
        ax.fmt_ydata = price
        ax.grid(True)

        fig2.autofmt_xdate()
        plt.tight_layout()
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        # plt.legend(loc="lower left",ncol=2)

        plt.ylabel("Volume(1e%s m**3)" % j)
        plt.xlabel("Date")
        plt.tight_layout()
        fig2 = plt.savefig(r"VOLUME_%s_%s_%s_graph.png" %(calivari, lake_id, name))
        # plt.subplots_adjust(hspace=0.45, wspace=0.2)

        fig1 = plt.figure(1)
        ax = plt.subplot(int('%s%s%s' %(rows, cols, lakenum)))
        ax.plot_date(time, ts1["Volume with T < 15 C"], 'C0o-', lw=2, ms=3)
        ax.plot_date(time, ts1["Volume with O2 > 3000"], 'C3o-', lw=2, ms=3)
        # plt.ylim(0,max,1)
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        ax.autoscale_view()
        plt.title('%s %s' %(name, lake_id), size=12)
        ax.fmt_xdata = DateFormatter('%Y-%m-%d')
        ax.fmt_ydata = price
        ax.grid(True)

        fig1.autofmt_xdate()

        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)

        q =(nlakes)-(ceil(rows / 2))
        p = rows *(ceil(cols / 2)- 1)+ 1
        if lakenum == p:
            plt.ylabel("Volume(m**3)", size=12)
        if lakenum == q:
            plt.xlabel('Date', size=12)
    # fig1.text(0.5, 0.01, 'Date', ha='center', fontproperties='bold')
    # fig1.text(0.01, 0.5, "Volume(1e%s m**3)" %j, va='center', rotation='vertical', fontproperties='bold')
    plt.tight_layout()

    fig1 = plt.savefig(r"VOLUME_%s_graph.png" %(calivari))

    plt.gcf().clear()
    return plt.figure(1)

def FishNiche_cathegories(lakelistfile):
    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)
    lake_list = [["Lake_Name", "Lake_ID", "Depth", "Surface", "Volume"]]

    for lakenum in np.arange(1, nlakes):
        lake_id, subid, name, eh, area, depth, longitude, latitude, volume \
            = lakes[lakenum].strip().split(',')
        surface_area = float(area)
        max_depth = float(depth)
        volume = 0
        for i in np.arange(0, ceil(max_depth)):
            volume = volume + area_at_depth(surface_area, i, max_depth)
        lake_row = [name, lake_id, max_depth, surface_area, volume]
        lake_list.append(lake_row)
    final = pd.DataFrame(lake_list[1:], columns=lake_list[0])
    # final.to_csv('volume_by_lake.csv',index=False,header=True)
    params = {
        'axes.labelsize': 13,
        'text.fontsize': 13,
        'legend.fontsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'text.usetex': False,
        'figure.figsize': [4.5, 4]  # instead of 4.5, 4.5
    }
    fontsize = 14
    plt.rcParams.update(params)
    fig1 = plt.figure(100)

    p = plt.hist(final["Volume"], bins=np.logspace(np.log10(247419.07882405), np.log10(1.75631E+11), 29),
                   color="# 3F5D7D", edgecolor='k')

    plt.tick_params(axis='both', which='major') # , labelsize=14)
    ax = plt.gca()
    ax.set_xscale("log")

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    plt.xlabel(r'Volume($\bf{m^3}$)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.yticks(np.arange(0, 26, 5))
    plt.tight_layout()
    final['Lake_group'] = 2
    final.loc[final['Volume'] < 1.1e8, 'Lake_group'] = 1
    final.loc[final['Volume'] > 1.1e10, 'Lake_group'] = 3
    # final.to_csv('volume_by_lake_1.csv', index=False, header=True)
    # fig1.savefig("histogram.eps")
    # fig1.savefig("histogram.png")
    plt.show()

def FishNiche_plot_volume01(lakelistfile, listscenarios, listmodels, calivari, datafolder):
    years = YearLocator() # every year
    months = MonthLocator() # every month
    yearsFmt = DateFormatter('%Y')
    monthsFmt = DateFormatter('%M')
    # fig2,fig3=plt.figure(),plt.figure()
    modelname = ['KNM', 'DMI', 'MPI', 'MOH', 'IPS', 'CNR']
    color2 = ['white', 'blue', 'black', 'magenta', 'cyan', 'red', 'yellow']
    i = 0
    for group in [1]:
        datasheet_all = pd.read_csv(path.join(datafolder, 'complete_data_%s.csv' % group))
        datasheet_all['Date'] = pd.to_datetime(datasheet_all['Date'], format="%Y-%m-%d")
        datasheet_all.set_index('Date', inplace=True)
        datasheet2 = datasheet_all
        fig1 = plt.figure()
        ax1 = plt.subplot(211)
        for scenario in ['historical',  'rcp85']:
            
            i += 1
            tt = pd.date_range(start='2000-01-01', end='2000-12-31')

            datasheet_scenario = datasheet_all.loc[datasheet_all['Scenario'] == scenario]
            if scenario == 'historical':
                datasheet_scenario = datasheet_scenario.loc[:'1982-12-31']
            else:
                datasheet_scenario = datasheet_scenario.loc['2090-12-31':]
                # datasheet_scenario = datasheet_scenario.loc['2060-12-31':'2080-12-31']
                # datasheet_scenario = datasheet_scenario.loc[:'2050-12-31']
            # ttyear = datasheet2.index
            # ttyear = ttyear.drop_duplicates(keep='first')
            listonplot = []
            for model in [1, 2, 3, 4, 5, 6]:
                datasheet = datasheet_scenario.loc[datasheet_scenario['Model'] == model]
                datasheet['%0_T'] = datasheet['%_T'] * 100
                datasheet['%0_O2'] = datasheet['%_O2'] * 100
                datasheet['%0_PAR'] = datasheet['%_PAR'] * 100

                if len(datasheet)!= 0:
                    listonplot.append(model)

                    medianbyday = datasheet.groupby([datasheet.index.month, datasheet.index.day]).quantile(0.5)
                    minbyday = datasheet.groupby([datasheet.index.month, datasheet.index.day]).quantile(0.25)
                    maxbyday = datasheet.groupby([datasheet.index.month, datasheet.index.day]).quantile(0.75)

                    meanbyday = datasheet.groupby([datasheet.index.month, datasheet.index.day]).mean()
                    stdbyday = datasheet.groupby([datasheet.index.month, datasheet.index.day]).std()

                    # z_critical = stats.norm.ppf(q=0.975) # Get the z-critical value*
                    meanbyday['%0_O22'] = 100 - meanbyday['%0_O2']
                    meanbyday['%0_PAR2'] = 100 - meanbyday['%0_PAR']
                    stdbyday['%0_O22'] = stdbyday['%0_O2']
                    stdbyday['%0_PAR2'] = stdbyday['%0_PAR']
                    # minbyday['%0_O22'] =100- minbyday['%0_O2']
                    # maxbyday['%0_O22'] = 100-maxbyday['%0_O2']
                    # minbyday['%0_PAR2'] = 100 - minbyday['%0_PAR']
                    # maxbyday['%0_PAR2'] = 100 - maxbyday['%0_PAR']

                    stats.norm.ppf(q=0.025)
                    # margin_of_error = z_critical *(stdbyday/sqrt(countbyday.iloc[0,0]))
                    meanplusmargin = meanbyday + stdbyday
                    meanplusmargin.loc[meanplusmargin['%0_T'] > 100, '%0_T'] = 100
                    meanplusmargin.loc[meanplusmargin['%0_PAR2'] > 100, '%0_PAR2'] = 100
                    meanplusmargin.loc[meanplusmargin['%0_O22'] > 100, '%0_O22'] = 100
                    meanlessmargin = meanbyday - stdbyday
                    meanlessmargin.loc[meanlessmargin['%0_T'] < 0, '%0_T'] = 0
                    meanlessmargin.loc[meanlessmargin['%0_PAR2'] < 0, '%0_PAR2'] = 0
                    meanlessmargin.loc[meanlessmargin['%0_O22'] < 0, '%0_O22'] = 0

                    #fig1 = plt.figure(2 * group * i, figsize=(18.0, 10.0))
                    
                    #ax1.fill_between(tt, meanbyday['%0_T'], meanlessmargin['%0_T'], color='lightblue', alpha='0.5')
                    #ax1.fill_between(tt, meanbyday['%0_T'], meanplusmargin['%0_T'], color='lightblue', alpha='0.5')
                    #ax1.plot_date(tt, meanbyday['%0_T'], '-', color=plt.cm.binary((model / 10)+ 0.3), lw=2, ms=3)
                    # ax1.annotate('Temperature barrier(T>15C)', xy=(80, 80), xycoords='figure points')
                    # ax1.plot_date(tt, meanbyday['%_T']-margin_of_error['%_T'],'k--',lw=2, ms=3)
                    #ax1.fill_between(tt, meanbyday['%0_PAR2'], meanplusmargin['%0_PAR2'], color='green', alpha='0.2')
                    #ax1.fill_between(tt, meanbyday['%0_PAR2'], meanlessmargin['%0_PAR2'], color='green', alpha='0.2')
                    #ax1.plot_date(tt, meanbyday['%0_PAR2'], '-', color=plt.cm.binary((model / 10)+ 0.3), lw=2,
                    #                ms=3)
                    # ax1.annotate('PAR barrier(PAR)', xy=(80, 80), xycoords='figure points')
                    # ax1.plot_date(tt, meanbyday['%_PAR'] - margin_of_error['%_PAR'], 'g--', lw=2, ms=3)
                    #ax1.fill_between(tt, meanbyday['%0_O22'], meanlessmargin['%0_O22'], color='red', alpha='0.1')
                    if scenario =='historical':
                        #if model == 5:
                        ax1.fill_between(tt, 0, meanplusmargin['%0_O22'], color='black', alpha='0.1')
                            #ax1.fill_between(tt, meanbyday['%0_O2'], meanplusmargin['%0_O2'], color='red', alpha='0.3')
                        ax1.plot_date(tt, meanplusmargin['%0_O22'], '-', color=plt.cm.binary((model / 10)+ 0.3), lw=2,ms=3)

                    else:
                        #if model ==1:
                        ax1.fill_between(tt, 0, meanplusmargin['%0_O22'], color='red', alpha='0.1')
                            #ax1.fill_between(tt, meanbyday['%0_O2'], meanplusmargin['%0_O2'], color='red', alpha='0.3')
                        ax1.plot_date(tt, meanplusmargin['%0_O22'], '-', color=plt.cm.Reds((model / 10)+ 0.3), lw=2,ms=3)

                    #ax1.plot_date(tt, meanplusmargin['%0_O22'], '-', color=plt.cm.binary((model / 10)+ 0.3), lw=2,ms=3)

                    ax1.autoscale_view()

                    plt.ylim(0, 100, 50)
                    plt.yticks([0, 25, 50, 75, +100])
                    ax1.set_xlim([datetime(2000, 1, 1), datetime(2000, 12, 31)])
                    plt.ylabel("% Volume O2")

                    ax4 = ax1
                    ax4.set_ylim(0, 100, 50)
                    #plt.yticks([0, 25, 50, 75, +100])
                    #ax4.plot_date(tt, meanbyday['%0_O2'], '-', color=plt.cm.binary((model / 10)+ 0.3), lw=2,
                    #                ms=3)
                    # ax4.plot_date(tt, meanlessmargin['%_O2'], ':', color=color[model], lw=2, ms=3)
                    # ax4.annotate('Oxygen barrier(O>15C)', xy=(80, 80), xycoords='figure points')
                    # ax4.plot_date(tt, meanbyday['%_O2'] - margin_of_error['%_O2'], 'r--', lw=2, ms=3)
                    # ax4.fill_between(tt, meanbyday['%_O2'], meanlessmargin['%_O2'], color=color2[model],alpha='0.2')
                    # ax4.fill_between(tt, meanbyday['%_O2'], meanplusmargin['%_O2'], color=color2[model], alpha='0.2')
                    #plt.gca().invert_yaxis()
                    ax4.xaxis.set_major_locator(months)
                    ax4.xaxis.set_minor_locator(mondays)
                    ax4.xaxis.set_major_formatter(weekFormatter)
                    ax1.fmt_ydata = price
                    ax1.yaxis.grid(True)

                    # winter_months = tt[
                    #    ((tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                    #     |((tt.month == 3)&(tt.day == 20))|(
                    #      (tt.month == 12)&(tt.day == 31))]
                    # summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                    # Loop through groups and add axvspan between winter month
                    # for i in np.arange(0, len(winter_months), 2):
                    #    ax1.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                    # for i in np.arange(0, len(summer_months), 2):
                    #    ax1.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                    # fig1.autofmt_xdate()
                    ax1.tick_params(axis='y', labelsize=16)
                    ax1.tick_params(axis='x', labelsize=16)
                    # plt.legend(loc="lower left", ncol=2)

                    #plt.ylabel("% Volume T")
                    plt.xlabel("Date")

                    # ax1.set_xlim(pd.Timestamp("2-12-01"), pd.Timestamp("2011-02-01"))
                    # plt.legend(loc='best')
                    #ax4.xaxis_date()

                    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

                    # plt.title('group %s scenario %s'%(group,scenario))# scenario %s' %(group,scenario))
                    plt.tight_layout()

            interet = []
            ert = []
#            for i in [1, 2, 3, 4, 5, 6]:
#                interet.append(mlines.Line2D([], [], color=plt.cm.binary((i / 10)+ 0.3), markersize=3,
#                                                 label=modelname[i - 1]))
#
#            ert.append(Patch(color='lightblue', alpha=0.5, label='T'))
#            ert.append(Patch(color='red', alpha=0.1, label='DO'))
#
#            first_legend = plt.legend(loc='upper center', bbox_to_anchor=(0.4, -0.2), fancybox=True, shadow=True,
#                                        ncol=1, handles=ert)
#            plt.gca().add_artist(first_legend)
#
#            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1,
#                         handles=interet)
            plt.show()
            # fig1.savefig(path.join(datafolder, "Figure_synthese_A2_group_%s_scenario_%s_mean.png" %(group, scenario)))
            fig1.savefig(
                    path.join(datafolder, "Figure_synthese_A2_group_%s_scenario_%s_mean.png" %(group, scenario)))
            print('completed')

def FishNiche_plot_volume(lakelistfile, listscenarios, listmodels, calivari, datafolder):
    years = YearLocator() # every year
    months = MonthLocator() # every month
    yearsFmt = DateFormatter('%Y')
    monthsFmt = DateFormatter('%M')
    # fig2,fig3=plt.figure(),plt.figure()
    modelname = ['KNM', 'DMI', 'MPI', 'MOH', 'IPS', 'CNR']
    color2 = ['white', 'blue', 'black', 'magenta', 'cyan', 'red', 'yellow']
    i = 0
    for group in [1,2, 3]:
        datasheet_all = pd.read_csv(path.join(datafolder, 'complete_data_%s1.csv' % group))
        datasheet_all['Date'] = pd.to_datetime(datasheet_all['Date'], format="%Y-%m-%d")
        datasheet_all.set_index('Date', inplace=True)
        datasheet2 = datasheet_all
        for scenario in ['historical', 'rcp45', 'rcp85']:
            fig1 = plt.figure()
            i += 1
            tt = pd.date_range(start='2000-01-01', end='2000-12-31')

            datasheet_scenario = datasheet_all.loc[datasheet_all['Scenario'] == scenario]
            if scenario == 'historical':
                datasheet_scenario = datasheet_scenario.loc[:'1982-12-31']
            else:
                datasheet_scenario = datasheet_scenario.loc['2090-12-31':]
                # datasheet_scenario = datasheet_scenario.loc['2060-12-31':'2080-12-31']
                # datasheet_scenario = datasheet_scenario.loc[:'2050-12-31']
            # ttyear = datasheet2.index
            # ttyear = ttyear.drop_duplicates(keep='first')
            listonplot = []
            for model in [1, 2, 3, 4, 5, 6]:
                datasheet = datasheet_scenario.loc[datasheet_scenario['Model'] == model]
                datasheet['%0_T'] = datasheet['%_T'] * 100
                datasheet['%0_O2'] = datasheet['%_O2'] * 100
                datasheet['%0_PAR'] = datasheet['%_PAR'] * 100
                datasheet['%0_habitable'] = datasheet['%_habitable']*100

                if len(datasheet)!= 0:
                    listonplot.append(model)

                    medianbyday = datasheet.groupby([datasheet.index.month, datasheet.index.day]).quantile(0.5)
                    minbyday = datasheet.groupby([datasheet.index.month, datasheet.index.day]).quantile(0.25)
                    maxbyday = datasheet.groupby([datasheet.index.month, datasheet.index.day]).quantile(0.75)

                    meanbyday = datasheet.groupby([datasheet.index.month, datasheet.index.day]).mean()
                    stdbyday = datasheet.groupby([datasheet.index.month, datasheet.index.day]).std()

                    # z_critical = stats.norm.ppf(q=0.975) # Get the z-critical value*
                    meanbyday['%0_O22'] = 100 - meanbyday['%0_O2']
                    meanbyday['%0_PAR2'] = 100 - meanbyday['%0_PAR']
                    stdbyday['%0_O22'] = stdbyday['%0_O2']
                    stdbyday['%0_PAR2'] = stdbyday['%0_PAR']
                    # minbyday['%0_O22'] =100- minbyday['%0_O2']
                    # maxbyday['%0_O22'] = 100-maxbyday['%0_O2']
                    # minbyday['%0_PAR2'] = 100 - minbyday['%0_PAR']
                    # maxbyday['%0_PAR2'] = 100 - maxbyday['%0_PAR']
                    datahabitable = pd.DataFrame()
                    datahabitable['%0_O2'] = meanbyday['%0_O2']
                    meanbyday.loc[meanbyday['%0_O2']>meanbyday['%0_PAR'],'mean'] = meanbyday['%0_O2']+ meanbyday['%0_T']
                    meanbyday.loc[meanbyday['%0_O2']<=meanbyday['%0_PAR'],'mean'] = meanbyday['%0_PAR']+ meanbyday['%0_T']
                    datahabitable['mean']= meanbyday['mean']
                    #datahabitable.loc[datahabitable['mean']>100,'mean'] = 100
                    
                    stdbyday.loc[stdbyday['%0_O2']>stdbyday['%0_PAR'],'std'] = stdbyday['%0_O2']+ stdbyday['%0_T']
                    stdbyday.loc[stdbyday['%0_O2']<=stdbyday['%0_PAR'],'std'] = stdbyday['%0_PAR']+ stdbyday['%0_T']
                    datahabitable['std']= stdbyday['std']
                    #datahabitable.loc[datahabitable['std']>100,'mean'] = 100
                    
                    medianbyday.loc[medianbyday['%0_O2']>medianbyday['%0_PAR'],'median'] = medianbyday['%0_O2']+ medianbyday['%0_T']
                    medianbyday.loc[medianbyday['%0_O2']<=medianbyday['%0_PAR'],'median'] = medianbyday['%0_PAR']+ medianbyday['%0_T']
                    datahabitable['median']= medianbyday['median']
                    #datahabitable.loc[datahabitable['median']>100,'mean'] = 100
                    
                    minbyday.loc[minbyday['%0_O2']>minbyday['%0_PAR'],'min'] = minbyday['%0_O2']+ minbyday['%0_T']
                    minbyday.loc[minbyday['%0_O2']<=minbyday['%0_PAR'],'min'] = minbyday['%0_PAR']+ minbyday['%0_T']
                    datahabitable['min']= minbyday['min']
                    #datahabitable.loc[datahabitable['min']>100,'mean'] = 100
                    
                    
                    datahabitable['max']= np.nan
                    datahabitable['%0_O2']= maxbyday['%0_O2']
                    datahabitable['%0_PAR']= maxbyday['%0_PAR']
                    datahabitable['%0_T']= maxbyday['%0_T']
                    datahabitable['max'] = datahabitable['%0_O2']+ datahabitable['%0_T']
                    
                    datahabitable.loc[datahabitable['%0_O2']<= datahabitable['%0_PAR'],'max'] = datahabitable['%0_PAR']+ datahabitable['%0_T']
                    
                    #datahabitable.loc[datahabitable['max']>100,'mean'] = 100
                    
                    
                    datahabitable.to_csv(path.join(datafolder, 'data_habitable_g%s_s%s_m%s.csv'%(group,scenario,model)),index=False)

                    stats.norm.ppf(q=0.025)
                    # margin_of_error = z_critical *(stdbyday/sqrt(countbyday.iloc[0,0]))
                    meanplusmargin = meanbyday + stdbyday
                    meanplusmargin.loc[meanplusmargin['%0_T'] > 100, '%0_T'] = 100
                    meanplusmargin.loc[meanplusmargin['%0_PAR2'] > 100, '%0_PAR2'] = 100
                    meanplusmargin.loc[meanplusmargin['%0_O22'] > 100, '%0_O22'] = 100
                    meanlessmargin = meanbyday - stdbyday
                    meanlessmargin.loc[meanlessmargin['%0_T'] < 0, '%0_T'] = 0
                    meanlessmargin.loc[meanlessmargin['%0_PAR2'] < 0, '%0_PAR2'] = 0
                    meanlessmargin.loc[meanlessmargin['%0_O22'] < 0, '%0_O22'] = 0

                    fig1 = plt.figure(2 * group * i, figsize=(18.0, 10.0))
                    ax1 = plt.subplot(211)
                    ax1.fill_between(tt, meanbyday['%0_T'], meanlessmargin['%0_T'], color='lightblue', alpha='0.5')
                    ax1.fill_between(tt, meanbyday['%0_T'], meanplusmargin['%0_T'], color='lightblue', alpha='0.5')
                    ax1.plot_date(tt, meanbyday['%0_T'], '-', color=plt.cm.binary((model / 10)+ 0.3), lw=2, ms=3)
                    # ax1.annotate('Temperature barrier(T>15C)', xy=(80, 80), xycoords='figure points')
                    # ax1.plot_date(tt, meanbyday['%_T']-margin_of_error['%_T'],'k--',lw=2, ms=3)
                    ax1.fill_between(tt, meanbyday['%0_PAR2'], meanplusmargin['%0_PAR2'], color='green', alpha='0.2')
                    ax1.fill_between(tt, meanbyday['%0_PAR2'], meanlessmargin['%0_PAR2'], color='green', alpha='0.2')
                    ax1.plot_date(tt, meanbyday['%0_PAR2'], '-', color=plt.cm.binary((model / 10)+ 0.3), lw=2,
                                    ms=3)
                    # ax1.annotate('PAR barrier(PAR)', xy=(80, 80), xycoords='figure points')
                    # ax1.plot_date(tt, meanbyday['%_PAR'] - margin_of_error['%_PAR'], 'g--', lw=2, ms=3)
                    ax1.fill_between(tt, meanbyday['%0_O22'], meanlessmargin['%0_O22'], color='red', alpha='0.1')
                    ax1.fill_between(tt, meanbyday['%0_O22'], meanplusmargin['%0_O22'], color='red', alpha='0.1')
                    ax1.plot_date(tt, meanbyday['%0_O22'], '-', color=plt.cm.binary((model / 10)+ 0.3), lw=2,
                                    ms=3)

                    ax1.autoscale_view()

                    plt.ylim(0, 100, 50)
                    plt.yticks([0, 25, 50, 75, +100])
                    ax1.set_xlim([datetime(2000, 1, 1), datetime(2000, 12, 31)])
                    plt.ylabel("% Volume O2 and PAR")

                    ax4 = ax1.twinx()
                    ax4.set_ylim(0, 100, 50)
                    plt.yticks([0, 25, 50, 75, +100])
                    ax4.plot_date(tt, meanbyday['%0_O2'], '-', color=plt.cm.binary((model / 10)+ 0.3), lw=2,
                                    ms=3)
                    # ax4.plot_date(tt, meanlessmargin['%_O2'], ':', color=color[model], lw=2, ms=3)
                    # ax4.annotate('Oxygen barrier(O>15C)', xy=(80, 80), xycoords='figure points')
                    # ax4.plot_date(tt, meanbyday['%_O2'] - margin_of_error['%_O2'], 'r--', lw=2, ms=3)
                    # ax4.fill_between(tt, meanbyday['%_O2'], meanlessmargin['%_O2'], color=color2[model],alpha='0.2')
                    # ax4.fill_between(tt, meanbyday['%_O2'], meanplusmargin['%_O2'], color=color2[model], alpha='0.2')
                    plt.gca().invert_yaxis()
                    ax4.xaxis.set_major_locator(months)
                    ax4.xaxis.set_minor_locator(mondays)
                    ax4.xaxis.set_major_formatter(weekFormatter)
                    ax1.fmt_ydata = price
                    ax1.yaxis.grid(True)

                    # winter_months = tt[
                    #    ((tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                    #     |((tt.month == 3)&(tt.day == 20))|(
                    #      (tt.month == 12)&(tt.day == 31))]
                    # summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                    # Loop through groups and add axvspan between winter month
                    # for i in np.arange(0, len(winter_months), 2):
                    #    ax1.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                    # for i in np.arange(0, len(summer_months), 2):
                    #    ax1.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                    # fig1.autofmt_xdate()
                    ax1.tick_params(axis='y', labelsize=16)
                    ax1.tick_params(axis='x', labelsize=16)
                    # plt.legend(loc="lower left", ncol=2)

                    plt.ylabel("% Volume T")
                    plt.xlabel("Date")

                    # ax1.set_xlim(pd.Timestamp("2-12-01"), pd.Timestamp("2011-02-01"))
                    # plt.legend(loc='best')
                    ax4.xaxis_date()

                    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

                    # plt.title('group %s scenario %s'%(group,scenario))# scenario %s' %(group,scenario))
                    plt.tight_layout()

            interet = []
            ert = []
            for i in [1, 2, 3, 4, 5, 6]:
                interet.append(mlines.Line2D([], [], color=plt.cm.binary((i / 10)+ 0.3), markersize=3,
                                                 label=modelname[i - 1]))

            ert.append(Patch(color='green', alpha=0.2, label='PAR'))
            ert.append(Patch(color='lightblue', alpha=0.5, label='T'))
            ert.append(Patch(color='red', alpha=0.1, label='DO'))

            first_legend = plt.legend(loc='upper center', bbox_to_anchor=(0.4, -0.2), fancybox=True, shadow=True,
                                        ncol=1, handles=ert)
            plt.gca().add_artist(first_legend)

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1,
                         handles=interet)
            #plt.show()
            # fig1.savefig(path.join(datafolder, "Figure_synthese_A2_group_%s_scenario_%s_mean.png" %(group, scenario)))
            fig1.savefig(
                path.join(datafolder, "Figure_synthese_A1_group_%s_scenario_%s_mean.png" %(group, scenario)))
            print('completed')

def FishNiche_plot_volume_param(param, lakelistfile, listscenarios, listmodels, calivari, datafolder):
    years = YearLocator() # every year
    months = MonthLocator() # every month
    yearsFmt = DateFormatter('%Y')
    monthsFmt = DateFormatter('%M')
    # fig2,fig3=plt.figure(),plt.figure()
    modelname = ['KNM', 'DMI', 'MPI', 'MOH', 'IPS', 'CNR']
    color = ['c', 'b', 'r', 'm', 'orange', 'k', 'lightblue']
    # color2 = ['white', 'blue', 'black', 'magenta', 'cyan', 'red', 'yellow']
    # jet = cm = plt.get_cmap('Oranges')
    # cNorm = colors.Normalize(vmin=1, vmax=6)
    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    # print(scalarMap.get_clim())
    tt = [[datetime(i - 1, 12, 1), datetime(i + 9, 12, 31)] for i in [1971, 2001, 2031, 2061, 2091]]

    for group in [3, 1]:

        j = 0
        datasheet_all = pd.read_csv(path.join(datafolder, 'His_data_%s2.csv' % group))
        datasheet_all['Date'] = pd.to_datetime(datasheet_all['Date'], format="%Y-%m-%d")

        for scenario in ['rcp85', 'rcp45', 'historical']:
            i = 0
            datasheet_scenario = datasheet_all.loc[datasheet_all['Scenario'] == scenario]
            lakelist = datasheet_scenario['lake_id']
            lakelist = lakelist.drop_duplicates(keep='first')
            sumbyyearlake = pd.DataFrame(columns=['Lake', 'iceIndi'])

            for model in [1, 2, 3, 4, 5, 6]:
                datasheet = datasheet_scenario.loc[datasheet_scenario['Model'] == model]
                datasheet['Year'] = pd.DatetimeIndex(datasheet['Date']).year
                sumall = pd.DataFrame(columns=['iceIndi', 'std', 'more', 'less', 'model'])
                sumall1 = datasheet.groupby(['Year', 'lake_id']).sum()['IceIndicator']
                # sumall['iceIndi'] = sumall1.groupby(['Year']).median()
                # sumall['more'] = sumall1.groupby(['Year']).quantile(0.90)
                # sumall['less'] = sumall1.groupby(['Year']).quantile(0.10)
                sumall['iceIndi'] = sumall1.groupby(['Year']).mean()
                sumall['std'] = sumall1.groupby(['Year']).std()
                sumall['more'] = sumall['iceIndi'] + sumall['std']
                sumall['less'] = sumall['iceIndi'] - sumall['std']
                sumall.loc[sumall['less'] < 0, 'less'] = 0
                sumall['model'] = model
                if len(sumall)!= 0:

                    if i == 0:
                        sumallmodels = sumall
                        i += 1
                    else:
                        sumallmodels = pd.concat([sumallmodels, sumall])

            sumallmodels['scenario'] = scenario
            if j == 0:
                sumallmodelsscena = sumallmodels
                j += 1
            else:
                sumallmodelsscena = pd.concat([sumallmodelsscena, sumallmodels])

        drange = [list(range(1971, 1981)), list(range(2001, 2011)), list(range(2031, 2041)),
                  list(range(2061, 2071)), list(range(2091, 2101))]
        # # create as many subplots as there are date ranges
        params = {
            'axes.labelsize': 16,
            'text.fontsize': 16,
            'legend.fontsize': 16,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'text.usetex': False,  # instead of 4.5, 4.5
            'figure.figsize': [18, 6]  # instead of 4.5, 4.5
        }

        plt.rcParams.update(params)
        fig, axes = plt.subplots(ncols=len(drange), sharey=True)
        fig.subplots_adjust(bottom=0.3, wspace=0)
        ymax = 1.1 * sumallmodels['iceIndi'].max()
        # # loop over subplots and limit each to one date range
        listonplothis = []
        for i, ax in enumerate(axes):
            print(i)
            ax.set_xlim(drange[i][0], drange[i][9])
            if group == 3:
                ax.set_ylim(-1, 60)
            else:
                ax.set_ylim(0, 250)
            interethis = []
            interet85 = []
            interet45 = []
            listonplot45 = []
            listonplot85 = []

            for model in [1, 2, 3, 4, 5, 6]:
                if len(sumallmodelsscena['model'] == model)!= 0:
                    summodel = sumallmodelsscena[sumallmodelsscena['model'] == model]

                    if i < 2:
                        y = summodel[summodel.index < drange[i][9] + 1]['iceIndi']
                        more = summodel[summodel.index < drange[i][9] + 1]['more']
                        less = summodel[summodel.index < drange[i][9] + 1]['less']
                        x = y.index
                        ax.fill_between(x, y, less, color='lightblue', alpha='0.3')
                        ax.fill_between(x, y, more, color='lightblue', alpha='0.3')
                        ax.plot(x, y, '-o', markersize=8, color=plt.cm.Blues((model / 10)+ 0.3))
                        ax.xaxis.set_ticks(np.arange(drange[i][0], drange[i][9] + 2, 3))
                        plt.setp(ax.get_xticklabels()) # ,rotation=90)
                        listonplothis.append(model)
                        print(listonplothis)
                        if i != 0:
                            ax.tick_params(axis="y", which="both", length=0)
                    elif i < 4:
                        y = summodel[summodel.index < drange[i + 1][0]]
                        y = y[y.index > drange[i - 1][9]]

                        x = y.index
                        y1 = y[y['scenario'] == 'rcp45']
                        more1 = y1['more']
                        less1 = y1['less']
                        y1 = y1['iceIndi']

                        x1 = y1.index
                        y2 = y[y['scenario'] == 'rcp85']
                        more2 = y2['more']
                        less2 = y2['less']
                        y2 = y2['iceIndi']

                        x2 = y2.index
                        if len(y1)!= 0:
                            ax.fill_between(x1, y1, less1, color='black', alpha='0.2')
                            ax.fill_between(x1, y1, more1, color='black', alpha='0.2')
                            ax.plot(x1, y1, '-o', markersize=8, color=plt.cm.binary((model / 10)+ 0.3))
                            ax.xaxis.set_ticks(np.arange(drange[i][0], drange[i][9] + 2, 3))
                            plt.setp(ax.get_xticklabels())
                            listonplot45.append(model)
                            if i != 0:
                                ax.tick_params(axis="y", which="both", length=0)
                        if len(y2)!= 0:
                            if model != 5:
                                ax.fill_between(x2, y2, less2, color='orange', alpha='0.2')
                                ax.fill_between(x2, y2, more2, color='orange', alpha='0.2')
                                ax.plot(x2, y2, '-o', markersize=8,
                                          color=plt.cm.Wistia((model / 10)+ 0.067 * model))

                                ax.xaxis.set_ticks(np.arange(drange[i][0], drange[i][9] + 2, 3))
                                plt.setp(ax.get_xticklabels())
                                listonplot85.append(model)
                                if i != 0:
                                    ax.tick_params(axis="y", which="both", length=0)

                    else:
                        y = summodel[summodel.index >(drange[i][0] - 1)]
                        x = y.index

                        y1 = y[y['scenario'] == 'rcp45']
                        more1 = y1['more']
                        less1 = y1['less']
                        y1 = y1['iceIndi']
                        x1 = y1.index
                        y2 = y[y['scenario'] == 'rcp85']
                        more2 = y2['more']
                        less2 = y2['less']
                        y2 = y2['iceIndi']
                        x2 = y2.index
                        if len(y1)!= 0:
                            ax.fill_between(x1, y1, less1, color='black', alpha='0.2')
                            ax.fill_between(x1, y1, more1, color='black', alpha='0.2')
                            ax.plot(x1, y1, '-o', markersize=8, color=plt.cm.binary((model / 10)+ 0.4))
                            loc = YearLocator()
                            fmt = DateFormatter("%Y")
                            ax.xaxis.set_ticks(np.arange(drange[i][0], drange[i][9] + 2, 3))
                            plt.setp(ax.get_xticklabels())
                            listonplot45.append(model)
                            if i != 0:
                                ax.tick_params(axis="y", which="both", length=0)
                        if len(y2)!= 0:
                            ax.fill_between(x2, y2, less2, color='orange', alpha='0.2')
                            ax.fill_between(x2, y2, more2, color='orange', alpha='0.2')
                            ax.plot(x2, y2, '-o', markersize=8, color=plt.cm.Wistia((model / 10)+ 0.067 * model))
                            loc = YearLocator()
                            fmt = DateFormatter("%Y")
                            ax.xaxis.set_ticks(np.arange(drange[i][0], drange[i][9] + 2, 3))
                            plt.setp(ax.get_xticklabels())
                            listonplot85.append(model)
                            if i != 0:
                                ax.tick_params(axis="y", which="both", length=0)
        listonplot85 = list(set(listonplot85))
        listonplot45 = list(set(listonplot45))
        listonplothis = list(set(listonplothis))
        for k in [1, 2, 3, 4, 5, 6]:
            interethis.append(
                mlines.Line2D([], [], color=plt.cm.Blues((k / 10)+ 0.3), markersize=3, label=modelname[k - 1]))
        for i in [1, 2, 3, 4, 5, 6]:
            interet45.append(
                mlines.Line2D([], [], color=plt.cm.binary((i / 10)+ 0.4), markersize=3, label=modelname[i - 1]))
        for j in [1, 2, 3, 4, 5, 6]:
            interet85.append(mlines.Line2D([], [], color=plt.cm.Wistia((j / 10)+ 0.067 * j), markersize=3,
                                               label=modelname[j - 1]))

        interethis.append(Patch(color='lightblue', alpha=0.3, label='Historical'))
        interet85.append(Patch(color='orange', alpha=0.2, label='rcp 8.5'))
        interet45.append(Patch(color='black', alpha=0.2, label='rcp 4.5'))

        # first_legend = plt.legend(loc='upper center', bbox_to_anchor=(-4.21, 1.145), fancybox=True, shadow=True, ncol=1, handles=interethis)
        # plt.gca().add_artist(first_legend)
        # second_legend = plt.legend(loc='upper center', bbox_to_anchor=(-1.89, 1.145), fancybox=True, shadow=True,ncol=1, handles=interet45)
        # plt.gca().add_artist(second_legend)
        # plt.legend(loc='upper center', bbox_to_anchor=(-0.4, 1.145), fancybox=True, shadow=True,ncol=1, handles=interet85)
        fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
        fig.text(0.5, 0.03, 'Years', ha='center')
        fig.text(0.03, 0.55, 'Sum of days with ice cover', va='center', rotation='vertical')
        plt.show()
        fig.savefig(
            path.join(datafolder, "Figure_indicator_A2_group_%s_mean.png" %(group)))
        fig.savefig(
            path.join(datafolder, "Figure_indicator_A2_group_%s_mean.eps" %(group)))
        print('completed')

def generate_timeseries_his_by_model(listmodels, listscenarios, lakelistfile, datafolder):
    i = 0
    j = 0
    complete_data = pd.DataFrame()
    for model in listmodels:
        for scenario in listscenarios:
            exA, y1A, exB, y1B = scenarios[scenario]
            y2B = y1B + 4
            m1, m2 = models[model]
            with open(lakelistfile, 'rU')as f:
                skip = f.readline()
                lakes = f.readlines()
                nlakes = len(lakes)
                f.close()
#            lake_id, subid, name, eh, area, depth, longitude, latitude, volume \
#                = lakes[1].strip().split(',')
            lake_id, subid, name, eh, area, depth, longitude, latitude, volume, mean_depth, sediment, mean_calculated = lakes[i].strip().split(',')
            eh = eh[2:] if eh[:2] == '0x' else eh
            while len(eh)< 6:
                eh = '0' + eh
            
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]
            outdir = os.path.join(datafolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
            tzt_dir = path.join(outdir, 'Tzt.csv')# print(tzt_dir)
            if j ==0:
                 datasheet2 = os.path.join(datafolder, 'fish_niche_export_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(
                         m1, exA, exB, m2, y1A, y2B))
                 timeseries2 = pd.read_csv(datasheet2)
                 timeseries2 = timeseries2[['lakeid','Total Volume']].drop_duplicates(keep='first').reset_index()
                 listlake = pd.read_csv(os.path.join(lakelistfile),encoding='cp1252')
                 volumelist =  pd.DataFrame(columns=['lake_id','Total Volume'])
                 volumelist['lake_id']=listlake['lake_id']
     
                 volumelist['Total Volume']=timeseries2['Total Volume']
                 volumelist=volumelist.drop_duplicates(keep='first')
                     # volumelist.to_csv(os.path.join(datafolder,'volume_list.csv'),index=False)
                 j+=1
            if not os.path.exists(os.path.join(datafolder, 'fish_niche_export_his_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(m1, exA, exB, m2, y1A, y2B))):
                 # if os.path.exists(tzt_dir):
                 cmd = 'matlab -wait -r -nosplash -nodesktop generateParamTimeseries(\'%s\',\'%s\',\'%s\',\'%s\',%d,\'%s\',%d,\'%s\');quit' %(lakelistfile, m1, m2, exA, y1A, exB, y1B, datafolder)
                 print(lakelistfile,m1,m2,exA,y1A,exB,y2B,datafolder,cmd)
                 os.system(cmd)
                 print('complete')
            else:
                 print('nan')

#     
#    i = 0
#    j = 0
#    for model in listmodels:
#        for scenario in listscenarios:
#            exA, y1A, exB, y1B = scenarios[scenario]
#            y2B = y1B + 4
#            m1, m2 = models[model]
#
#            with open(lakelistfile, 'rU')as f:
#                lakes = f.readlines()
#                nlakes = len(lakes)
#                f.close()
#
#            lake_id, subid, name, eh, area, depth, longitude, latitude, volume , meandepth,sedimentArea,meanCalculated  = lakes[1].strip().split(',')
#            eh = eh[2:] if eh[:2] == '0x' else eh
#            while len(eh)< 6:
#                eh = '0' + eh
#            d1, d2, d3 = eh[:2], eh[:4], eh[:6]
#            outdir = path.join(datafolder, d1, d2, d3,
#                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
#            if j == 0:
#                datasheet2 = path.join(datafolder,
#                                         'fish_niche_export_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(
#                                             m1, exA, exB, m2, y1A, y2B))
#                timeseries2 = pd.read_csv(datasheet2)
#                timeseries2 = timeseries2[['lakeid', 'Total Volume']].drop_duplicates(
#                    keep='first').reset_index()
#                listlake = pd.read_csv(path.join('2017SwedenList.csv'), encoding='cp1252')
#                volumelist = pd.DataFrame(columns=['lake_id', 'Total Volume'])
#                volumelist['lake_id'] = listlake['lake_id']
#
#                volumelist['Total Volume'] = timeseries2['Total Volume']
#                volumelist = volumelist.drop_duplicates(keep='first')
#                volumelist.to_csv(path.join(datafolder, 'volume_list.csv'), index=False)
#                j += 1
#            if path.exists(path.join(datafolder, 'fish_niche_export_His_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(
#                    m1, exA, exB, m2, y1A, y2B))):
#
#                his_dir = path.join(datafolder, 'fish_niche_export_His_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(
#                    m1, exA, exB, m2, y1A, y2B))
#                print(his_dir)
#                datasheet = pd.read_csv(his_dir)
#                datasheet['Date'] = pd.to_datetime(datasheet['Date'], format="%d.%m.%Y")
#                timeseries = pd.DataFrame(
#                    columns=['Date', 'Model', 'Scenario', 'Lake_group', 'rho_snow', 'IceIndicator', 'Total Volume'])
#                timeseries['Date'] = datasheet['Date']
#                timeseries['Model'] = model
#                timeseries['Scenario'] = exA
#                timeseries['rho_snow'] = datasheet['rho_snow']
#                timeseries['IceIndicator'] = datasheet['IceIndicator']
#
#                timeseries['Lake_group'] = 2
#                for lakenum in np.arange(1, nlakes):
#                    lake_id, subid, name, eh, area, depth, longitude, latitude, volume \
#                        = lakes[lakenum].strip().split(',')
#                    # volume = volumelist.loc[volumelist['lake_id']==int(lake_id),'Total Volume'].reset_index().iloc[0,1]
#                    datasheet.loc[datasheet['lakeid'] == int(lake_id), 'TotalV'] = float(volume)
#
#                timeseries['Total Volume'] = datasheet['TotalV']
#                timeseries['lake_id'] = datasheet['lakeid']
#
#                timeseries.loc[timeseries['Total Volume'] < 1.0e7, 'Lake_group'] = 1
#                timeseries.loc[timeseries['Total Volume'] > 5.0e9, 'Lake_group'] = 3
#
#                print('completed')
#                if i == 0:
#                    complete_data = timeseries
#                    print('first')
#                    i += 1
#                else:
#                    complete_data = complete_data.append(timeseries, ignore_index=True)
#                    print('added')
#
#    complete_data.loc[complete_data['Lake_group'] == 1].to_csv(path.join(datafolder, 'His_data_12.csv'),
#                                                                 index=False)
#    print('1_save')
#    complete_data.loc[complete_data['Lake_group'] == 2].to_csv(path.join(datafolder, 'His_data_22.csv'),
#                                                                 index=False)
#    print('2_save')
#    complete_data.loc[complete_data['Lake_group'] == 3].to_csv(path.join(datafolder, 'His_data_32.csv'),
#                                                                 index=False)
    #print('end')
            if j ==0:
                datasheet2 = os.path.join(datafolder, 'fish_niche_export_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(
                        m1, exA, exB, m2, y1A, y2B))
                timeseries2 = pd.read_csv(datasheet2)
                timeseries2 = timeseries2[['lakeid','Total Volume']].drop_duplicates(keep='first').reset_index()
                listlake = pd.read_csv(os.path.join(lakelistfile),encoding='cp1252')
                volumelist =  pd.DataFrame(columns=['lake_id','Total Volume'])
                volumelist['lake_id']=listlake['lake_id']

                volumelist['Total Volume']=timeseries2['Total Volume']
                volumelist=volumelist.drop_duplicates(keep='first')
                    # volumelist.to_csv(os.path.join(datafolder,'volume_list.csv'),index=False)
                j+=1

            tzt_dir = os.path.join(outdir,'Tzt.csv')
            if os.path.exists(os.path.join(datafolder, 'fish_niche_export_his_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(m1, exA, exB, m2, y1A, y2B))):
                # if os.path.exists(tzt_dir):
                cmd = 'matlab -wait -r -nosplash -nodesktop generateParamTimeseries(\'%s\',\'%s\',\'%s\',\'%s\',%d,\'%s\',%d,\'%s\');quit' %(lakelistfile, m1, m2, exA, y1A, exB, y1B, datafolder)
                print(lakelistfile,m1,m2,exA,y1A,exB,y2B,datafolder,cmd)
                os.system(cmd)
                print('complete')
            else:
                print('nan')

    # 
    # i = 0
    # j = 0
    # for model in listmodels:
    #     for scenario in listscenarios:
    #         exA, y1A, exB, y1B = scenarios[scenario]
    #         y2B = y1B + 4
    #         m1, m2 = models[model]
    #
    #         with open(lakelistfile, 'rU')as f:
    #             lakes = f.readlines()
    #             nlakes = len(lakes)
    #             f.close()
    #
    #         lake_id, subid, name, eh, area, depth, longitude, latitude, volume = lakes[1].strip().split(',')
    #         eh = eh[2:] if eh[:2] == '0x' else eh
    #         while len(eh)< 6:
    #             eh = '0' + eh
    #         d1, d2, d3 = eh[:2], eh[:4], eh[:6]
    #         outdir = path.join(datafolder, d1, d2, d3,
    #                              'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
    #         if j == 0:
    #             datasheet2 = path.join(datafolder,
    #                                      'fish_niche_export_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(
    #                                          m1, exA, exB, m2, y1A, y2B))
    #             timeseries2 = pd.read_csv(datasheet2)
    #             timeseries2 = timeseries2[['lakeid', 'Total Volume']].drop_duplicates(
    #                 keep='first').reset_index()
    #             listlake = pd.read_csv(path.join('2017SwedenList.csv'), encoding='cp1252')
    #             volumelist = pd.DataFrame(columns=['lake_id', 'Total Volume'])
    #             volumelist['lake_id'] = listlake['lake_id']
    #
    #             volumelist['Total Volume'] = timeseries2['Total Volume']
    #             volumelist = volumelist.drop_duplicates(keep='first')
    #             volumelist.to_csv(path.join(datafolder, 'volume_list.csv'), index=False)
    #             j += 1
    #         if path.exists(path.join(datafolder, 'fish_niche_export_His_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(
    #                 m1, exA, exB, m2, y1A, y2B))):
    #
    #             his_dir = path.join(datafolder, 'fish_niche_export_His_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(
    #                 m1, exA, exB, m2, y1A, y2B))
    #             datasheet = pd.read_csv(his_dir)
    #             datasheet['Date'] = pd.to_datetime(datasheet['Date'], format="%d.%m.%Y")
    #             timeseries = pd.DataFrame(
    #                 columns=['Date', 'Model', 'Scenario', 'Lake_group', 'rho_snow', 'IceIndicator', 'Total Volume'])
    #             timeseries['Date'] = datasheet['Date']
    #             timeseries['Model'] = model
    #             timeseries['Scenario'] = exA
    #             timeseries['rho_snow'] = datasheet['rho_snow']
    #             timeseries['IceIndicator'] = datasheet['IceIndicator']
    #
    #             timeseries['Lake_group'] = 2
    #             for lakenum in np.arange(1, nlakes):
    #                 lake_id, subid, name, eh, area, depth, longitude, latitude, volume \
    #                     = lakes[lakenum].strip().split(',')
    #                 # volume = volumelist.loc[volumelist['lake_id']==int(lake_id),'Total Volume'].reset_index().iloc[0,1]
    #                 datasheet.loc[datasheet['lakeid'] == int(lake_id), 'TotalV'] = float(volume)
    #
    #             timeseries['Total Volume'] = datasheet['TotalV']
    #             timeseries['lake_id'] = datasheet['lakeid']
    #
    #             timeseries.loc[timeseries['Total Volume'] < 1.0e7, 'Lake_group'] = 1
    #             timeseries.loc[timeseries['Total Volume'] > 5.0e9, 'Lake_group'] = 3
    #
    #             print('completed')
    #             if i == 0:
    #                 complete_data = timeseries
    #                 print('first')
    #                 i += 1
    #             else:
    #                 complete_data = complete_data.append(timeseries, ignore_index=True)
    #                 print('added')
    #
    # complete_data.loc[complete_data['Lake_group'] == 1].to_csv(path.join(datafolder, 'His_data_12.csv'),
    #                                                              index=False)
    # print('1_save')
    # complete_data.loc[complete_data['Lake_group'] == 2].to_csv(path.join(datafolder, 'His_data_22.csv'),
    #                                                              index=False)
    # print('2_save')
    # complete_data.loc[complete_data['Lake_group'] == 3].to_csv(path.join(datafolder, 'His_data_32.csv'),
    #                                                              index=False)
    # print('end')

def generate_timeseries_by_model(listmodels, listscenarios, lakelistfile, datafolder):
    i = 0
    complete_data = pd.DataFrame()
    for model in listmodels:
        for scenario in listscenarios:
            exA, y1A, exB, y1B = scenarios[scenario]
            y2B = y1B + 4
            m1, m2 = models[model]
            with open(lakelistfile, 'rU')as f:
                skip = f.readline()
                lakes = f.readlines()
                nlakes = len(lakes)
                f.close()
#            lake_id, subid, name, eh, area, depth, longitude, latitude, volume \
#                = lakes[1].strip().split(',')
            lake_id, subid, name, eh, area, depth, longitude, latitude, volume, mean_depth, sediment, mean_calculated = lakes[i].strip().split(',')
            
            eh = eh[2:] if eh[:2] == '0x' else eh
            while len(eh)< 6:
                eh = '0' + eh
            
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]
            outdir = path.join(datafolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
            tzt_dir = path.join(outdir, 'Tzt.csv')
            # print(tzt_dir)
            if path.exists(path.join(datafolder, 'fish_niche_export1_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(
                    m1, exA, exB, m2, y1A, y2B))):
                 # if os.path.exists(os.path.join(datafolder, 'fish_niche_export_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(m1, exA, exB, m2, y1A, y2B))):
                 print(tzt_dir)
                 if os.path.exists(tzt_dir):
                     cmd = 'matlab -wait -r -nosplash -nodesktop generateVolumeTimeseries(\'%s\',\'%s\',\'%s\',\'%s\',%d,\'%s\',%d,\'%s\');quit' %(lakelistfile, m1, m2, exA, y1A, exB, y1B, datafolder)
                     print(cmd)
                     os.system(cmd)
                 print('nan')
            else:

#               datasheet = path.join(datafolder, 'fish_niche_export1_EUR-11_%s_%s-%s_%s_%s0101-%s1231.csv' %(
#                   m1, exA, exB, m2, y1A, y2B))
#               #print(datasheet)
#               timeseries = pd.read_csv(datasheet)
#               timeseries['Date'] = pd.to_datetime(timeseries['Date'], format="%d.%m.%Y")
#               timeseries_select = pd.DataFrame(
#                   columns=['Date', 'Model', 'Scenario', 'Lake_group', '%_T', '%_O2', '%_PAR', 'Total Volume','%_habitable'])
#               timeseries_select['Date'] = timeseries['Date']
#               timeseries_select['Model'] = model
#               timeseries_select['Scenario'] = exA
#               timeseries_select['Lake_group'] = 2
#               timeseries_select['Total Volume'] = timeseries['Total Volume']
#               timeseries_select.loc[timeseries['Total Volume'] < 1.0e7, 'Lake_group'] = 1
#               timeseries_select.loc[timeseries['Total Volume'] > 5.0e9, 'Lake_group'] = 3
#               timeseries_select['%_T'] = timeseries['Volume with T < 15 C'] / timeseries['Total Volume']
#               timeseries_select['%_O2'] = timeseries['Volume with O2 > 3000'] / timeseries['Total Volume']
#               timeseries_select['%_PAR'] = timeseries['Volume with PPFD > 1%'] / timeseries['Total Volume']
#               timeseries_select['%_habitable'] = timeseries_select['%_T']+timeseries_select['%_O2']+timeseries_select['%_PAR']
#               timeseries_select.loc[timeseries_select['%_habitable']> 1,'%_habitable'] = 1
               print('completed')
#               if i == 0:
#                   complete_data = timeseries_select
#                   print('first')
#                   i += 1
#               else:
#                   complete_data = complete_data.append(timeseries_select, ignore_index=True)
#                   print('added')
#    complete_data.loc[complete_data['Lake_group'] == 1].to_csv(path.join(datafolder, 'complete_data_11.csv'),
#                                                                 index=False)
#    print('1_save')
#    complete_data.loc[complete_data['Lake_group'] == 2].to_csv(path.join(datafolder, 'complete_data_21.csv'),
#                                                                 index=False)
#    print('2_save')
#    complete_data.loc[complete_data['Lake_group'] == 3].to_csv(path.join(datafolder, 'complete_data_31.csv'),
#                                                                 index=False)
    print('end')

def FishNiche_validate_results(scenarioid, modelid, lakelistfile, calivari, k_BOD):
    global fig1
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2B = y1B + 4
    timeaxis = pd.date_range("%s-01-01" % y1A, "%s-12-31" % y2B, freq="D")
    timestep = len(timeaxis)
    fig1, fig2, fig3, fig4 = plt.figure(), plt.figure(), plt.figure(), plt.figure()

    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)

    # lakeswehavedatafor = [698]
    lakeswehavedatafor = [14939, 67035, 16765, 698, 33494, 33590, 99045, 30704, 19167, 31895, 310, 32276, 99045, 99516,
                          6950, 33590, 33494, 31358, 23177]
    # lakeswehavedatafor = [698, 67035, 31895, 310, 32276]
    calibrationdata = pd.DataFrame(index=np.arange(0, len(lakeswehavedatafor)), columns=['lake_id', calivari])
    place = -1
    numberlakes = 0
    for lakenum in np.arange(1, nlakes):
        lake_id, subid, name, eh, area, depth, longitude, latitude, volume, meandepth \
            = lakes[lakenum].strip().split(',')
        if int(lake_id)in lakeswehavedatafor:
            numberlakes += 1
            maxdepth = float(depth)
            # getOutputPathFromEbHex
            eh = eh[2:] if eh[:2] == '0x' else eh
            while len(eh)< 6:
                eh = '0' + eh
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]

            outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
            print(outdir)

            O2model = pd.read_csv(path.join(outdir, 'O2zt.csv'), header=None)
            Tmodel = pd.read_csv(path.join(outdir, 'Tzt.csv'), header=None)

            filename = '../sweden_inflow_data/Validation_data_for_lookup.xlsx'
            # need to be change if the file change
            worksheet = pd.read_excel(filename, sheet_name=lake_id)
            dates = worksheet['date']
            depthdata = worksheet['depth(max)']
            O2raw = worksheet['O2(mg/l)'] * 1000
            Traw = worksheet['Water temp(°C)']
            anydata_t, anydata_o = False, False
            T_model_samples, T_data_samples, T_depths = [], [], []
            O2_model_samples, O2_data_samples, O2_depths = [], [], []
            crashed = False

            try:
                for ii in np.arange(0, len(dates)):
                    dnum = datetime.strptime(dates[ii], "%Y-%m-%d").date()
                    start = timeaxis[0].date()
                    dateindex =(dnum - start).days
                    if dateindex >= 1 and dateindex <= timestep:
                        if lake_id == '698':
                            if depthdata[ii] <= 2:
                                depth = ceil(depthdata[ii])
                                T_data = Traw[ii]
                                T_model = Tmodel.loc[dateindex, depth - 1]

                                if not isnan(T_data):
                                    T_data_samples.append(T_data)
                                    T_model_samples.append(T_model)
                                    T_depths.append(depth)
                                    anydata_t = True

                            depthO = ceil(depthdata[ii])
                            O2_data = O2raw[ii]
                            O2_model = O2model.loc[dateindex, depthO - 1]
                            if not isnan(O2_data):
                                O2_data_samples.append(O2_data)
                                O2_model_samples.append(O2_model)
                                O2_depths.append(depthO)
                                anydata_o = True

                        elif lake_id == '67035':
                            if depthdata[ii] <= 1 or depthdata[ii] >= 60:
                                depth = ceil(depthdata[ii])
                                T_data = Traw[ii]
                                T_model = Tmodel.loc[dateindex, depth - 1]

                                if not isnan(T_data):
                                    T_data_samples.append(T_data)
                                    T_model_samples.append(T_model)
                                    T_depths.append(depth)
                                    anydata_t = True

                            depthO = ceil(depthdata[ii])
                            O2_data = O2raw[ii]
                            O2_model = O2model.loc[dateindex, depthO - 1]
                            if not isnan(O2_data):
                                O2_data_samples.append(O2_data)
                                O2_model_samples.append(O2_model)
                                O2_depths.append(depthO)
                                anydata_o = True

                        else:
                            depth = ceil(depthdata[ii])
                            O2_data = O2raw[ii]
                            T_data = Traw[ii]
                            O2_model = O2model.loc[dateindex, depth - 1]
                            T_model = Tmodel.loc[dateindex, depth - 1]

                            if not isnan(T_data):
                                T_data_samples.append(T_data)
                                T_model_samples.append(T_model)
                                T_depths.append(depth)
                                anydata_t = True

                            if not isnan(O2_data):
                                O2_data_samples.append(O2_data)
                                O2_model_samples.append(O2_model)
                                O2_depths.append(depth)
                                anydata_o = True
            except:
                crashed = True
                print("Error when handling lake %s %s \n" %(name, lake_id))
            label_t = "%s %s" %(name, lake_id)
            label_o = "%s %s" %(name, lake_id)
            if crashed:
                label_t =("%s - Error in program,unreliable" % label_t)
                label_o =("%s - Error in program,unreliable" % label_o)
            else:
                if not anydata_t:
                    label_t = "%s\n(No data in timespan)" % label_t
                else:
                    RMSE_T = round(sqrt(sklearn.metrics.mean_squared_error(T_data_samples, T_model_samples)),
                                     2)
                    corr_T = scipy.stats.pearsonr(T_data_samples, T_model_samples)
                    pearsont = '%.3f' %(round(corr_T[0], 3))
                    label_t = "%s\n(RMSE: %s Pearson: %s)" %(label_t, RMSE_T, pearsont)

                if not anydata_o:
                    label_o = "%s\n(No data in timespan)" % label_o
                else:
                    RMSE_O = round(sqrt(sklearn.metrics.mean_squared_error(O2_data_samples, O2_model_samples)),
                                     2)
                    corr_O = scipy.stats.pearsonr(O2_data_samples, O2_model_samples)
                    pearsono = '%.3f' %(round(corr_O[0], 3))

                    label_o = "%s\n(RMSE: %s Pearson: %s)" %(label_o, RMSE_O, pearsono)

            lineStart = 0
            lineEnd = 25
            rows = ceil(sqrt(nlakes - 1))
            cols = ceil((nlakes - 1)/ rows)

            plt.rc('axes', linewidth=2)
            fontsize = 30
            lw = 0.5
            markers = ['o', 'v', '+', '*', '8', '^', 'D', 's', 'd', 'p', 'x']
            markers = ['d', 'v', '+', '*', '8', 's', 'D', '^', 'o']
            s = [100, 100 * 1.5, 100, 100 * 2, 100, 100 * 1.5, 100, 100, 100 * 1.5, 100, 100]
            i = 100 * 1.5
            s = [i * 1.5, i * 2, i, i * 2.5, i, i * 1.5, i, i * 2, i]
            if lakenum < len(markers):
                mark = markers[lakenum - 1]
                size = s[lakenum - 1]
            else:
                mark = markers[0]
                size = s[0]
            alpha = 0.8
            params = {
                'axes.labelsize': 30,
                'text.fontsize': 30,
                'legend.fontsize': 30,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'text.usetex': False,
                'figure.figsize': [14, 12]  # instead of 4.5, 4.5
            }

            plt.rcParams.update(params)
            colorset = ['k', 'cyan', 'maroon', 'C1', 'C6', 'C2', 'C3', 'maroon', 'gold',
                        'C8']  # ['C1','C3','y','k','w','C7','C0','C2','C9']
            place += 1
            if anydata_t:
                fig1 = plt.figure(k_BOD * 100) # ,figsize=(3.0, 3.0))
                ax = plt.gca()
                color = [1 -(i / maxdepth)for i in T_depths]
                points = plt.scatter(T_data_samples, T_model_samples, label=label_t, s=size, c=color, edgecolors='k',
                                       linewidths=lw, cmap='Blues_r', marker=mark, alpha=alpha)
                # for i,txt in enumerate(T_depths):
                # plt.annotate(txt,(T_data_samples[i],T_model_samples[i]))
                plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', linewidth=1.0)
                plt.tick_params(axis='both', which='major') # , labelsize=14)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(fontsize)
                    # tick.label1.set_fontweight('bold')
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(fontsize)
                    # tick.label1.set_fontweight('bold')
                plt.xlabel('Observed Temperature($\degree$C)') # ,fontweight='bold')
                plt.ylabel('Modeled Temperature($\degree$C)') # ,fontweight='bold')
                # plt.legend(loc=4)
                if numberlakes == len(lakeswehavedatafor):
                    cb = plt.colorbar(points, ticks=np.linspace(1.1, -0.1, 12, endpoint=True))
                    # cb.set_label(weight='bold')
                    cb.ax.set_yticklabels(['surface', '', '', '', '', '', '', '', 'max depth']) # ,weight='bold')
                plt.tight_layout()

            fig2 = plt.figure((k_BOD + 1)* 100) # ,figsize=(18.0, 10.0))
            ax = plt.gca()

            plt.subplot(rows, cols, lakenum)
            points = plt.scatter(T_data_samples, T_model_samples, c=T_depths, edgecolors='k', label=label_t, s=40,
                                   cmap='bone_r') # T_depths
            plt.colorbar(points)
            plt.clim([0, maxdepth])
            plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
            plt.title(label_t, fontsize=14, y=1.02)
            plt.subplots_adjust(hspace=0.6, wspace=0.25)
            plt.tick_params(axis='both', which='major') # , labelsize=14)
            plt.yticks(np.arange(0, 35, 5))
            fig2.text(0.5, 0.05, 'Modeled Temperature($\degree$C)', ha='center', size=20)
            fig2.text(0.05, 0.5, 'Observed Temperature($\degree$C)', va='center', rotation='vertical', size=20)
            fig2.text(0.92, 0.5, 'Depth(m)', va='center', rotation='vertical', size=20)

            lineStart = 0
            lineEnd = 20000
            if anydata_o:
                fig3 = plt.figure((k_BOD + 3)* 100) # ,figsize=(18.0, 10.0))
                ax = plt.gca()
                color = [1 -(i / maxdepth)for i in O2_depths]
                points = plt.scatter([i / 1000 for i in O2_data_samples], [i / 1000 for i in O2_model_samples],
                                       label=label_o, alpha=alpha, s=size, linewidths=lw, c=color, edgecolors='k',
                                       cmap='Reds_r', marker=mark)

                # for i,txt in enumerate(O2_depths):
                # plt.annotate(txt,(O2_data_samples[i],O2_model_samples[i]))
                plt.plot([lineStart, lineEnd / 10 ** 3], [lineStart, lineEnd / 10 ** 3], 'k-', linewidth=1.0)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(fontsize)
                # tick.label1.set_fontweight('bold')
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(fontsize)
                    # tick.label1.set_fontweight('bold')
                plt.tick_params(axis='both', which='major') # , labelsize=14)

                plt.xlabel(r'Observed Oxygen(mg/L)') # ,fontweight='bold' )
                plt.ylabel(r'Modeled Oxygen(mg/L)') # ,fontweight='bold' )
                # plt.legend(loc=4)
                # plt.ylim(0,20000,2)
                plt.yticks([0, 5, 10, 15, 20])
                plt.xticks([0, 5, 10, 15, 20])
                if numberlakes == len(lakeswehavedatafor):
                    cb = plt.colorbar(points, ticks=np.linspace(1.1, -0.1, 12, endpoint=True))
                    # cb.set_label(weight='bold')
                    cb.ax.set_yticklabels(['surface', '', '', '', '', '', '', '', 'max depth']) # ,weight='bold')
                plt.tight_layout()

            fig4 = plt.figure((k_BOD + 4)* 100) # ,figsize=(18.0, 10.0))
            ax = plt.gca()
            plt.subplot(rows, cols, lakenum)
            points = plt.scatter([i / 10 ** 3 for i in O2_data_samples], [j / 10 ** 3 for j in O2_model_samples],
                                   c=O2_depths, edgecolors='k', label=label_o, s=40, cmap='bone_r')
            cb = plt.colorbar(points)
            plt.clim([0, maxdepth])
            cb.ax.tick_params(labelsize=14)
            plt.plot([lineStart, lineEnd / 10 ** 3], [lineStart, lineEnd / 10 ** 3], 'k-')
            plt.title(label_o, fontsize=14, y=1.02)
            # ax.text(3, 2, u'u')
            plt.subplots_adjust(hspace=0.6, wspace=0.25)
            plt.tick_params(axis='both', which='major') # , labelsize=14)
            plt.yticks(np.arange(0, 24, 4))
            fig4.text(0.5, 0.05, 'Modeled Oxygen(mg/L)', ha='center', size=20)
            fig4.text(0.05, 0.5, 'Observed Oxygen(mg/L)', va='center', rotation='vertical', size=20)
            fig4.text(0.92, 0.5, 'Depth(m)', va='center', rotation='vertical', size=20)



            # calibration = abs(sum(T_model_samples)-sum(T_data_samples))**2
    # calibrationdata.to_csv(os.path.join(outputfolderdata, 'calibration,csv'))
    # df_csv = pd.read_csv(os.path.join(outputfolderdata, 'calibration.csv'))
    # result = pd.merge(df_csv, calibrationdata, on='lake_id')
    # result.to_csv(os.path.join(outputfolderdata, 'calibration.csv'),index=False)

    fig1.savefig(path.join(datafolder, "Figure_1_temp_test%s.png" % calivari))
    # plt.gcf().clear(fig1)
    fig2.savefig(path.join(datafolder, "Figure_2_temp alltest_%s.png" % calivari))
    # plt.gcf().clear(fig2)
    fig3.savefig(path.join(datafolder, "Figure_1_Oxygen_test%s.png" % calivari))
    # plt.gcf().clear(fig3)
    fig4.savefig(path.join(datafolder, "Figure_2_Oxygen alltest_%s.png" % calivari))

    # return calibrationdata

def FishNiche_graph_temp_time(scenarioid, modelid, lakelistfile, k_BOD, i):
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2B = y1B + 4
    timeaxis = pd.date_range("%s-01-01" % y1A, "%s-12-31" % y2B, freq="D")
    timestep = len(timeaxis)
    fig = plt.figure()
    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)

    # lakeswehavedatafor = [698]
    lakeswehavedatafor = [698, 67035, 19167, 31895, 310, 32276, 99045, 99516, 6950]

    for lakenum in np.arange(1, nlakes):
        lake_id, subid, name, eh, area, depth, longitude, latitude \
            = lakes[lakenum].strip().split(',')
        if int(lake_id)in lakeswehavedatafor:
            t_0_2 = 0
            t_25 = 0
            t_05 = 0
            t_20 = 0
            t_60 = 0
            td_02 = 0
            td_25, td_01, td_20, td_60 = 0, 0, 0, 0
            # getOutputPathFromEbHex
            eh = eh[2:] if eh[:2] == '0x' else eh
            while len(eh)< 6:
                eh = '0' + eh
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]

            outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231\\Tzt.csv' %(m1, exA, exB, m2, y1A, y2B))
            print(outdir)
            with open(outdir, 'rU')as f:
                file = reader(f, delimiter=',')
                y1, y2, y25 = [], [], []
                tt = pd.date_range("2001-01-01", "2010-12-31", freq="D")
                if int(lake_id)== 698:
                    for row in file:
                        y1.append(float(row[0]))
                        y2.append(float(row[1]))
                        y25.append(float(row[24]))
                    t_0 = pd.Series(y1, tt)
                    t_2 = pd.Series(y2, tt)
                    t_0_2 = t_0.append(t_2)
                    t_25 = pd.Series(y25, tt)
                if int(lake_id)== 67035:
                    for row in file:
                        y1.append(float(row[0]))
                        y2.append(float(row[19]))
                        y25.append(float(row[59]))
                    t_05 = pd.Series(y1, tt)
                    t_20 = pd.Series(y2, tt)
                    t_60 = pd.Series(y25, tt)

            filename = '../sweden_inflow_data/Validation_data_for_lookup.xlsx'
            # need to be change if the file change
            worksheet = pd.read_excel(filename, sheet_name=lake_id)

            test1 = worksheet.loc[worksheet['date'].str.contains('2001|2002|2003|2004|2005|2006|2007|2008|2010')]
            test1.loc[:, 'date'] = pd.to_datetime(test1.loc[:, 'date'], format="%Y-%m-%d")
            test1 = test1.sort_values(by='date')
            test1 = test1.dropna(subset=['Water temp(°C)'])
            test1.dropna(subset=['Water temp(°C)'])
            if lake_id == '698':
                test2 = test1.loc[test1['depth(max)'] <= 2]
                test3 = test1.loc[test1['depth(max)'] >= 35]

                list02 = test2['Water temp(°C)'].tolist()
                list25 = test3['Water temp(°C)'].tolist()
                td_02 = pd.Series(list02, test2['date'])
                td_25 = pd.Series(list25, test3['date'])
            if lake_id == '67035':
                test2 = test1.loc[test1['depth(max)'] <= 1]
                test3 = test1.loc[test1['depth(max)'] >= 20]
                test4 = test1.loc[test1['depth(max)'] >= 60]

                list02 = test2['Water temp(°C)'].tolist()
                list25 = test3['Water temp(°C)'].tolist()
                list5 = test4['Water temp(°C)'].tolist()
                td_01 = pd.Series(list02, test2['date'])
                td_20 = pd.Series(list25, test3['date'])
                td_60 = pd.Series(list5, test4['date'])

            plt.figure(1 * i, figsize=(18.0, 10.0))
            if int(lake_id)== 698:
                plt.subplot(211)
                t_0_2.plot(style=['k-'], markersize=3, label="1-2 m");
                plt.legend(loc='best')
                t_25.plot(style=['r-'], markersize=3, label="25 m");
                plt.legend(loc='best')
                td_02.plot(style=['go'], markersize=5, label="1-2 m data");
                plt.legend(loc='best')
                td_25.plot(style=['C4o'], markersize=5, label="25 m data");
                plt.legend(loc='best')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, ncol=5)
                frame1 = plt.gca()
                frame1.axes.get_xaxis().set_visible(False)
                plt.title('%s %s' %(lake_id, name))
            if lake_id == '67035':
                plt.subplot(212)
                t_05.plot(style=['k-'], markersize=3, label="0.5m");
                plt.legend(loc='best')
                t_20.plot(style=['r-'], markersize=3, label="20 m");
                plt.legend(loc='best')
                t_60.plot(style=['b-'], markersize=3, label="60 m");
                plt.legend(loc='best')
                td_01.plot(style=['go'], markersize=5, label="0.5 m data");
                plt.legend(loc='best')
                td_20.plot(style=['C4o'], markersize=5, label=">20 m data");
                plt.legend(loc='best')
                td_60.plot(style=['C1o'], markersize=5, label=">60 m data");
                plt.legend(loc='best')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)

                plt.title('%s %s' %(lake_id, name))
    fig1 = plt.figure(1 * i, figsize=(18.0, 10.0))
    fig1.savefig("figure/Figure 1_TEMP_swa_%s.png" %(k_BOD))
    plt.gcf().clear()
    # return plt.figure(1)

def FishNiche_graph_oxy_time(scenarioid, modelid, lakelistfile, swa_I_scDOC, k_BOD):
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2B = y1B + 4
    timeaxis = pd.date_range("%s-01-01" % y1A, "%s-12-31" % y2B, freq="D")
    fig1, fig2, fig3, fig4, fig5 = plt.figure(), plt.figure(), plt.figure(), plt.figure(), plt.figure()
    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)

    # lakeswehavedatafor = [698]
    lakeswehavedatafor = [698, 67035, 19167, 31895, 310, 32276, 99045, 99516, 6950]
    # lakeswehavedatafor = [698, 67035, 31895, 310, 32276]
    j = 0
    for lakenum in np.arange(1, nlakes):
        t_1, t_2, t_3, td_02, td_03, td_04 = 0, 0, 0, 0, 0, 0
        lake_id, subid, name, eh, area, depth, longitude, latitude \
            = lakes[lakenum].strip().split(',')
        lake_pos = 0
        if int(lake_id)in lakeswehavedatafor:
            level1 = 1
            level3 = 0
            if int(float(depth))< 10:
                level1 = 1
                level2 = int(depth)- 3
            elif int(float(depth))< 20:
                level2 = 16

            elif int(float(depth))< 35:
                level2 = 20
                level4 = ceil(level1 +((level2 - level1)/ 2))
            elif int(float(depth))< 50:
                level2 = 30
                level4 = ceil(level1 +((level2 - level1)/ 2))
            elif int(float(depth))>= 100:
                level2 = 50
                level3 = 75
                level4 = ceil(level1 +((level2 - level1)/ 2))
            else:
                level2 = 25
                level4 = ceil(level1 +((level2 - level1)/ 2))
            lake_pos += 1
            # getOutputPathFromEbHex
            eh = eh[2:] if eh[:2] == '0x' else eh
            while len(eh)< 6:
                eh = '0' + eh
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]

            outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231\\O2zt.csv' %(m1, exA, exB, m2, y1A, y2B))

            print(outdir)
            with open(outdir, 'rU')as f:
                file = reader(f, delimiter=',')
                y1, y2, y3, y25 = [], [], [], []
                tt = pd.date_range("2001-01-01", "2010-12-31", freq="D")

                for row in file:
                    y1.append(float(row[int(level1)]))
                    y2.append(float(row[int(level2)]))

                    if float(depth)>= 110:
                        y3.append(float(row[int(level3)]))

            t_1 = pd.Series(y1, tt)
            t_2 = pd.Series(y2, tt)

            if float(depth)>= 110:
                t_3 = pd.Series(y3, tt)

            filename = '../sweden_inflow_data/Validation_data_for_lookup.xlsx'
            worksheet = pd.read_excel(filename, sheet_name=lake_id)
            test1 = worksheet.loc[worksheet['date'].str.contains('2001|2002|2003|2004|2005|2006|2007|2008|2010')]
            test1.loc[:, 'date'] = pd.to_datetime(test1.loc[:, 'date'], format="%Y-%m-%d")
            test1 = test1.sort_values(by='date')
            test1 = test1.dropna(subset=['O2(mg/l)'])
            test1.dropna(subset=['O2(mg/l)'])
            if test1.empty != True:
                test2 = test1.loc[test1['depth(max)'] <= level1]
                test3 = test1.loc[test1['depth(max)'] >= level2]

                if float(depth)> 110:
                    test4 = test1.loc[test1['depth(max)'] >= level3]
                    list4 = test4['O2(mg/l)'].tolist()
                    list4 = [i * 1000 for i in list4]
                    td_04 = pd.Series(list4, test4['date'])
                list02 = test2['O2(mg/l)'].tolist()
                list02 = [i * 1000 for i in list02]
                list25 = test3['O2(mg/l)'].tolist()
                list25 = [i * 1000 for i in list25]
                td_02 = pd.Series(list02, test2['date'])
                td_03 = pd.Series(list25, test3['date'])

                if td_02.empty != True and td_03.empty != True:
                    if int(lake_id)== 698:
                        fig1 = plt.figure(5 * k_BOD, figsize=(18.0, 10.0))
                        ax1 = plt.subplot(211)
                        ax1.plot_date(tt, y1, 'ko-', lw=2, ms=3)
                        td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        ax1.xaxis.set_major_locator(years)
                        ax1.xaxis.set_major_formatter(yearsFmt)
                        ax1.xaxis.set_minor_locator(months)
                        ax1.autoscale_view()

                        ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax1.fmt_ydata = price
                        ax1.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax1.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax1.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig1.autofmt_xdate()
                        ax1.tick_params(axis='y', labelsize=16)
                        ax1.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)
                        ax1.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        plt.title('%s %s' %(lake_id, name))

                        ax2 = plt.subplot(212)
                        ax2.plot_date(tt, y2, 'k-', lw=2, ms=3)
                        td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        ax2.xaxis.set_major_locator(years)
                        ax2.xaxis.set_major_formatter(yearsFmt)
                        ax2.xaxis.set_minor_locator(months)
                        ax2.autoscale_view()

                        ax2.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax2.fmt_ydata = price
                        ax2.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax2.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax2.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig1.autofmt_xdate()
                        ax2.tick_params(axis='y', labelsize=16)
                        ax2.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)
                        plt.xlabel("Date")
                        ax2.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        fig1.text(0.02, 0.5, "Dissolved Oxygen", va='center', rotation='vertical', fontsize=16)

                        plt.tight_layout(rect=[0.03, 0.0, 1, 1])
                        # plt.subplot(211)
                        # t_1.plot(style=['k-'], markersize=3, label="%s m" % int(level1))
                        # plt.legend(loc='best')
                        # td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        # plt.legend(loc='best')
                        # plt.title('%s %s' %(lake_id, name))

                        # plt.subplot(212)

                        # t_2.plot(style=['k-'], markersize=3, label="%s m" %(int(level2)))
                        # plt.legend(loc='best')
                        # td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        # plt.legend(loc='best')

                        # fig1 = plt.figure(10*k_BOD,figsize=(18.0, 10.0))


                    elif int(lake_id)== 67035:
                        fig2 = plt.figure(5 * k_BOD, figsize=(18.0, 10.0))
                        ax1 = plt.subplot(211)
                        ax1.plot_date(tt, y1, 'ko-', lw=2, ms=3)
                        td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        ax1.xaxis.set_major_locator(years)
                        ax1.xaxis.set_major_formatter(yearsFmt)
                        ax1.xaxis.set_minor_locator(months)
                        ax1.autoscale_view()

                        ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax1.fmt_ydata = price
                        ax1.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax1.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax1.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig2.autofmt_xdate()
                        ax1.tick_params(axis='y', labelsize=16)
                        ax1.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)
                        ax1.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        plt.title('%s %s' %(lake_id, name))

                        ax2 = plt.subplot(212)
                        ax2.plot_date(tt, y2, 'k-', lw=2, ms=3)
                        td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        ax2.xaxis.set_major_locator(years)
                        ax2.xaxis.set_major_formatter(yearsFmt)
                        ax2.xaxis.set_minor_locator(months)
                        ax2.autoscale_view()

                        ax2.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax2.fmt_ydata = price
                        ax2.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax2.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax2.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig2.autofmt_xdate()
                        ax2.tick_params(axis='y', labelsize=16)
                        ax2.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)
                        plt.xlabel("Date")
                        ax2.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        fig2.text(0.02, 0.5, "Dissolved Oxygen", va='center', rotation='vertical', fontsize=16)

                        plt.tight_layout(rect=[0.03, 0.0, 1, 1])
                        # plt.subplot(211)
                        # t_1.plot(style=['k-'], markersize=3, label="%s m" % int(level1))
                        # plt.legend(loc='best')
                        # td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        # plt.legend(loc='best')
                        # plt.title('%s %s' %(lake_id, name))

                        # plt.subplot(212)

                        # t_2.plot(style=['k-'], markersize=3, label="%s m" %(int(level2)))
                        # plt.legend(loc='best')
                        # t_3.plot(style=['k:'], markersize=3, label="%s m" %(int(level3)))
                        # plt.legend(loc='best')
                        # td_03.plot(style=['ro'], markersize=5, label="%s m data" %(int(level2)))
                        # plt.legend(loc='best')
                        # td_04.plot(style=['C2o'], markersize=5, label="%s-%s m data" %(int(level3), depth))
                        # plt.legend(loc='best')
                        # fig2 = plt.figure(2*k_BOD,figsize=(18.0, 10.0))


                    elif int(lake_id)== 31895:
                        fig3 = plt.figure(5 * k_BOD, figsize=(18.0, 10.0))
                        ax1 = plt.subplot(211)
                        ax1.plot_date(tt, y1, 'ko-', lw=2, ms=3)
                        td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        ax1.xaxis.set_major_locator(years)
                        ax1.xaxis.set_major_formatter(yearsFmt)
                        ax1.xaxis.set_minor_locator(months)
                        ax1.autoscale_view()

                        ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax1.fmt_ydata = price
                        ax1.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax1.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax1.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig3.autofmt_xdate()
                        ax1.tick_params(axis='y', labelsize=16)
                        ax1.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)
                        ax1.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        plt.title('%s %s' %(lake_id, name))

                        ax2 = plt.subplot(212)
                        ax2.plot_date(tt, y2, 'k-', lw=2, ms=3)
                        td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        ax2.xaxis.set_major_locator(years)
                        ax2.xaxis.set_major_formatter(yearsFmt)
                        ax2.xaxis.set_minor_locator(months)
                        ax2.autoscale_view()

                        ax2.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax2.fmt_ydata = price
                        ax2.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax2.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax2.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig3.autofmt_xdate()
                        ax2.tick_params(axis='y', labelsize=16)
                        ax2.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)
                        plt.xlabel("Date")
                        ax2.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        fig3.text(0.02, 0.5, "Dissolved Oxygen", va='center', rotation='vertical', fontsize=16)

                        plt.tight_layout(rect=[0.03, 0.0, 1, 1])
                        # plt.subplot(211)
                        # t_1.plot(style=['k-'], markersize=3, label="%s m" % int(level1))
                        # plt.legend(loc='best')
                        # td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        # plt.legend(loc='best')
                        # plt.title('%s %s' %(lake_id, name))

                        # plt.subplot(212)

                        # t_2.plot(style=['k-'], markersize=3, label="%s m" %(int(level2)))
                        # plt.legend(loc='best')
                        # td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        # plt.legend(loc='best')

                        # fig3 = plt.figure(3*k_BOD,figsize=(18.0, 10.0))


                    elif int(lake_id)== 32276:
                        fig4 = plt.figure(4 * k_BOD, figsize=(18.0, 10.0))
                        ax1 = plt.subplot(211)
                        ax1.plot_date(tt, y1, 'ko-', lw=2, ms=3)
                        td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        ax1.plot_date(tt, y2, '-', color='0.5', lw=2, ms=3)
                        td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        ax1.xaxis.set_major_locator(years)
                        ax1.xaxis.set_major_formatter(yearsFmt)
                        ax1.xaxis.set_minor_locator(months)
                        ax1.autoscale_view()

                        ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax1.fmt_ydata = price
                        ax1.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax1.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax1.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig4.autofmt_xdate()
                        ax1.tick_params(axis='y', labelsize=16)
                        ax1.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)

                        plt.ylabel("Temperature")
                        plt.xlabel("Date")

                        ax1.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        plt.title('%s %s' %(lake_id, name))
                        plt.tight_layout()
                        # plt.subplot(211)
                        # t_1.plot(style=['k-'], markersize=3, label="%s m" % int(level1))
                        # plt.legend(loc='best')
                        # td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        # plt.legend(loc='best')
                        # plt.title('%s %s' %(lake_id, name))
                        # plt.subplot(212)

                        # t_2.plot(style=['k-'], markersize=3, label="%s m" %(int(level2)))
                        # plt.legend(loc='best')
                        # td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        # plt.legend(loc='best')

                        # fig4 = plt.figure(4*k_BOD,figsize=(18.0, 10.0))

                    elif int(lake_id)== 310:
                        fig5 = plt.figure(5 * k_BOD, figsize=(18.0, 10.0))
                        ax1 = plt.subplot(211)
                        ax1.plot_date(tt, y1, 'ko-', lw=2, ms=3)
                        td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        ax1.xaxis.set_major_locator(years)
                        ax1.xaxis.set_major_formatter(yearsFmt)
                        ax1.xaxis.set_minor_locator(months)
                        ax1.autoscale_view()

                        ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax1.fmt_ydata = price
                        ax1.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax1.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax1.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig5.autofmt_xdate()
                        ax1.tick_params(axis='y', labelsize=16)
                        ax1.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)
                        ax1.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        plt.title('%s %s' %(lake_id, name))

                        ax2 = plt.subplot(212)
                        ax2.plot_date(tt, y2, 'k-', lw=2, ms=3)
                        td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        ax2.xaxis.set_major_locator(years)
                        ax2.xaxis.set_major_formatter(yearsFmt)
                        ax2.xaxis.set_minor_locator(months)
                        ax2.autoscale_view()

                        ax2.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax2.fmt_ydata = price
                        ax2.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax2.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax2.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig5.autofmt_xdate()
                        ax2.tick_params(axis='y', labelsize=16)
                        ax2.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)
                        plt.xlabel("Date")
                        ax2.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        fig5.text(0.02, 0.5, "Dissolved Oxygen", va='center', rotation='vertical', fontsize=16)

                        plt.tight_layout(rect=[0.03, 0.0, 1, 1])
                        # t_1.plot(style=['k-'], markersize=3, label="%s m" % int(level1))
                        # plt.legend(loc='best')
                        # td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        # plt.legend(loc='best')
                        # plt.title('%s %s' %(lake_id, name))

                        # plt.subplot(212)
                        # t_2.plot(style=['k-'], markersize=3, label="%s m" %(level2))
                        # plt.legend(loc='best')
                        # td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        # plt.legend(loc='best')
                        # fig5 = plt.figure(5*k_BOD,figsize=(18.0, 10.0))

            if int(lake_id)== 698:
                fig1.savefig("Figure 1_OXYGEN_%s_%s_%s.png" %(lake_id, name, swa_I_scDOC))
            elif int(lake_id)== 67035:
                fig2.savefig("Figure 1_OXYGEN_%s_%s_%s.png" %(lake_id, name, swa_I_scDOC))
            elif int(lake_id)== 31895:
                fig3.savefig("Figure 1_OXYGEN_%s_%s_%s.png" %(lake_id, name, swa_I_scDOC))
            elif int(lake_id)== 32276:
                fig4.savefig("Figure 1_OXYGEN_%s_%s_%s.png" %(lake_id, name, swa_I_scDOC))
            elif int(lake_id)== 310:
                fig5.savefig("Figure 1_OXYGEN_%s_%s_%s.png" %(lake_id, name, swa_I_scDOC))
        plt.gcf().clear()
        # return fig1,fig2,fig3,fig4,fig5

def FishNiche_graph_temp_complete_time(scenarioid, modelid, lakelistfile, swa_I_scDOC, k_SOD):
    years = YearLocator() # every year
    months = MonthLocator() # every month
    yearsFmt = DateFormatter('%Y')
    monthsFmt = DateFormatter('%M')
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2B = y1B + 4
    timeaxis = pd.date_range("%s-01-01" % y1A, "%s-12-31" % y2B, freq="D")
    fig1, fig2, fig3, fig4, fig5 = plt.figure(), plt.figure(), plt.figure(), plt.figure(), plt.figure()
    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)

    # lakeswehavedatafor = [698]
    lakeswehavedatafor = [698, 67035, 19167, 31895, 310, 32276, 99045, 99516, 6950]
    # lakeswehavedatafor = [698, 67035, 31895, 310, 32276]
    j = 0
    for lakenum in np.arange(1, nlakes):
        t_1, t_2, t_3, td_02, td_03, td_04 = 0, 0, 0, 0, 0, 0
        lake_id, subid, name, eh, area, depth, longitude, latitude \
            = lakes[lakenum].strip().split(',')
        lake_pos = 0
        if int(lake_id)in lakeswehavedatafor:
            level1 = 1
            level3 = 0
            if int(float(depth))< 10:
                level1 = 1
                level2 = int(depth)- 3
            elif int(float(depth))< 20:
                level2 = 16

            elif int(float(depth))< 35:
                level2 = 20
                level4 = ceil(level1 +((level2 - level1)/ 2))
            elif int(float(depth))< 50:
                level2 = 30
                level4 = ceil(level1 +((level2 - level1)/ 2))
            elif int(float(depth))>= 100:
                level2 = 50
                level3 = 75
                level4 = ceil(level1 +((level2 - level1)/ 2))
            else:
                level2 = 25
                level4 = ceil(level1 +((level2 - level1)/ 2))
            lake_pos += 1
            # getOutputPathFromEbHex
            eh = eh[2:] if eh[:2] == '0x' else eh
            while len(eh)< 6:
                eh = '0' + eh
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]

            outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231\\Tzt.csv' %(m1, exA, exB, m2, y1A, y2B))

            print(outdir)
            with open(outdir, 'rU')as f:
                file = reader(f, delimiter=',')
                y1, y2, y3, y25 = [], [], [], []
                tt = pd.date_range("2001-01-01", "2010-12-31", freq="D")

                for row in file:
                    y1.append(float(row[int(level1)]))
                    y2.append(float(row[int(level2)]))

                    if float(depth)>= 110:
                        y3.append(float(row[int(level3)]))

            t_1 = pd.Series(y1, tt)
            t_2 = pd.Series(y2, tt)

            if float(depth)>= 110:
                t_3 = pd.Series(y3, tt)

            filename = '../sweden_inflow_data/Validation_data_for_lookup.xlsx'
            worksheet = pd.read_excel(filename, sheet_name=lake_id)
            test1 = worksheet.loc[worksheet['date'].str.contains('2001|2002|2003|2004|2005|2006|2007|2008|2010')]
            test1.loc[:, 'date'] = pd.to_datetime(test1.loc[:, 'date'], format="%Y-%m-%d")
            test1 = test1.sort_values(by='date')
            test1 = test1.dropna(subset=['Water temp(°C)'])
            test1.dropna(subset=['Water temp(°C)'])
            if test1.empty != True:
                test2 = test1.loc[test1['depth(max)'] <= level1]
                test3 = test1.loc[test1['depth(max)'] >= level2]

                if float(depth)> 110:
                    test4 = test1.loc[test1['depth(max)'] >= level3]
                    list4 = test4['Water temp(°C)'].tolist()
                    list4 = [i for i in list4]
                    td_04 = pd.Series(list4, test4['date'])
                list02 = test2['Water temp(°C)'].tolist()
                list02 = [i for i in list02]
                list25 = test3['Water temp(°C)'].tolist()
                list25 = [i for i in list25]
                td_02 = pd.Series(list02, test2['date'])
                td_03 = pd.Series(list25, test3['date'])

                if td_02.empty != True and td_03.empty != True:
                    if int(lake_id)== 698:
                        fig1 = plt.figure(10 * k_SOD, figsize=(18.0, 10.0))
                        ax = plt.subplot(211)
                        ax.plot_date(tt, y1, 'ko-', lw=2, ms=3)
                        td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        ax.plot_date(tt, y2, 'o-', color='0.5', lw=2, ms=3)
                        td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))

                        ax.xaxis.set_major_locator(years)
                        ax.xaxis.set_major_formatter(yearsFmt)
                        ax.xaxis.set_minor_locator(months)
                        ax.autoscale_view()

                        ax.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax.fmt_ydata = price
                        ax.yaxis.grid(True)
                        # ax.fill_between(tt, 0, 1, where=tt > theta,
                        #                  facecolor='green', alpha=0.5, transform=trans)
                        # ax2 = plt.subplot(212)
                        # ax2.plot_date(tt, y2, 'o-', color='0.5', lw=2, ms=3)
                        # td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))

                        # ax2.xaxis.set_major_locator(years)
                        # ax2.xaxis.set_major_formatter(yearsFmt)
                        # ax2.xaxis.set_minor_locator(months)
                        # ax2.autoscale_view()
                        # ax2.fmt_xdata = DateFormatter('%Y-%m-%d')
                        # ax2.fmt_ydata = price
                        # ax2.grid(True)


                        # period = tt[(tt.month>10)or(tt.month<4)]
                        # ax.axvspan(min(period)- MonthBegin(), max(period)+ MonthEnd(),
                        #             facecolor='g', edgecolor='none', alpha=.2)
                        # for i in [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]:

                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                           (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            print(winter_months[i + 1])
                            ax.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            print(summer_months[i + 1])
                            ax.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)



                            # plt.subplot(211)
                            # t_1.plot(style=['k-'], markersize=3, label="%s m" % int(level1))
                            # plt.legend(loc='best')
                            # td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                            # plt.legend(loc='best')
                        plt.title('%s %s' %(lake_id, name))
                        # ax.xaxis.set_major_locator(years)
                        # ax.xaxis.set_major_formatter(yearsFmt)
                        # ax.xaxis.set_minor_locator(months)
                        # ax.autoscale_view()

                        # ax.fmt_xdata = DateFormatter('%Y-%m-%d')
                        # ax.fmt_ydata = price
                        # ax.grid(True)

                        fig1.autofmt_xdate()
                        plt.tight_layout()
                        ax.tick_params(axis='y', labelsize=16)
                        ax.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)

                        plt.ylabel("Temperature")
                        plt.xlabel("Date")
                        plt.tight_layout()
                        ax.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        # plt.subplot(212)

                        # t_2.plot(style=['k-'], markersize=3, label="%s m" %(int(level2)))
                        # plt.legend(loc='best')
                        # td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        # plt.legend(loc='best')

                        # fig1 = plt.figure(10*k_SOD,figsize=(18.0, 10.0))


                    elif int(lake_id)== 67035:
                        fig2 = plt.figure(2 * k_SOD, figsize=(18.0, 10.0))
                        ax1 = plt.subplot(211)
                        ax1.plot_date(tt, y1, 'ko-', lw=2, ms=3)
                        td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        ax1.plot_date(tt, y2, '-', color='0.5', lw=2, ms=3)
                        td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        ax1.xaxis.set_major_locator(years)
                        ax1.xaxis.set_major_formatter(yearsFmt)
                        ax1.xaxis.set_minor_locator(months)
                        ax1.autoscale_view()

                        ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax1.fmt_ydata = price
                        ax1.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax1.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax1.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig2.autofmt_xdate()
                        ax1.tick_params(axis='y', labelsize=16)
                        ax1.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)

                        plt.ylabel("Temperature")
                        plt.xlabel("Date")

                        ax1.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        plt.title('%s %s' %(lake_id, name))
                        plt.tight_layout()
                        # ax2=plt.subplot(212)
                        # ax2.plot_date(tt, y2, 'k-', lw=2, ms=3)
                        # td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        # ax2.plot_date(tt, y3, 'o-', color='0.5', lw=2, ms=3)
                        # td_04.plot(style=['C2o'], markersize=5, label="%s-%s m data" %(int(level3), depth))
                        # ax2.xaxis.set_major_locator(years)
                        # ax2.xaxis.set_major_formatter(yearsFmt)
                        # ax2.xaxis.set_minor_locator(months)
                        # ax2.autoscale_view()

                        # ax2.fmt_xdata = DateFormatter('%Y-%m-%d')
                        # ax2.fmt_ydata = price
                        # ax2.yaxis.grid(True)
                        # winter_months = tt[
                        #   ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                        #    |((tt.month == 3)&(tt.day == 20))|(
                        #   (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        # summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        # for i in np.arange(0, len(winter_months), 2):
                        #    print(winter_months[i + 1])
                        #    ax2.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        # for i in np.arange(0, len(summer_months), 2):
                        #    print(summer_months[i + 1])
                        #    ax2.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        # plt.tight_layout()
                        # ax2.tick_params(axis='y', labelsize=16)
                        # ax2.tick_params(axis='x', labelsize=16)
                        # plt.legend(loc="lower left", ncol=2)

                        # plt.ylabel("Temperature")
                        # plt.xlabel("Date")
                        # ax2.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))

                        # plt.legend(loc='best')


                    elif int(lake_id)== 31895:
                        fig3 = plt.figure(3 * k_SOD, figsize=(18.0, 10.0))
                        ax1 = plt.subplot(211)
                        ax1.plot_date(tt, y1, 'ko-', lw=2, ms=3)
                        td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        ax1.plot_date(tt, y2, '-', color='0.5', lw=2, ms=3)
                        td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        ax1.xaxis.set_major_locator(years)
                        ax1.xaxis.set_major_formatter(yearsFmt)
                        ax1.xaxis.set_minor_locator(months)
                        ax1.autoscale_view()

                        ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax1.fmt_ydata = price
                        ax1.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax1.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax1.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig3.autofmt_xdate()
                        ax1.tick_params(axis='y', labelsize=16)
                        ax1.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)

                        plt.ylabel("Temperature")
                        plt.xlabel("Date")

                        ax1.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        plt.title('%s %s' %(lake_id, name))
                        plt.tight_layout()
                        # plt.subplot(211)
                        # t_1.plot(style=['k-'], markersize=3, label="%s m" % int(level1))
                        # plt.legend(loc='best')
                        # td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        # plt.legend(loc='best')
                        # plt.title('%s %s' %(lake_id, name))

                        # plt.subplot(212)

                        # t_2.plot(style=['k-'], markersize=3, label="%s m" %(int(level2)))
                        # plt.legend(loc='best')
                        # td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        # plt.legend(loc='best')

                        # fig3 = plt.figure(3*k_SOD,figsize=(18.0, 10.0))


                    elif int(lake_id)== 32276:
                        fig4 = plt.figure(4 * k_SOD, figsize=(18.0, 10.0))
                        ax1 = plt.subplot(211)
                        ax1.plot_date(tt, y1, 'ko-', lw=2, ms=3)
                        td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        ax1.plot_date(tt, y2, '-', color='0.5', lw=2, ms=3)
                        td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        ax1.xaxis.set_major_locator(years)
                        ax1.xaxis.set_major_formatter(yearsFmt)
                        ax1.xaxis.set_minor_locator(months)
                        ax1.autoscale_view()

                        ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax1.fmt_ydata = price
                        ax1.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax1.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax1.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig4.autofmt_xdate()
                        ax1.tick_params(axis='y', labelsize=16)
                        ax1.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)

                        plt.ylabel("Temperature")
                        plt.xlabel("Date")

                        ax1.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        plt.title('%s %s' %(lake_id, name))
                        plt.tight_layout()
                        # plt.subplot(211)
                        # t_1.plot(style=['k-'], markersize=3, label="%s m" % int(level1))
                        # plt.legend(loc='best')
                        # td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        # plt.legend(loc='best')
                        # plt.title('%s %s' %(lake_id, name))
                        # frame1 = plt.gca()
                        # frame1.axes.get_xaxis().set_visible(False)
                        # plt.subplot(212)
                        # t_5.plot(style=['k-'], markersize=3, label="%s-%s m" %(int(level1), int(level2)))
                        # plt.legend(loc='best')
                        # td_05.plot(style=['ro'], markersize=5,
                        #             label="%s-%s m data" %(int(level1), int(level2)))
                        # plt.legend(loc='best')
                        # frame1 = plt.gca()
                        # frame1.axes.get_xaxis().set_visible(False)
                        # plt.subplot(313)
                        # t_2.plot(style=['k-'], markersize=3, label="%s m" %(int(level2)))
                        # plt.legend(loc='best')
                        # td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        # plt.legend(loc='best')

                        # fig4 = plt.figure(4*k_SOD,figsize=(18.0, 10.0))

                    elif int(lake_id)== 310:
                        fig5 = plt.figure(5 * k_SOD, figsize=(18.0, 10.0))
                        ax1 = plt.subplot(211)
                        ax1.plot_date(tt, y1, 'ko-', lw=2, ms=3)
                        td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        ax1.plot_date(tt, y2, '-', color='0.5', lw=2, ms=3)
                        td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        ax1.xaxis.set_major_locator(years)
                        ax1.xaxis.set_major_formatter(yearsFmt)
                        ax1.xaxis.set_minor_locator(months)
                        ax1.autoscale_view()

                        ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
                        ax1.fmt_ydata = price
                        ax1.yaxis.grid(True)
                        winter_months = tt[
                           ((tt.year == 2001)&(tt.month == 1)&(tt.day == 1))|((tt.month == 12)&(tt.day == 21))
                            |((tt.month == 3)&(tt.day == 20))|(
                               (tt.year == 2010)&(tt.month == 12)&(tt.day == 31))]
                        summer_months = tt[((tt.month == 6)&(tt.day == 21))|((tt.month == 9)&(tt.day == 22))]
                        # Loop through groups and add axvspan between winter month
                        for i in np.arange(0, len(winter_months), 2):
                            ax1.axvspan(winter_months[i], winter_months[i + 1], facecolor='lavender', alpha=0.5)
                        for i in np.arange(0, len(summer_months), 2):
                            ax1.axvspan(summer_months[i], summer_months[i + 1], facecolor='wheat', alpha=0.5)
                        fig5.autofmt_xdate()
                        ax1.tick_params(axis='y', labelsize=16)
                        ax1.tick_params(axis='x', labelsize=16)
                        plt.legend(loc="lower left", ncol=2)

                        plt.ylabel("Temperature")
                        plt.xlabel("Date")

                        ax1.set_xlim(pd.Timestamp("2000-12-01"), pd.Timestamp("2011-02-01"))
                        plt.legend(loc='best')
                        plt.title('%s %s' %(lake_id, name))
                        plt.tight_layout()
                        # plt.subplot(211)
                        # t_1.plot(style=['k-'], markersize=3, label="%s m" % int(level1))
                        # plt.legend(loc='best')
                        # td_02.plot(style=['bo'], markersize=5, label="0-%s m data" % int(level1))
                        # plt.legend(loc='best')
                        # plt.title('%s %s' %(lake_id, name))

                        # plt.subplot(212)

                        # t_2.plot(style=['k-'], markersize=3, label="%s m" %(level2))
                        # plt.legend(loc='best')
                        # td_03.plot(style=['ro'], markersize=5, label="%s-%s m data" %(int(level2), depth))
                        # plt.legend(loc='best')
                        # fig5 = plt.figure(5*k_SOD,figsize=(18.0, 10.0))

            if int(lake_id)== 698:
                fig1.savefig("Figure 1_TEMPERATURE_%s_%s_%s.png" %(lake_id, name, swa_I_scDOC))
            elif int(lake_id)== 67035:
                fig2.savefig("Figure 1_TEMPERATURE_%s_%s_%s.png" %(lake_id, name, swa_I_scDOC))
            elif int(lake_id)== 31895:
                fig3.savefig("Figure 1_TEMPERATURE_%s_%s_%s.png" %(lake_id, name, swa_I_scDOC))
            elif int(lake_id)== 32276:
                fig4.savefig("Figure 1_TEMPERATURE_%s_%s_%s.png" %(lake_id, name, swa_I_scDOC))
            elif int(lake_id)== 310:
                fig5.savefig("Figure 1_TEMPERATURE_%s_%s_%s.png" %(lake_id, name, swa_I_scDOC))
        plt.gcf().clear()
        # return fig1,fig2,fig3,fig4,fig5

def FishNiche_secchi_graph(scenarioid, modelid, lakelistfile, calivari, k_BOD):
    label_secchi, label_o = '', ''
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2B = y1B + 4
    timeaxis = pd.date_range("%s-01-01" % y1A, "%s-12-31" % y2B, freq="D")
    timestep = len(timeaxis)
    plt.gcf().clear()
    fig1, fig2, fig3, fig4 = plt.figure(), plt.figure(), plt.figure(), plt.figure()

    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)

    # lakeswehavedatafor = [698]
    lakeswehavedatafor = [32276, 310, 14939, 30704, 31895, 6950, 99045, 33590, 33494, 16765, 698, 67035]
    # lakeswehavedatafor = [6950, 67035, 31895, 310, 32276, 99045]
    meansamples=[]
    for lakenum in np.arange(1, nlakes):
        lake_id, subid, name, eh, area, depth, longitude, latitude, volume, mean_depth, sediment = lakes[lakenum].strip().split(',')
        
        if int(lake_id)in lakeswehavedatafor:

            # getOutputPathFromEbHex
            if eh[:2] == '0x':
                eh = eh[2:]
            else:
                eh = eh
            while len(eh)< 6:
                eh = '0' + eh
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]
            outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(
                                     m1, exA, exB, m2, y1A, y2B))
            print(outdir)

            O2model = pd.read_csv(path.join(outdir, 'O2zt.csv'), header=None)
            lambdamodel = pd.read_csv(path.join(outdir, 'lambdazt.csv'), header=None)

            filename = '../sweden_inflow_data/Validation_data_for_lookup.xlsx'

            # need to be change if the file change
            worksheet = pd.read_excel(filename, sheet_name=lake_id)

            dates = worksheet['date']
            depthdata = worksheet['depth(max)']
            O2raw = worksheet['O2(mg/l)'] * 1000
            secchiraw = worksheet['Siktdjup (m)']

            anydata_secchi, anydata_o = False, False
            O2_model_samples, O2_data_samples, O2_depths = [], [], []
            secchi_model_samples, secchi_data_samples = [], []
            crashed = False

            try:
                for ii in np.arange(0, len(dates)):
                    dnum = datetime.strptime(dates[ii], "%Y-%m-%d").date()
                    start = timeaxis[0].date()
                    dateindex =(dnum - start).days
                    if dateindex >= 1 and dateindex <= timestep:
                        secchi_data = secchiraw[ii]
                        secchi_calculated = []
                        for i in [1,2,3,4]:
                            secchi_calculated.append((1.48 /(np.mean(lambdamodel.loc[dateindex, :]))))
                        secchi_model = secchi_calculated

                        if not isnan(secchi_data):
                            secchi_data_samples.append(secchi_data)
                            secchi_model_samples.append(secchi_model)
                            anydata_secchi = True

                for ii in np.arange(0, len(dates)):
                    dnum = datetime.strptime(dates[ii], "%Y-%m-%d").date()
                    start = timeaxis[0].date()
                    dateindex =(dnum - start).days
                    if dateindex >= 1 and dateindex <= timestep:
                        depthO = ceil(depthdata[ii])
                        O2_data = O2raw[ii]
                        O2_model = O2model.loc[dateindex, depthO - 1]

                        if not isnan(O2_data):
                            O2_data_samples.append(O2_data)
                            O2_model_samples.append(O2_model)
                            O2_depths.append(depthO)
                            anydata_o = True


            except:
                crashed = True
                print("Error when handling lake %s %s \n" %(name, lake_id))
            label_secchi = "%s %s" %(name, lake_id)
            label_o = "%s %s" %(name, lake_id)
            if crashed:
                label_secchi =("%s - Error in program,unreliable" % label_secchi)
                label_o =("%s - Error in program,unreliable" % label_o)
            else:
                if not anydata_secchi:
                    label_secchi = "%s(No data in timespan)" % label_secchi
                else:
                    RMSE_secchi, corr_secchi = [], []
                    for i in np.arange(0, len(secchi_model_samples[0])):
                        RMSE_secchi.append(round(sqrt(sklearn.metrics.mean_squared_error(secchi_data_samples,
                                                                                                 [item[i] for item in
                                                                                                  secchi_model_samples])
                                                           ), 2))
                        corr_secchi.append(
                            scipy.stats.pearsonr(secchi_data_samples, [item[i] for item in secchi_model_samples]))
                    personsecchi = '1.4: %.3f 1.5: %.3f 1.6: %.3f 1.7: %.3f' %(round(corr_secchi[0][0], 3),
                                                                                round(corr_secchi[1][0], 3),
                                                                                round(corr_secchi[2][0], 3),
                                                                                round(corr_secchi[3][0], 3))
                    RMSEsecchi = '1.4: %.3f 1.5: %.3f 1.6: %.3f 1.7: %.3f' %(
                        round(RMSE_secchi[0], 3), round(RMSE_secchi[1], 3), round(RMSE_secchi[2], 3),
                        round(RMSE_secchi[3], 3))
                    label_secchi = "\n%s,RMSE: %s\n Pearson: %s" %(label_secchi, RMSEsecchi, personsecchi)
                if not anydata_o:
                    label_o = "%s(No data in timespan)" % label_o
                else:
                    RMSE_O = round(sqrt(sklearn.metrics.mean_squared_error(O2_data_samples, O2_model_samples)),
                                     2)
                    corr_O = scipy.stats.pearsonr(O2_data_samples, O2_model_samples)
                    pearsono = '%.3f' %(round(corr_O[0], 3))

                    label_o = "%s,RMSE: %s\n Pearson: %s" %(label_o, RMSE_O, pearsono)

            lineStart = 0
            lineEnd = 20
            rows = ceil(sqrt(nlakes - 1))
            cols = ceil((nlakes - 1)/ rows)
            rows1 = 4
            cols1 = 1
            variable = [1.4, 1.5, 1.6, 1.7]

            if secchi_model_samples != []:
                meansample = st.mean(secchi_data_samples)
                fig1 = plt.figure(k_BOD * 10, figsize=(18.0, 10.0))
                for i in np.arange(1, len(secchi_model_samples[0]), 1):
                    model_data = [item[i] for item in secchi_model_samples]
                    plt.subplot(int('%s%s%s' %(rows1, cols1, i)))
                    plt.scatter(secchi_data_samples, model_data, label=lake_id, s=30, edgecolors='k')
                    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color='r')
                    plt.title(label_secchi, fontsize=10)
                    plt.xlabel("data_samples", fontsize=14)
                    plt.ylabel("model_samples", fontsize=14)
                    plt.subplots_adjust(hspace=0.6, wspace=0.3)
                    plt.tick_params(axis='both', which='major', labelsize=14)
                plt.legend()

                fig2 = plt.figure((k_BOD + 1)* 10, figsize=(18.0, 10.0))
                plt.subplot(rows, cols, lakenum)
                plt.scatter(secchi_data_samples, [item[0] for item in secchi_model_samples], label='1.4', s=20)
                plt.scatter(secchi_data_samples, [item[1] for item in secchi_model_samples], label='1.5', s=20)
                plt.scatter(secchi_data_samples, [item[2] for item in secchi_model_samples], label='1.6', s=20)
                plt.scatter(secchi_data_samples, [item[3] for item in secchi_model_samples], label='1.7', s=20)
                plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color='r')
                plt.title(label_secchi, fontsize=10)
                # plt.xlabel("data_samples", fontsize=14)
                plt.ylabel("model_samples", fontsize=14)
                plt.subplots_adjust(hspace=0.6, wspace=0.3)
                plt.tick_params(axis='both', which='major', labelsize=14)
            else:
                fig1 = plt.figure(k_BOD * 10, figsize=(18.0, 10.0))
                for i in np.arange(1, len(variable)+ 1, 1):
                    plt.subplot(rows1, cols1, i)
                    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color='r')
                    plt.title(label_secchi, fontsize=10)
                    plt.xlabel("data_samples", fontsize=14)
                    plt.ylabel("model_samples", fontsize=14)
                    plt.subplots_adjust(hspace=0.6, wspace=0.3)
                    plt.tick_params(axis='both', which='major', labelsize=14)

                fig2 = plt.figure((k_BOD + 1)* 10, figsize=(18.0, 10.0))
                plt.subplot(rows, cols, lakenum)
                plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color='r')
                plt.title(label_secchi, fontsize=10)
                # plt.xlabel("data_samples", fontsize=14)
                plt.ylabel("model_samples", fontsize=14)
                plt.subplots_adjust(hspace=0.6, wspace=0.3)
                plt.tick_params(axis='both', which='major', labelsize=14)

            lineStart = 0
            lineEnd = 20000

            fig3 = plt.figure((k_BOD + 3)* 10, figsize=(18.0, 10.0))
            plt.scatter(O2_data_samples, O2_model_samples, label=label_o, s=30, edgecolors='k')
            plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color='r')
            plt.title(label_o, fontsize=10)
            plt.xlabel("data_samples", fontsize=14)
            plt.ylabel("model_samples", fontsize=14)
            plt.subplots_adjust(hspace=0.6, wspace=0.3)
            plt.tick_params(axis='both', which='major', labelsize=14)

            fig4 = plt.figure((k_BOD + 4)* 100, figsize=(18.0, 10.0))
            plt.subplot(rows, cols, lakenum)
            points = plt.scatter(O2_data_samples, O2_model_samples, c=O2_depths, edgecolors='k', label=label_o, s=20,
                                   cmap='bone_r')
            plt.colorbar(points)
            plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color='r')
            plt.title(label_o, fontsize=10)
            plt.xlabel("data_samples", fontsize=14)
            plt.ylabel("model_samples", fontsize=14)
            plt.subplots_adjust(hspace=0.6, wspace=0.3)
            plt.tick_params(axis='both', which='major', labelsize=14)
            meansamples.append(meansample)

    fig2.savefig(path.join(datafolder, "Figure 2 Secchi all_%s.png" %(calivari)))
    fig1.savefig(path.join(datafolder, "Figure 1 Secchi_%s.png" %(calivari)))
    plt.gcf().clear()
    # fig3.savefig("figure\\Figure 1 Oxygen_%s.png" %(calivari))
    # fig4.savefig("figure\\Figure 2 Oxygen all_%s.png" %(calivari))

def FishNiche_mean_secchi_graph(scenarioid, modelid, lakelistfile, calivari, variable_test):
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2B = y1B + 4
    timeaxis = pd.date_range("%s-01-01" % y1A, "%s-12-31" % y2B, freq="D")
    timestep = len(timeaxis)

    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)

    # lakeswehavedatafor = [698]
    lakeswehavedatafor = [67035, 16765, 33494, 33590, 99045, 6950, 31895, 30704, 14939, 310, 32276]
    all_mean_secchi_model, all_mean_secchi_data = [], []
    all_std_secchi_model, all_std_secchi_data = [], []

    for lakenum in np.arange(1, nlakes):
        lake_id, subid, name, eh, area, depth, longitude, latitude, volume \
            = lakes[lakenum].strip().split(',')
        if int(lake_id)in lakeswehavedatafor:

            # getOutputPathFromEbHex
            eh = eh[2:] if eh[:2] == '0x' else eh
            while len(eh)< 6:
                eh = '0' + eh
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]

            outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
            print(outdir)

            O2model = pd.read_csv(path.join(outdir, 'O2zt.csv'), header=None)
            lambdamodel = pd.read_csv(path.join(outdir, 'lambdazt.csv'), header=None)

            filename = '../sweden_inflow_data/Validation_data_for_lookup.xlsx'

            # need to be change if the file change
            worksheet = pd.read_excel(filename, sheet_name=lake_id)

            dates = worksheet['date']
            depthdata = worksheet['depth(max)']
            O2raw = worksheet['O2(mg/l)'] * 1000
            secchiraw = worksheet['Siktdjup(m)']

            anydata_secchi, anydata_o = False, False
            O2_model_samples, O2_data_samples, O2_depths = [], [], []
            secchi_model_samples, secchi_data_samples = [], []
            crashed = False
            try:
                for ii in np.arange(0, len(dates)):
                    dnum = datetime.strptime(dates[ii], "%Y-%m-%d").date()
                    start = timeaxis[0].date()
                    dateindex =(dnum - start).days
                    if dateindex >= 1 and dateindex <= timestep:
                        secchi_data = secchiraw[ii]
                        secchi_calculated = []
                        for i in [1.4, 1.5, 1.6, 1.7]:
                            secchi_calculated.append((i /(np.mean(lambdamodel.loc[dateindex, 0:4]))))
                        secchi_model = secchi_calculated

                        if not isnan(secchi_data):
                            secchi_data_samples.append(secchi_data)
                            secchi_model_samples.append(secchi_model)
                            anydata_secchi = True


            except:
                crashed = True
                print("Error when handling lake %s %s \n" %(name, lake_id))
            label_secchi = "%s %s" %(name, lake_id)
            if crashed:
                label_secchi =("%s - Error in program,unreliable" % label_secchi)
            else:
                if not anydata_secchi:
                    label_secchi = "%s(No data in timespan)" % label_secchi
                else:
                    RMSE_secchi, corr_secchi = [], []
                    for i in np.arange(0, len(secchi_model_samples[0])):
                        RMSE_secchi.append(round(sqrt(sklearn.metrics.mean_squared_error(secchi_data_samples,
                                                                                                 [item[i] for item in
                                                                                                  secchi_model_samples])),
                                                     2))
                        corr_secchi.append(
                            scipy.stats.pearsonr(secchi_data_samples, [item[i] for item in secchi_model_samples]))
                    personsecchi = '1.4: %.3f 1.5: %.3f 1.6: %.3f 1.7: %.3f' %(round(corr_secchi[0][0], 3),
                                                                                round(corr_secchi[1][0], 3),
                                                                                round(corr_secchi[2][0], 3),
                                                                                round(corr_secchi[3][0], 3))
                    RMSEsecchi = '1.4: %.3f 1.5: %.3f 1.6: %.3f 1.7: %.3f' %(
                        round(RMSE_secchi[0], 3), round(RMSE_secchi[1], 3), round(RMSE_secchi[2], 3),
                        round(RMSE_secchi[3], 3))
                    label_secchi = "\n%s,RMSE: %s\n Pearson: %s" %(label_secchi, RMSEsecchi, personsecchi)

            if secchi_model_samples != []:
                mean_secchi_model, std_secchi_model = [], []

                for i in np.arange(0, 4):
                    mean_secchi_model.append(np.mean([item[i] for item in secchi_model_samples]))
                    std_secchi_model.append(np.std([item[i] for item in secchi_model_samples]))
                mean_secchi_data = np.mean(secchi_data_samples)

                std_secchi_data = np.std(secchi_data_samples)

                all_mean_secchi_model.append(mean_secchi_model)
                all_mean_secchi_data.append(mean_secchi_data)
                all_std_secchi_model.append(std_secchi_model)
                all_std_secchi_data.append(std_secchi_data)

    mean_data = np.asarray(all_mean_secchi_data)

    # fig1 = plt.figure(1, figsize=(18.0, 10.0))
    coefficients = [1.4, 1.5, 1.6, 1.7]
    j = 0
    for i in np.arange(0, len(coefficients)):
        mean_model_data = [item[i] for item in all_mean_secchi_model]
        std_model_data = [item[i] for item in all_std_secchi_model]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_mean_secchi_data, mean_model_data)
        # plt.subplot(int('%s%s%s' %(2, 2, i+1)))
        # plt.errorbar(all_mean_secchi_data, mean_model_data, xerr=all_std_secchi_data, yerr=std_model_data,fmt='o',label=coefficients[j])
        # line = np.concatenate([[0], mean_model_data, [15]])
        # plt.plot(line, intercept + slope * line, 'r')
        # plt.title('coefficient: %s r_value: %s' %(coefficients[j],r_value))
        # plt.xlabel("data_samples")
        # plt.ylabel("model_samples")
        # plt.ylim(0, 16)
        # plt.xlim(0,16)
        results = graphique(all_mean_secchi_data, mean_model_data, all_std_secchi_data, std_model_data,
                              coefficients[j], r_value,
                              variable_test,
                              slope, intercept, 'SECCHI')
        # graphique(all_mean_secchi_data, mean_model_data, all_std_secchi_data, std_model_data,coefficients[j],r_value,variable_test,slope,intercept,'SECCHI')
        with open('OLS_Regression_Results.txt', 'a')as file:
            file.write('\nvariable_tested: SWA_b1 %s\n\n' % variable_test)
            file.write('\ncoefficient: %s \n\n %s\n\n' %(coefficients[j], results))
            nlakes = len(lakes)
        j += 1
    plt.gcf().clear()


    # fig1.savefig("%s\\Figure 1 Secchi_regression.png" %(r'C:\Users\Marianne\Desktop\figure_calibration'))

def FishNiche_mean_k_BOD_graph(scenarioid, modelid, lakelistfile, calivari, variable_test):
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2B = y1B + 4
    timeaxis = pd.date_range("%s-01-01" % y1A, "%s-12-31" % y2B, freq="D")
    timestep = len(timeaxis)

    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)

    # lakeswehavedatafor = [698]
    lakeswehavedatafor = [698, 67035, 19167, 31895, 310, 32276, 99045, 99516, 6950]
    # lakeswehavedatafor = [698, 67035, 31895, 310, 32276]
    all_mean_O2_model, all_mean_O2_data = [], []
    all_std_O2_model, all_std_O2_data = [], []

    for lakenum in np.arange(1, nlakes):
        lake_id, subid, name, eh, area, depth, longitude, latitude \
            = lakes[lakenum].strip().split(',')
        if int(lake_id)in lakeswehavedatafor:

            # getOutputPathFromEbHex
            eh = eh[2:] if eh[:2] == '0x' else eh
            while len(eh)< 6:
                eh = '0' + eh
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]

            outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
            print(outdir)

            O2model = pd.read_csv(path.join(outdir, 'O2zt.csv'), header=None)
            lambdamodel = pd.read_csv(path.join(outdir, 'lambdazt.csv'), header=None)

            filename = '../sweden_inflow_data/Validation_data_for_lookup.xlsx'

            # need to be change if the file change
            worksheet = pd.read_excel(filename, sheet_name=lake_id)

            dates = worksheet['date']
            depthdata = worksheet['depth(max)']
            O2raw = worksheet['O2(mg/l)'] * 1000
            secchiraw = worksheet['Siktdjup(m)']

            anydata_secchi, anydata_o = False, False
            O2_model_samples, O2_data_samples, O2_depths = [], [], []
            crashed = False
            try:
                for ii in np.arange(0, len(dates)):
                    dnum = datetime.strptime(dates[ii], "%Y-%m-%d").date()
                    start = timeaxis[0].date()
                    dateindex =(dnum - start).days
                    if dateindex >= 1 and dateindex <= timestep:
                        depthO = ceil(depthdata[ii])
                        O2_data = O2raw[ii]
                        O2_model = O2model.loc[dateindex, depthO - 1]
                        if not isnan(O2_data):
                            O2_data_samples.append(O2_data)
                            O2_model_samples.append(O2_model)
                            O2_depths.append(depthO)
                            anydata_o = True

            except:
                crashed = True
                print("Error when handling lake %s %s \n" %(name, lake_id))
            label_o = "%s %s" %(name, lake_id)
            if crashed:
                label_o =("%s - Error in program,unreliable" % label_o)
            else:
                if not anydata_o:
                    label_o = "%s(No data in timespan)" % label_o
                else:
                    RMSE_O = round(sqrt(sklearn.metrics.mean_squared_error(O2_data_samples, O2_model_samples)),
                                     2)
                    corr_O = scipy.stats.pearsonr(O2_data_samples, O2_model_samples)
                    pearsono = '%.3f' %(round(corr_O[0], 3))

                    label_o = "%s,RMSE: %s\n Pearson: %s" %(label_o, RMSE_O, pearsono)

            if O2_data_samples != []:
                mean_O2_model, std_O2_model = [], []

                mean_O2_model = np.mean(O2_model_samples)
                std_O2_model = np.std(O2_model_samples)
                mean_O2_data = np.mean(O2_data_samples)

                std_O2_data = np.std(O2_data_samples)

                all_mean_O2_model.append(mean_O2_model)
                all_mean_O2_data.append(mean_O2_data)
                all_std_O2_model.append(std_O2_model)
                all_std_O2_data.append(std_O2_data)

    # mean_data = np.asarray(all_mean_O2_data)

    # fig1 = plt.figure(1, figsize=(18.0, 10.0))
    coefficients = [1.4, 1.5, 1.6, 1.7]
    j = 0
    for i in np.arange(0, len(coefficients)):
        mean_model_data = all_mean_O2_model[i]
        std_model_data = all_std_O2_model[i]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_mean_O2_data, mean_model_data)
        # plt.subplot(int('%s%s%s' %(2, 2, i+1)))
        # plt.errorbar(all_mean_secchi_data, mean_model_data, xerr=all_std_secchi_data, yerr=std_model_data,fmt='o',
        # label=coefficients[j])
        # line = np.concatenate([[0], mean_model_data, [15]])
        # plt.plot(line, intercept + slope * line, 'r')
        # plt.title('coefficient: %s r_value: %s' %(coefficients[j],r_value))
        # plt.xlabel("data_samples")
        # plt.ylabel("model_samples")
        # plt.ylim(0, 16)
        # plt.xlim(0,16)
        graphique(all_mean_O2_data, mean_model_data, all_std_O2_data, std_model_data, coefficients[j], r_value,
                    calivari, slope, intercept, 'OXYGEN')
        j += 1
        plt.gcf().clear()
        # fig1.savefig("%s\\Figure 1 Secchi_regression.png" %(r'C:\Users\Marianne\Desktop\figure_calibration'))


if __name__ == '__main__':
    a = 0
    # runlakesGoran_par('test.csv', 4, 2)
    # lakes = FishNiche_generate_timeseries(r'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\
    #     2017SwedenList_only_validation_lakes.csv', 2, 4)
    # FishNiche_cathegories(r'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\2017SwedenList.csv')
    # FishNiche_validate_results(2, 4, r'C:\Users\Marianne\Documents\Fish_niche1\MDN_FishNiche_2017\lakes\
    # 2017SwedenList_only_validation_lakes.csv', 'Final', 0.0001)
    # i = 0.5
    # test = FishNiche_mean_k_DOC_graph(2, 4, r'T:\RMC\Usagers\macot620\Fish_niche\MDN_FishNiche_2017\lakes\
    #     2017SwedenList_only_validation_lakes.csv', 'K_SOD_equation', '_lineeq2')
    # FishNiche_graph_temp_time(2, 4, r'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\
    #     2017SwedenList_only_validation_lakes.csv', 'test', 0.001)
    # FishNiche_graph_oxy_time(2, 2, r'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\
    #     2017SwedenList_only_validation_lakes.csv', 0.3, 1)
    # FishNiche_mean_secchi_graph(2, 4, r'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\
    #     2017SwedenList_only_validation_lakes.csv', 'Equation_1', 'Equation_1')
    # plt.show()
    # FishNiche_graph_temp_complete_time(2, 4, r'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\
    #     2017SwedenList_only_validation_lakes.csv', 2, 1)
    # FishNiche_plot_timeseries(r'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\
    #     2017SwedenList_only_validation_lakes.csv', 4, 2, 1, "../test")
    # print(lakes)
    # FishNiche_graph_temp_time(2, 4, r'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\test.csv')
    # plt.show()
    #FishNiche_plot_volume_param('His',r'D:\Fish_niche\lakes\2017SwedenList.csv', [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8], 1, datafolder)
    generate_timeseries_by_model([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8],r'C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList.csv',datafolder)
    #FishNiche_plot_volume(r'C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList.csv', [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8], 1, datafolder)
    # 
    #generate_timeseries_his_by_model([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8], r'D:\Fish_niche\lakes\2017SwedenList.csv', datafolder)
    # FishNiche_plot_volume_param('His', r'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\
    #     2017SwedenList.csv', [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8], 1, datafolder)
    #FishNiche_plot_volume01(r'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\
    #     2017SwedenList.csv', [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8], 1, datafolder)
    # 
    #generate_timeseries_his_by_model([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8], r'C:\Users\Administrateur\Documents\GitHub\Fish_niche\lakes\2017SwedenList.csv', datafolder)
    #FishNiche_secchi_graph(2, 2, r'C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList_only_validation_12lakes.csv', 100, 10)
