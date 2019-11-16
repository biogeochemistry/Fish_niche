from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.api import add_constant, OLS
import pandas as pd
import numpy as np
import scipy.stats
import csv
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
from math import sqrt
import sys
import numpy as np
import statistics
from matplotlib.patches import Patch
import statistics as st
import matplotlib.lines as mlines
import sklearn.metrics
from math import sqrt, isnan, ceil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
# from pandas.tseries.offsets import MonthBegin,MonthEnd
# import matplotlib.dates as dates
import scipy.stats as stats
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter, DayLocator, WeekdayLocator, MONDAY
# import matplotlib as mpl
from datetime import date, datetime
from os import path
import os
import plotly.express as px
from statsmodels.graphics.gofplots import qqplot
from csv import reader
# from function_runlakesGoran_par_test import runlakesGoran_par
import pandas.plotting._converter

pandas.plotting._converter.register()
plt.style.use('seaborn-poster')

outputfolderdata = r'..\figure_calibration\Figure_test_oxygen'
datafolder = r'C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\output'
outputfolder = r'C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\output'
models = ["GFDL-ESM2M",
          "HadGEM2-ES",
          "IPSL-CM5A-LR",
          "MIROC5",
          "EWEMBI"
          ]
scenarios = ["historical",
             "piControl",
             "rcp26",
             "rcp60",
             "rcp85"
             ]
full_lake_list = ["Mendota", "Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal",
                      "CrystalBog", "Delavan", "FallingCreek", "Fish", "GreatPond",
                      "Green_lake", "Laramie",  "Monona", "Okauchee",
                      "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik",
                      "Trout", "TroutBog", "TwoSisters", "Washington", "Wingra",
                      "Biel", "LowerZurich", "Neuchatel", "Alqueva", "Annecy",
                      "Bourget", "Geneva", "Argyle", "BurleyGriffin", "MtBold",
                      "Dickie", "Eagle_lake", "Harp", "Ekoln", "Erken",
                      "EsthwaiteWater", "Windermere", "Feeagh", "Kilpisjarvi", "Kuivajarvi",
                      "Paajarvi", "Kinneret", "Kivu", "Klicava",
                      "Zlutice", "Langtjern", "Mozhaysk", "Vendyurskoe", "Muggelsee",
                      "Rappbode", "Stechlin", "Ngoring", "NohipaloMustjarv", "NohipaloValgejarv",
                      "Vortsjarv", "Sau", "Rotorua", "Tarawera", "Taupo","Rimov",
                      "Waahi"]
# full_lake_list = ["Feeagh", "Kilpisjarvi", "Kuivajarvi",
#                       "Paajarvi", "Kinneret", "Kivu", "Klicava",
#                       "Zlutice", "Langtjern", "Mozhaysk", "Vendyurskoe", "Muggelsee",
#                       "Rappbode", "Stechlin", "Ngoring", "NohipaloMustjarv", "NohipaloValgejarv",
#                       "Vortsjarv", "Sau", "Rotorua", "Tarawera", "Taupo","Rimov",
#                       "Waahi"]
#full_lake_list=["LowerZurich"]
regions = {"US": ["Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal", "CrystalBog", "Delavan",
                  "FallingCreek", "Fish", "GreatPond", "Green_lake", "Laramie", "Mendota", "Monona",
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

models = ["GFDL-ESM2M",
          "HadGEM2-ES",
          "IPSL-CM5A-LR",
          "MIROC5",
          "EWEMBI"
          ]
scenarios = ["historical",
             "piControl",
             "rcp26",
             "rcp60",
             "rcp85"
             ]

input_variables = ["hurs",
                   "pr",
                   "ps",
                   "rsds",
                   "sfcWind",
                   "tas"
                    ]

report = 'report.txt'

years = YearLocator()  # every year
# months = MonthLocator ()  # every month
days = DayLocator()
yearsFmt = DateFormatter('%Y')
# monthsFmt=DateFormatter('%M')
months = MonthLocator()
mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()              	# minor ticks on the days
weekFormatter = DateFormatter('%b')  # e.g., Jan 12
dayFormatter = DateFormatter('%d')
# format the coords message box



def print11111():
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


def findYPoint(xa,xb,ya,yb,xc):

    m = (ya - yb) / (xa - xb)
    yc = (xc - xb) * m + yb
    return yc




def FishNiche_csv_results_revise(scenarioid,modelid,lake_list,letter):
    global fig1
    scenario = scenarios[scenarioid]
    model = models[modelid]

    fig1,fig2,fig3,fig4 =plt.figure(),plt.figure(),plt.figure(),plt.figure()
    alldates,allobsT,allmodelT,alldepths,alllakes =[],[],[],[],[]

    for lakenum in range(0,len(lake_list)):
        fig1=plt.figure()
        lake = lake_list[lakenum]
        print(lake)
        reg = None
        for region in regions:
            if lake in regions[region]:
                reg = region
                break

        if reg == None:
            print("Cannot find {}'s region".format(lake))

        else:
            outdir = path.join(r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\output/{}/{}/{}/{}".format(reg, lake, model, scenario) )
            if not path.exists("{}/Calibration_Complete.txt".format(outdir)):
                print("{}/Calibration_Complete doesn't exist".format(outdir))
            else:
                with open("{}/Observed_Temperatures.csv".format(outdir), "r") as observation_file:
                    reader = list(csv.reader(observation_file))

                    start_year = int(reader[1][0][:4])
                    end_year = int(reader[-1][0][:4])
                if start_year < 2016:
                    if start_year > 1979:
                        y1 = start_year

                        if end_year < 2016:
                            y2 = end_year
                        else:
                            y2 = 2016
                    else:
                        y1 = 1979
                        if end_year < 2016:
                            y2 = end_year
                        else:
                            y2 = 2016
                    timeaxis = pd.date_range("%s-01-01" % y1, "%s-12-31" % y2, freq="D")
                    timestep = (timeaxis[-1] - timeaxis[0]).days
                    observed_T = pd.read_csv("{}/Observed_Temperatures.csv".format(outdir))


                    Tmodel = pd.read_csv( path.join( outdir, 'Tzt.csv' ), header=None )
                    maxdepth = len(Tmodel.columns) - 1
                    print(outdir)
                    dates = observed_T["Date"]
                    observed_T["Date"] = pd.to_datetime(observed_T["Date"], format="%Y%m%d")
                    observed_T = observed_T.set_index('Date')

                    depthdata=list(observed_T.columns)




                    anydata_t,anydata_o = False,False
                    T_model_samples,T_data_samples,T_depths = [],[],[]

                    crashed = False

                    #try:
                    for ii in np.arange(0,len(dates)):
                        dnum = datetime.strptime(str(dates[ii]),"%Y%m%d").date()
                        start = timeaxis[0].date()
                        dateindex = (dnum - start).days
                        if dateindex >= 1 and dateindex < timestep:

                            for depth in depthdata:
                                if maxdepth > int(float(depth))+1:
                                    if not observed_T.loc[dnum, depth] =="None":
                                        T_data = float(observed_T.loc[dnum, depth])
                                        if int(float(depth)) != float(depth):

                                            xa = int(float(depth))
                                            xb=int(float(depth))+1
                                            ya =float(Tmodel.iloc[dateindex, xa])

                                            yb=float(Tmodel.iloc[dateindex, xb])
                                            print(lake,dateindex,xb)
                                            T_model = findYPoint(xa, xb, ya, yb, float(depth))


                                        else:

                                            depth = float(depth)

                                            T_model = float(Tmodel.loc[dateindex, int(depth)])


                                        T_data_samples.append ( T_data )
                                        T_model_samples.append(T_model)
                                        T_depths.append(float(depth))
                                        anydata_t = True
                                        alldates.append(dnum)
                                        allobsT.append ( T_data )
                                        allmodelT.append(T_model)
                                        alldepths.append(float(depth)/maxdepth)
                                        alllakes.append(lake)



                    # except:
                    #     crashed = True
                    #     print("Error when handling lake %s \n"% (lake))
                    # label_t = "%s"% (lake)
                    #
                    #
                    # if crashed :
                    #     label_t =("%s - Error in program,unreliable"%label_t)
                    #
                    # else:
                    #     if not anydata_t:
                    #         label_t = "%s (No data in timespan)"%label_t
                    #     else:
                    #         RMSE_T = round( sqrt( sklearn.metrics.mean_squared_error( T_data_samples, T_model_samples ) ), 2 )
                    #         corr_T = scipy.stats.pearsonr( T_data_samples, T_model_samples )
                    #         pearsont = '%.3f'%(round ( corr_T[0],3 ))
                    #         label_t = "%s\n (RMSE: %s Pearson: %s)"%(label_t,RMSE_T,pearsont)



                    lineStart = 0
                    lineEnd = 25
                    rows = ceil(sqrt(len(lake_list)-1))
                    if rows == 0:
                        rows = 1
                        cols = 1
                    else:
                        cols = ceil((len(lake_list)-1)/rows)


                    plt.rc ( 'axes', linewidth=2 )
                    fontsize = 30
                    lw=0.5
                    markers = ['o', 'v', '+', '*', '8', '^','D' ,'s', 'd', 'p', 'x']
                    markers = ['d','v', '+', '*', '8', 's', 'D', '^','o' ]
                    s = [100,100*1.5,100,100*2,100,100*1.5,100,100,100*1.5,100,100]
                    i=100*1.5
                    s = [ i*1.5, i*2, i, i * 2.5, i, i*1.5, i, i*2, i]
                    if lakenum < len(markers):
                        mark = markers[lakenum-1]
                        size = s[lakenum-1]
                    else:
                        mark = markers[0]
                        size = s[0]
                    alpha= 0.8
                    # params = {
                    #     'axes.labelsize': 30,
                    #     'text.fontsize': 30,
                    #     'legend.fontsize': 30,
                    #     'xtick.labelsize': 14,
                    #     'ytick.labelsize': 14,
                    #     'text.usetex': False,
                    #     'figure.figsize': [14,10]  # instead of 4.5, 4.5
                    # }
                    #
                    # plt.rcParams.update ( params )
                    colorset = ['k','cyan','maroon','C1','C6','C2','C3','maroon','gold','C8']#['C1','C3','y','k','w','C7','C0','C2','C9']

                    if anydata_t:
                        fig1=plt.figure(100*lakenum)#,figsize=(3.0, 3.0))
                        ax = plt.gca ()
                        color = [1-(int(i) / maxdepth) for i in T_depths]
                        #points=plt.scatter(T_data_samples,T_model_samples,label=label_t,s=size,c=color,edgecolors='k',linewidths=lw,cmap='Blues_r',marker=mark,alpha=alpha)
                        points=plt.scatter(T_data_samples,T_model_samples,s=size,c=color,edgecolors='k',linewidths=lw,cmap='Blues_r',marker=mark,alpha=alpha)
                    #for i,txt in enumerate(T_depths):
                        #plt.annotate(txt,(T_data_samples[i],T_model_samples[i]))
                        plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-',linewidth=1.0)
                        plt.tick_params ( axis='both', which='major')#, labelsize=14 )
                        for tick in ax.xaxis.get_major_ticks ():
                            tick.label1.set_fontsize ( fontsize )
                            #tick.label1.set_fontweight ( 'bold' )
                        for tick in ax.yaxis.get_major_ticks ():
                            tick.label1.set_fontsize ( fontsize )
                            #tick.label1.set_fontweight ( 'bold' )
                        plt.xlabel ('Observed Temperature($\degree$C)')#,fontweight='bold' )
                        plt.ylabel ( 'Modeled Temperature($\degree$C)')#,fontweight='bold' )
                        #plt.legend(loc=4)
                        #if numberlakes == len(lakeswehavedatafor):
                         #cb = plt.colorbar ( points , ticks=np.linspace ( 1.1, -0.1, 12, endpoint=True ) )
                         #cb.set_label ( weight='bold' )
                         #cb.ax.set_yticklabels ( ['surface', '', '', '','','','','',   'max depth'])#,weight='bold' )
                        plt.tight_layout ()
                        fig1.savefig(path.join(datafolder, "Figure_1_temp_%s_%s.png" % (letter, lake)))


                        lineStart = 0
                        lineEnd = 35
                        fontsize = 30
                        alldf.set_index(alldf['dates'])
                        fig4=plt.figure(101 * lakenum)  # ,figsize=(3.0, 3.0))

                        #color = [i for i in alldf['depths']]

                        plt.plot_date(alldf['dates'],alldf['residuals'],marker='.', alpha=0.8,c="Blue",  linestyle='None')

                        plt.xlabel('Time')  # ,fontweight='bold' )
                        plt.ylabel('Residuals')
                        #plt.show()

                        #plt.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                        #plt.xaxis.set_major_formatter(DateFormatter("%m-%d"))
                        fig4.savefig(path.join(datafolder, "Figure_4_residualvstime_%s_%s.png" % (letter,lake)))

                        #fig2=plt.figure((2)*110)#,figsize=(18.0, 10.0))
                        #ax = plt.gca ()

                        #plt.subplot(rows,cols,lakenum+1)
                        #points = plt.scatter(T_data_samples,T_model_samples,c=T_depths,edgecolors='k',label=label_t,s=20,cmap='bone_r')#T_depths
                        #points = plt.scatter(T_data_samples, T_model_samples)#, c=T_depths, edgecolors='k', s=20, cmap='bone_r')  # T_depths
                        #plt.colorbar(points)
                        #plt.clim ( [0, maxdepth ])
                        #plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
                        #plt.title(label_t,fontsize=14,y=1.02)
                        #plt.subplots_adjust ( hspace=0.45, wspace=0.2)
                        #plt.tick_params ( axis='both', which='major')#, labelsize=14 )
                        #plt.yticks ( np.arange ( 0, 35, 5 ) )
                        #fig2.text ( 0.5, 0.02, 'Modeled Temperature($\degree$C)', ha='center' )
                        #fig2.text ( 0.07, 0.5, 'Observed Temperature($\degree$C)', va='center', rotation='vertical' )
                        #fig2.text ( 0.92, 0.5, 'depth(m)', va='center', rotation='vertical' )

        lineStart = 0
        lineEnd = 25
        fontsize = 30
        fig2 = plt.figure(100 * 67)  # ,figsize=(3.0, 3.0))
        ax = plt.gca()
        color = [i for i in alldepths]
        # points=plt.scatter(T_data_samples,T_model_samples,label=label_t,s=size,c=color,edgecolors='k',linewidths=lw,cmap='Blues_r',marker=mark,alpha=alpha)
        points = plt.scatter(allobsT, allmodelT, s=100, c=color, edgecolors='k', linewidths=0.5,
                             cmap='Blues_r', marker='o', alpha=0.8)
        # for i,txt in enumerate(T_depths):
        # plt.annotate(txt,(T_data_samples[i],T_model_samples[i]))
        plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', linewidth=1.0)
        plt.tick_params(axis='both', which='major')  # , labelsize=14 )
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            # tick.label1.set_fontweight ( 'bold' )
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            # tick.label1.set_fontweight ( 'bold' )
        plt.xlabel('Observed Temperature($\degree$C)')  # ,fontweight='bold' )
        plt.ylabel('Modeled Temperature($\degree$C)')  # ,fontweight='bold' )
        # plt.legend(loc=4)
        # if numberlakes == len(lakeswehavedatafor):
        # cb = plt.colorbar ( points , ticks=np.linspace ( 1.1, -0.1, 12, endpoint=True ) )
        # cb.set_label ( weight='bold' )
        # cb.ax.set_yticklabels ( ['surface', '', '', '','','','','',   'max depth'])#,weight='bold' )
        plt.tight_layout()


        alldata = list(zip(alldates,allobsT,allmodelT,alldepths,alllakes))
        alldf=pd.DataFrame(alldata,columns=['dates','obsT','modelT','depths','lakes'])
        alldf['residuals'] = alldf['obsT']-alldf['modelT']
        alldf.to_csv( path.join ( datafolder,"comparison_all_lakeD.csv"),index=False)


                    # calibration = abs(sum(T_model_samples)-sum(T_data_samples))**2
        #calibrationdata.to_csv(os.path.join ( outputfolderdata, 'calibration,csv' ))
        #df_csv = pd.read_csv ( os.path.join ( outputfolderdata, 'calibration.csv' ) )
        #result = pd.merge(df_csv, calibrationdata, on='lake_id')
        #result.to_csv ( os.path.join ( outputfolderdata, 'calibration.csv' ) ,index=False)

        #fig1.savefig(path.join(datafolder, "Figure_1_temp_Btest%s.png" % letter))
        #plt.gcf ().clear (fig1)
        fig2.savefig ( path.join ( datafolder, "Figure_2_temp_alltest_%s.png" % letter ) )
        #plt.gcf ().clear (fig2)

        #return calibrationdata

def analysisalllake():
    alldf = pd.read_csv(path.join ( datafolder,"comparison_all_lakeC.csv"))
    letter='C'

    # sos1 = sums_of_squares(alldf["obsT"], alldf["modelT"])
    # rms1 = root_mean_square(alldf["obsT"], alldf["modelT"])
    # r_squ1 = r2_score(alldf["obsT"], alldf["modelT"])
    # rmsesd = rmse_by_sd(alldf["obsT"], rms1)
    # analyse = 'SS: %s RMSE: %s \n R2: %s RMSE/SD: %s ' % (sos1, rms1, r_squ1, rmsesd)
    analyse = "1"
    print(analyse)
    # # Density Plot and Histogram of all arrival delays
    # ax = sns.distplot(alldf['residuals'], hist=True, kde=False,
    #              bins=int(180 / 5), color='darkblue',
    #              hist_kws={'edgecolor': 'black'},
    #              kde_kws={'linewidth': 4})
    # plt.axvline(np.mean(alldf['residuals']), 0,50000,linestyle='--',c='k')
    # ax.text(6, 45000,'Mean: %.3f Sd: %.3f \n %s' % (np.mean(alldf['residuals']), np.std(alldf['residuals']), analyse),fontsize=16)
    # fig1 = ax.get_figure()
    #
    #
    # ax.set(xlabel='Model_T - Obs_T($\degree$C)', ylabel='Frequency')
    # fig1.savefig(path.join(datafolder, "Figure_1_residual_%s.png" % (letter)))
    #

    # alldf['residuals2'] = alldf['obsT'] - alldf['modelT']
    # fig2 = qqplot(alldf['residuals2'],line='s')
    #
    # plt.xlabel('Sample Quantiles')  # ,fontweight='bold' )
    # plt.ylabel('Theorical Quantiles')  # ,fontweight='bold' )
    #
    # fig2.savefig(path.join(datafolder, "Figure_2_qqplot_%s.png" % (letter)))
    #
    # lineStart = 0
    # lineEnd = 35
    # fontsize = 30
    #
    #
    # fig3 = plt.figure(100 * 67)  # ,figsize=(3.0, 3.0))
    # ax = plt.gca()
    # color = [i for i in alldf['depths']]
    # # points=plt.scatter(T_data_samples,T_model_samples,label=label_t,s=size,c=color,edgecolors='k',linewidths=lw,cmap='Blues_r',marker=mark,alpha=alpha)
    # points = plt.scatter( alldf['residuals'],alldf['modelT'], s=20, c=color, edgecolors='k', linewidths=0.1,
    #                      cmap='Blues_r', marker='o', alpha=0.8)
    # # for i,txt in enumerate(T_depths):
    # # plt.annotate(txt,(T_data_samples[i],T_model_samples[i]))
    # plt.plot([0,30], [0, 0], 'k-', linewidth=2.0)
    # plt.tick_params(axis='both', which='major')  # , labelsize=14 )
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label1.set_fontsize(fontsize)
    #     # tick.label1.set_fontweight ( 'bold' )
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label1.set_fontsize(fontsize)
    #     # tick.label1.set_fontweight ( 'bold' )
    # plt.ylabel('Modeled Temperature($\degree$C)')  # ,fontweight='bold' )
    # plt.xlabel('Residuals')  # ,fontweight='bold' )
    # plt.tight_layout()
    #
    #
    # fig3.savefig(path.join(datafolder, "Figure_3_residualvsmodel_%s.png" % (letter)))
    #
    # lineStart = 0
    # lineEnd = 35
    # fontsize = 30
    # alldf.set_index(alldf['dates'])
    # fig4 = plt.figure(100 * 67)  # ,figsize=(3.0, 3.0))
    #
    # #color = [i for i in alldf['depths']]
    #
    # plt.plot_date(alldf['dates'],alldf['residuals'],marker='.', alpha=0.8,c="Blue",  linestyle='None')
    #
    # plt.xlabel('Time')  # ,fontweight='bold' )
    # plt.ylabel('Residuals')
    # #plt.show()
    #
    # #plt.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    # #plt.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    # plt.savefig(path.join(datafolder, "Figure_4_residualvstime_%s.png" % (letter)))

    lineStart = 0
    lineEnd = 1
    fontsize = 30

    fig5 = plt.figure(100 * 67)  # ,figsize=(3.0, 3.0))
    ax = plt.gca()
    color = [i for i in alldf['depths']]
    # points=plt.scatter(T_data_samples,T_model_samples,label=label_t,s=size,c=color,edgecolors='k',linewidths=lw,cmap='Blues_r',marker=mark,alpha=alpha)
    points = plt.scatter(alldf['residuals'], alldf['depths'], s=20, c=color, edgecolors='k', linewidths=0.1,
                         cmap='Blues_r', marker='o', alpha=0.8)
    # for i,txt in enumerate(T_depths):
    # plt.annotate(txt,(T_data_samples[i],T_model_samples[i]))
    plt.plot([0, 0], [1, 0], 'k--', linewidth=1.0)
    plt.tick_params(axis='both', which='major')  # , labelsize=14 )
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        # tick.label1.set_fontweight ( 'bold' )
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        # tick.label1.set_fontweight ( 'bold' )
    plt.xlabel('Residuals')  # ,fontweight='bold' )
    plt.ylabel('Depth')  # ,fontweight='bold' )
    plt.tight_layout()
    #plt.plot([0,0], [30, 0], 'k--', linewidth=1.0)
    #alldf.set_index(alldf['dates'])
    fig5.savefig(path.join(datafolder, "Figure_5_residualvsdepth_%s.png" % (letter)))

    lineStart = 0
    lineEnd = 30
    fontsize = 30

    fig6 = plt.figure(100 * 67)  # ,figsize=(3.0, 3.0))
    ax = plt.gca()
    color = [i for i in alldf['depths']]
    # points=plt.scatter(T_data_samples,T_model_samples,label=label_t,s=size,c=color,edgecolors='k',linewidths=lw,cmap='Blues_r',marker=mark,alpha=alpha)
    points = plt.scatter(alldf['modelT'], alldf['obsT'], s=20, c=color, edgecolors='k', linewidths=0.1,
                         cmap='Blues_r', marker='o', alpha=0.5)
    # for i,txt in enumerate(T_depths):
    # plt.annotate(txt,(T_data_samples[i],T_model_samples[i]))
    plt.plot([lineStart, lineStart], [lineEnd, lineEnd], 'k-', linewidth=1.0)
    plt.tick_params(axis='both', which='major')  # , labelsize=14 )
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        # tick.label1.set_fontweight ( 'bold' )
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        # tick.label1.set_fontweight ( 'bold' )
    plt.xlabel('Modeled Temperature($\degree$C)')  # ,fontweight='bold' )
    plt.ylabel('observedT')  # ,fontweight='bold' )
    plt.tight_layout()

    alldf.set_index(alldf['dates'])
    fig6.savefig(path.join(datafolder, "Figure_6_modelvsobs_%s.png" % (letter)))

def findgoodHypso(lake_list):







    for lake in lake_list:
        corrected_names = ["Allequash_Lake", "Big_Muskellunge_Lake", "Black_Oak_Lake", "Burley_Griffin", "Crystal_Bog",
                           "Crystal_Lake",
                           "Dickie_Lake", "Eagle_Lake", "Ekoln_basin_of_Malaren", "Esthwaite_Water",
                           "Falling_Creek_Reservoir",
                           "Fish_Lake", "Great_Pond", "Green_Lake", "Harp_Lake", "Laramie_Lake", "Lower_Zurich",
                           "Mt_Bold",
                           "Nohipalo_Mustjarv", "Nohipalo_Valgejarv", "Okauchee_Lake", "Rappbode_Reservoir",
                           "Sau_Reservoir",
                           "Sparkling_Lake", "Toolik_Lake", "Trout_Bog", "Trout_Lake", "Two_Sisters_Lake"]

        f_lake = lake
        for name in corrected_names:
            if lake == "Crystal":
                f_lake = "Crystal_Lake"
                break
            elif lake == "Trout":
                f_lake = "Trout_Lake"
                break

            if lake in name.replace("_", ''):
                f_lake = name
                break


        reg = None
        for region in regions:
            if lake in regions[region]:
                reg = region
                break

        if reg == None:
            print("Cannot find {}'s region".format(lake))
            return None

        else:
            hypso2 = path.join(
            r"C:\Users\macot620\Documents\GitHub\Fish_niche\\ISIMIP\observations/{}/{}/{}_hypsometry2.csv".format(reg,
                                                                                                                  lake,
                                                                                                                  lake))

            print(lake)
            i=0
            id=True
            needed=False
            hypsometry=[]
            #if not os.path.exists(r"C:\Users\macot620\Documents\GitHub\Fish_niche\\ISIMIP\observations/{}/{}/{}_hypsometry2.csv".format(reg, lake, lake)) and os.path.exists(r"C:\Users\macot620\Documents\GitHub\Fish_niche\\ISIMIP\observations/{}/{}/{}_hypsometry.csv".format(reg, lake, lake)):
            if os.path.exists(r"C:\Users\macot620\Documents\GitHub\Fish_niche\\ISIMIP\observations/{}/{}/{}_hypsometry.csv".format(reg, lake, lake)) and os.path.exists(r"C:\Users\macot620\Documents\GitHub\Fish_niche\\ISIMIP\observations/{}/{}/{}_hypsometry.csv".format(reg, lake, lake)):

                with open(r"C:\Users\macot620\Documents\GitHub\Fish_niche\\ISIMIP\observations/{}/{}/{}_hypsometry.csv".format(reg, lake, lake)) as hypso:
                    rows = list(csv.reader(hypso))
                    for row in rows:
                        if id is True:
                            id = False
                        else:

                            if i == 0:
                                site = row[0]
                                name = row[1]

                            if float(row[2]) <= i:
                                hypsometry.append(row)
                                if float(row[2]) == i:
                                    i += 1

                            elif float(row[2]) > i:
                                for k in range(i, i + 20):

                                    if not hypsometry:
                                        hypsometry.append([site, name, i, np.nan])
                                        i += 1
                                    elif float(row[2])== i:
                                        hypsometry.append(row)
                                        i += 1
                                        break
                                    elif float(row[2]) != i:
                                        hypsometry.append([site, name, i, np.nan])
                                        i+=1


                df = pd.DataFrame(hypsometry, columns=["SITE_ID","SITE_NAME","DEPTH","BATHYMETRY_AREA"])

                df['DEPTH']= pd.to_numeric(df['DEPTH'], errors='coerce')
                df = df.set_index(df["DEPTH"])
                df['BATHYMETRY_AREA']= pd.to_numeric(df['BATHYMETRY_AREA'], errors='coerce')
                df3 =  df['BATHYMETRY_AREA']

                df['BATHYMETRY_AREA'] = df3.interpolate(method='index', limit_direction="both")

                depth = [x for x in range(0,int(df.iloc[-1]["DEPTH"])+1)]
                depth.append(df.iloc[-1]["DEPTH"])
                df2= df[df['DEPTH'].isin(depth)]
                df2.to_csv(r"C:\Users\macot620\Documents\GitHub\Fish_niche\\ISIMIP\observations/{}/{}/{}_hypsometry2.csv".format(reg, lake, lake),index=False)









if __name__ == "__main__":


    init_file = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\input\CA\Dic\Dic_init"
    parameter_file = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\input\CA\Dic\Dic_par"
    input_file = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\input\CA\Dic\Dic_EWEMBI_historical_input"
    outfolder = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\output\CA\Dickie\EWEMBI\historical"
    y1,y2 = 1979,2016
    cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\');quit' % (
    init_file, parameter_file, input_file, y1, y2, outfolder)
    #print(cmd)
    #os.system(cmd)

    FishNiche_csv_results_revise(0, -1, full_lake_list, "D")
    #findgoodHypso(full_lake_list)
    #analysisalllake()