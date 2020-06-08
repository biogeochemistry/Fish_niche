"""
run_mylake_parallel.py

Module that allows running multiple simulations or calibrations in parallel.
"""
import run_myLake_ISIMIP
import matplotlib.pyplot as plt
import myLake_post
import csv
import sys
from datetime import datetime, timedelta, date
import netCDF4 as ncdf
import math
import pandas as pd
import os
import pysftp
from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import fileinput
import sys
from shutil import copyfile
from netCDF4 import Dataset
import txt_to_netcdf

num_cores = mp.cpu_count()-1
hyspomissing =[ "Monona","Ngoring",]
allmissing = ["Annie" ,"Klicava","Zlutice",]
forcingdatamissing=["Mozhaysk","Muggelsee","Taupo","Waahi",]
order = ["Eagle", "Ekoln","Sau","Tahoe", "TwoSisters","Vendyurskoe","Washington",]

problematic = ["Trout","FallingCreek", "GreatPond","Feeagh","MtBold","Rotorua",]

optimisation_completed = ["BigMuskellunge", "Allequash","CrystalBog","Erken","BurleyGriffin",
                          "Dickie", "Langtjern","LowerZurich","Mendota","Delavan",
                          "Sparkling", "Trout", "TroutBog","Fish","Wingra",
                          "Harp","NohipaloValgejarv","Alqueva","Toolik","Sunapee",
                          "Kuivajarvi","Crystal"]
test =[]
coundntop =[ "Sammamish",
           "Geneva","Kinneret","Bourget","Stechlin","Argyle","Rappbode","BlackOak",
           "NohipaloMustjarv","Green_lake","Paajarvi","Annecy","Okauchee","EsthwaiteWater",  "Laramie","Rimov",
           "Biel","Tarawera", "Neuchatel",
           "Kilpisjarvi","Vortsjarv","Kivu"]


lake_list = ["Allequash", "Annecy", "Biel", "BigMuskellunge", "BlackOak",
             "BurleyGriffin", "Crystal", "Delavan","Dickie", "Erken",
             "Fish", "Geneva", "Green","Harp","Kilpisjarvi",
             "Kinneret", "Kivu", "Langtjern", "Laramie","LowerZurich",
             "Mendota", "Mozaisk", "Neuchatel", "Okauchee", "Paajarvi",
             "Rotorua", "Sparkling", "Stechlin","Sunapee", "Tarawera",
             "Toolik", "Trout", "TroutBog", "TwoSisters","Wingra"
             ]
problem = []
full_lake_list =["Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal","CrystalBog", "Delavan","FallingCreek", "Fish", "Great_Pond",
                 "Green_lake", "Laramie","Mendota" , "Monona","Okauchee", "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik",
                 "Trout", "TroutBog", "Two_Sisters","Washington", "Wingra", "Biel", "LowerZurich", "Neuchatel","Alqueva","Annecy",
                 "Bourget", "Geneva","Argyle", "BurleyGriffin", "MtBold","Dickie", "Eagle_lake", "Harp","Ekoln", "Erken",
                 "EsthwaiteWater", "Windermere","Feeagh","Kilpisjarvi", "Kuivajarvi","Paajarvi","Kinneret","Klicava", "Rimov",
                 "Zlutice","Langtjern","Mozhaysk", "Vendyurskoe","Muggelsee", "Rappbode", "Stechlin","Ngoring","NohipaloMustjarv", "NohipaloValgejarv",
                 "Vortsjarv","Sau","Rotorua", "Tarawera", "Taupo","Waahi"]#"Kivu",

#full_lake_list =["Langtjern"]
#"Ngoring","Klicava",
full_lake_list1 =["Allequash", "Annie",  "BlackOak", "Crystal","CrystalBog",
                 "Delavan","Falling_Creek", "Fish", "Great_Pond","Green_Lake",
                 "Laramie", "Mendota", "Okauchee","Sparkling", "Sunapee",
                 "Toolik","Trout", "TroutBog", "Two_Sisters_Lake", "Wingra",
                 "Biel", "LowerZurich", "Neuchatel","Annecy","Bourget",
                 "BurleyGriffin", "MtBold","Dickie", "Eagle_lake", "Harp",
                 "Ekoln", "Erken", "EsthwaiteWater", "Windermere","Feeagh",
                 "Kilpisjarvi", "Kuivajarvi","Paajarvi","Kinneret","Kivu",
                 "Rimov", "Langtjern","Mozhaysk","Rappbode", "Stechlin",
                 "NohipaloMustjarv", "NohipaloValgejarv","Vortsjarv","Sau", "Tarawera",
                 "Washington","Sammamish", "Geneva","Tahoe"]

full_lake_list_done=['Langtjern','Annecy','Argyle','Crystal','Dickie',
                'Ekoln','Erken','EsthwaiteWater','Feeagh','Fish',
                'Harp','Kilpisjarvi','Kinneret','Kivu','LowerZurich',
                'Mendota','Neuchatel','Okauchee','Rimov','Rotorua',
                'Sammamish','Sau','Sparkling','Stechlin','Sunapee',
                'Vendyurskoe','Vortsjarv','Windermere']


bug = ['Allequash_Lake','Big_Muskellunge_Lake','Black_Oak_Lake', 'Bourget','Crystal_Bog','Falling_Creek_Reservoir',
       'Green_Lake','Kuivajarvi','Mozaisk','Mueggelsee', 'Nohipalo_Mustjarv','Nohipalo_Valgejarv','Paajarvi','Tahoe','Tarawera',]
#'Ngoring',
#
full_lake_list1= ['Annie',  'Alqueva', 'Annecy', 'Argyle','Biel',
                  'Burley_Griffin','Crystal_Lake', 'Delavan', 'Dickie_Lake', 'Eagle_Lake',
                   'Ekoln_basin_of_Malaren', 'Erken', 'Esthwaite_Water',  'Feeagh','Fish_Lake',
                  'Geneva', 'Great_Pond',  'Harp_Lake', 'Kilpisjarvi', 'Kinneret',
                  'Kivu', 'Langtjern','Laramie_Lake', 'Lower_Zurich', 'Mendota',
                  'Monona', 'Mt_Bold',  'Neuchatel','Ngoring','Okauchee_Lake',
                  'Rappbode_Reservoir','Rimov','Rotorua','Sammamish', 'Sau_Reservoir',
                  'Sparkling_Lake','Stechlin','Sunapee', 'Toolik_Lake','Trout_Bog',
                 'Trout_Lake','Two_Sisters_Lake', 'Vendyurskoe', 'Vortsjarv', 'Washington',
                 'Windermere','Wingra']
# #
full_lake_list_done=['Big_Muskellunge_Lake',]
full_lake_lis1t= [ 'Annie','Allequash_Lake', 'Alqueva','Biel','Big_Muskellunge_Lake',
                  'Black_Oak_Lake', 'Bourget', 'Burley_Griffin','Crystal_Bog','Delavan',
                  'Eagle_Lake', 'Falling_Creek_Reservoir', 'Geneva', 'Great_Pond','Green_Lake',
                  'Kuivajarvi',  'Monona','Mozaisk','Mt_Bold', 'Mueggelsee',
                  'Ngoring', 'Nohipalo_Mustjarv','Nohipalo_Valgejarv','Paajarvi', 'Rappbode_Reservoir',
                  'Tahoe', 'Tarawera','Toolik_Lake','Trout_Bog','Trout_Lake',
                  'Two_Sisters_Lake',  'Washington','Wingra']
#full_lake_list= [ 'Tahoe']
full_lake_list= ['Tahoe', 'Tarawera','Annie','Allequash_Lake',  'Annecy', 'Argyle','Alqueva','Biel', 'Big_Muskellunge_Lake', 'Black_Oak_Lake',
                 'Bourget', 'Burley_Griffin','Crystal_Bog', 'Crystal_Lake', 'Delavan', 'Dickie_Lake', 'Eagle_Lake','Ekoln_basin_of_Malaren', 'Erken', 'Esthwaite_Water',
                 'Falling_Creek_Reservoir', 'Feeagh','Fish_Lake', 'Geneva', 'Great_Pond','Green_Lake', 'Harp_Lake','Kilpisjarvi', 'Kinneret','Kivu',
                 'Kuivajarvi','Langtjern','Laramie_Lake','Lower_Zurich', 'Mendota', 'Monona','Mozaisk','Mt_Bold','Mueggelsee', 'Neuchatel',
                 'Ngoring', 'Nohipalo_Mustjarv','Nohipalo_Valgejarv','Okauchee_Lake', 'Paajarvi', 'Rappbode_Reservoir','Rimov','Rotorua','Sammamish', 'Sau_Reservoir',
                 'Sparkling_Lake','Stechlin','Sunapee', 'Toolik_Lake','Trout_Bog','Trout_Lake','Two_Sisters_Lake', 'Vendyurskoe', 'Vortsjarv', 'Washington',
                 'Windermere','Wingra']#

#full_lake_list= [  'Tarawera','Tahoe',]#,
#full_lake_list =[]

full_done2=['Allequash_Lake','Biel', 'Big_Muskellunge_Lake', 'Black_Oak_Lake', 'Bourget',
            'Crystal_Bog','Delavan','Eagle_Lake','Falling_Creek_Reservoir','Great_Pond',
            'Green_Lake', 'Kuivajarvi','Monona','Mozaisk','Ngoring',
            'Nohipalo_Mustjarv', 'Nohipalo_Valgejarv','Paajarvi', 'Rappbode_Reservoir','Toolik_Lake',
            'Trout_Bog','Two_Sisters_Lake',   'Washington','Wingra']
full_lake_list_alldone= ['Annecy', 'Alqueva','Erken', 'Feeagh','Geneva',
                        'Kilpisjarvi', 'Kinneret', 'Langtjern','Mendota', 'Rotorua',
                        'Sammamish', 'Sunapee', 'Vendyurskoe', 'Vortsjarv', 'Windermere','Allequash_Lake']

full_lake_list1=['Big_Muskellunge_Lake']#"Annie","Two_Sisters_Lake"]
#full_lake_list=['Annie',"Falling_Creek", "Great_Pond",
#                  "Two_Sisters_Lake"]

#full_lake_list = "Allequash"
regions = {"US": ["Allequash_Lake", "Annie", "Big_Muskellunge_Lake", "Black_Oak_Lake", "Crystal_Lake",
                  "Crystal_Bog", "Delavan", "Falling_Creek_Reservoir", "Fish_Lake", "Great_Pond",
                  "Green_Lake", "Laramie_Lake", "Mendota", "Monona", "Okauchee_Lake",
                  "Sammamish", "Sparkling_Lake", "Sunapee", "Tahoe", "Toolik_Lake",
                  "Trout_Lake", "Trout_Bog", "Two_Sisters_Lake", "Washington", "Wingra"],
           "CH": ["Biel", "Lower_Zurich", "Neuchatel"],
           "PT": ["Alqueva"],
           "FR": ["Annecy", "Bourget", "Geneva"],
           "AU": ["Argyle", "Burley_Griffin", "Mt_Bold"],
           "CA": ["Dickie_Lake", "Eagle_Lake", "Harp_Lake"],
           "SE": ["Ekoln_basin_of_Malaren", "Erken"],
           "UK": ["Esthwaite_Water", "Windermere"],
           "IE": ["Feeagh"],
           "FI": ["Kilpisjarvi", "Kuivajarvi", "Paajarvi"],
           "IL": ["Kinneret"],
           "RW": ["Kivu"],
           "CZ": ["Klicava", "Rimov", "Zlutice"],
           "NO": ["Langtjern"],
           "RU": ["Mozaisk", "Vendyurskoe"],
           "DE": ["Mueggelsee", "Rappbode_Reservoir", "Stechlin"],
           "CN": ["Ngoring"],
           "EE": ["Nohipalo_Mustjarv", "Nohipalo_Valgejarv", "Vortsjarv"],
           "ES": ["Sau_Reservoir"],
           "NZ": ["Rotorua", "Tarawera", "Taupo", "Waahi"]}
#
models = ["EWEMBI","GFDL-ESM2M","MIROC5", "IPSL-CM5A-LR","HadGEM2-ES",]

scenarios = ["rcp60","rcp26","rcp85","piControl","historical",]

#models = ["IPSL-CM5A-LR"]#,"HadGEM2-ES","GFDL-ESM2M"]

#scenarios = ["piControl"]#,"rcp60"]


input_variables = ["hurs",
                   "pr",
                   "ps",
                   "rsds",
                   "sfcWind",
                   "tas"
                    ]
output_variables = [ "turbdiffheat.csv", "sedheatf.csv","albedo.csv","ice.csv", "watertemp.csv","icetick.csv","strat.csv",  "thermodepth.csv","snowtick.csv", "sensheatf.csv", "latentheatf.csv", "lakeheatf.csv"]

output_variables_depths = [ "watertemp.csv",
                    "turbdiffheat.csv", "sedheatf.csv"]
output_unit={"strat.csv":"unitless", "watertemp.csv":"K", "thermodepth.csv":"m", "ice.csv":"unitless",
             "icetick.csv":"m","snowtick.csv":"m", "sensheatf.csv":"W m-2",
             "latentheatf.csv":"W m-2", "lakeheatf.csv":"W m-2", "albedo.csv":"unitless",
             "turbdiffheat.csv":"m2 s-1", "sedheatf.csv":"W m-2"}
params_0 = np.array([0, 0.3, 0.55, 1, 0, 2.5, 1])
report = 'report.txt'
grid = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\grid.txt"
output = r"D:/output"
output1 = r"D:/output"

def findYPoint(xa,xb,ya,yb,xc):
    m = ((ya) - (yb)) / ((xa) -( xb))
    yc = (float(xc) - (float(xb))) * m + (yb)
    return yc
def performance_analysis(lake_name,  output_folder):
    """
    Opens the comparison file created by make_comparison_file, and prints the results of analysis functions.

    :param
    :param
    :param output_folder: A string, containing the folder containing the comparison file.
    :return: Score, a float representing the overall performance of the current simulation.
    """
    prefix = lake_name[:3]

    if lake_name == "Nohipalo_Valgejarv":
        prefix = 'NoV'
    elif lake_name == "Nohipalo_Mustjarv":
        prefix = 'NoM'
    elif lake_name == "Crystal_Bog":
        prefix = 'CrB'
    elif lake_name == "Great_Pond":
        prefix = 'GrP'
    elif lake_name == "Trout_Bog":
        prefix = 'TrB'
    elif lake_name == "Mt_Bold":
        prefix = 'MtB'
    file = prefix

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

    sos1 = myLake_post.sums_of_squares(obs_list_1, sims_list_1)
    sos2 = myLake_post.sums_of_squares(obs_list_2, sims_list_2)
    rms1 = myLake_post.root_mean_square(obs_list_1, sims_list_1)
    rms2 = myLake_post.root_mean_square(obs_list_2, sims_list_2)
    r_squ1 = myLake_post.r_squared(obs_list_1, sims_list_1)
    r_squ2 = myLake_post.r_squared(obs_list_2, sims_list_2)

    """if r_squ1 < 0:
        r_squ1_B = -r_squ1
    else: r_squ1_B = r_squ1
    if r_squ2 < 0:
        r_squ2_B = -r_squ2
    else: r_squ2_B = r_squ2"""

    score1 = (sos1 + sos2) #+ (rms1 + rms2) * 1000  + (1 - r_squ1_B) * 100 + (1 - r_squ2_B) * 100
    score = []
    #score.append("Analysis of {}.".format(output_folder[10:]))
    score.append(["Sums of squares" ,"%s %s"%(sos1, sos2)])
    score.append(["RMSE","%s %s"%(rms1, rms2)])
    score.append(["R squared","%s %s"%(r_squ1, r_squ2)])

    score.append(["RMSE/SD","%s %s"%(myLake_post.rmse_by_sd(obs_list_1, rms1), myLake_post.rmse_by_sd(obs_list_2, rms2))])
    score.append(["Score",score1])

    return score
from subprocess import Popen, PIPE
def plot_calibrations_par():
    for lake in full_lake_list:
        if not lake in full_lake_list_done:
            plot_calibrations(lake)

def plot_calibrations(lake):
    """
    Intermediary function to call nelder-mead optimisation function for a single lake.

    :param lake: Type string. The name of the lake to calibrate.
    :return: If Calibration_Complete file is found, returns None. Else, return the nelder-mead optimisation function
    from myLake_post module for the given lake.
    """
    print(lake)

    file = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"
    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    if reg == None:
        print("Cannot find {}'s region".format(lake))
        return None
    else:
        with open("{}/observations/{}/{}/{}_hypsometry_modified.csv".format(file,reg, lake, lake)) as obs:
            reader = list(csv.reader(obs))

    optimisation_completed_unsuccessfull = []
    prefix = lake[:3]
    if lake == "Nohipalo_Valgejarv":
        prefix = 'NoV'
    elif lake == "Nohipalo_Mustjarv":
        prefix = 'NoM'
    elif lake == "Crystal_Bog":
        prefix = 'CrB'
    elif lake == "Great_Pond":
        prefix = 'GrP'
    elif lake == "Trout_Bog":
        prefix = 'TrB'
    elif lake == "Mt_Bold":
        prefix = 'MtB'
    for region in regions:
        if lake in regions[region]:

            if os.path.exists("D:\output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(region, lake)):
                print("Calibration for {} is already complete.\n".format(lake))
                folder = "D:\output\{}\{}\EWEMBI\historical".format(region,lake)
                if lake == "Green_Lake":
                    y1,y2 = 2000, 2017
                elif lake == "Laramie_Lake":
                    y1,y2 = 2010, 2017
                elif lake == "Alqueva":
                    y1,y2 = 2015, 2020
                elif lake == "Rappbode_Reservoir":
                    y1,y2 = 2012, 2017
                else:
                    y1,y2 = 1979, 2017
                sdate = date(y1,1,1) # start date
                edate = date(y2, 1, 1)  # end date

                delta = edate - sdate  # as timedelta
                sims = []
                for i in range(delta.days):
                    day = sdate + timedelta(days=i)
                    sims.append(day)

                print(len(sims))
                data_comparison = pd.DataFrame(columns=["Date","Depth","Obs","Sim"])
                data_comparison["Date"] = sims

                data_comparison['Date1'] = pd.to_datetime(data_comparison['Date'])
                obs = pd.read_csv("{}/Observed_Temperatures.csv".format(folder))
                obs = obs.replace('None', np.nan)
                tzt = pd.read_csv("{}/tzt.csv".format(folder),header=None)
                date1 = []

                for datee in obs['Date']:
                    da = str(datee)
                    d = date(int(da[0:4]),int(da[4:6]),int(da[6:]))
                    date1.append(d)
                colonne = [x for x in obs.columns]
                depths = ["Date"]
                for data in obs.columns[1:]:
                    depths.append("obs_%s"%(float(data)) )
                obs.columns = depths
                obs['Date1'] = date1
                obs['Date1'] = pd.to_datetime(obs['Date1'])
                obs = obs.set_index('Date1')
                obs = obs.drop(['Date'], axis=1)
                depths = ["sim_%s"%(float(data) + 0.5) for data in tzt.columns]

                print(obs)
                for depth in colonne:
                    if depth == "Date":
                        colonne.remove(depth)
                        depth = colonne[0]

                    if not float(depth) >= 0.5 or not float(depth) <= float(tzt.columns[-1])+0.5:

                        obs = obs.drop(['obs_%s'%(float(depth))], axis=1)
                        colonne.remove(depth)
                print(colonne)
                colonnetzt = [x+0.5 for x in tzt.columns]
                colonneobs = [float(x) for x in colonne]
                comparisontzt = pd.DataFrame(columns=['Date'])
                comparisontzt['Date'] = sims
                xx = 0
                for depthobs in colonneobs:
                    for depthtzt in colonnetzt:
                        depthtzt = depthtzt+xx
                        if colonnetzt[-1] < depthtzt:
                            break
                        elif colonnetzt[-1] < depthobs:
                            break
                        elif depthtzt == depthobs:
                            comparisontzt['sim_%s' % (depthtzt)] = tzt.iloc[:, int(depthtzt - 0.5)]
                            xx+=1
                            break
                        elif depthtzt < depthobs and depthobs < (depthtzt+1):
                            try:
                                ya = tzt.iloc[:, int(depthtzt - 0.5)]
                                yb = tzt.iloc[:, int(depthtzt + 0.5)]
                                yc = findYPoint(depthtzt,depthtzt+1,ya,yb,depthobs)
                                comparisontzt['sim_%s' % (depthobs)] = yc
                                xx+=1
                                break
                            except:print("tet")

                # #for depth in
                # for depth in colonnetzt:
                #     if (depth) in colonneobs:
                #
                #     elif (depth-0.5) in colonneobs:
                #         yb = tzt.iloc[:,int(depth-0.5-1)]
                #         ya = tzt.iloc[:,int(depth-0.5)]
                #         m = yb - ya
                #         m2 = (depth-0.5 - (depth + 0.5)) * m
                #         yc =  m2 + tzt.iloc[:,int(depth-0.5)]
                #         comparisontzt['sim_%s'%(depth-0.5)] = yc



                comparisontzt['Date'] = pd.to_datetime(comparisontzt['Date'])
                comparisontzt = comparisontzt.set_index('Date')
                result = pd.concat([comparisontzt, obs], axis=1, join='inner')
                print(result)
                final_data = pd.DataFrame(columns=['obs', 'sim', 'depth'])
                for depth in colonneobs:
                    try:
                        temp_data= pd.DataFrame(columns=['obs','sim','depth'])
                        temp_data['obs'] = result.loc[:,'obs_%s'%depth]
                        temp_data['obs'] = temp_data['obs'].astype(float)
                        temp_data['sim'] = result.loc[:,'sim_%s'%depth]
                        temp_data['depth'] = depth * -1
                        temp_data.reset_index(inplace=True)
                        final_data = final_data.append(temp_data)
                    except:
                        print('problem')
                final_data.to_csv("%s/all_data_comparison.csv"%folder)
                x = final_data.loc[:,'sim']
                y = final_data.loc[:,'obs']

                fig, ax = plt.subplots(figsize=(12,10))
                plot = ax.scatter(x,y,c=final_data.loc[:,'depth'], s=50, cmap= "bone", alpha=0.8, marker="o")

                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                ]

                # now plot both limits against eachother
                ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
                ax.set_aspect('equal')
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.set_ylabel("Observed T")
                ax.set_xlabel("Simulated T")
                table_data = performance_analysis(lake, folder)
                plt.title("%s %s \n %s \n %s \n %s \n %s \n %s"%(lake, region, table_data[0],table_data[1],table_data[2],table_data[3],table_data[4]))
                fig.colorbar(plot, label= "Depth(m)")
                plt.savefig(r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\figure_%s.png"%(lake))


                #plt.show()
                print("test")









    print("Cannot find {}'s region".format(lake))

def createpersogrid(file,lake,searchExp,replaceExp,searchExp1,replaceExp1):
    copyfile("%s.txt" % file, "%s_%s.txt" % (file, lake))
    for line in fileinput.input("%s_%s.txt" % (file, lake), inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        if searchExp1 in line:
            line = line.replace(searchExp1, replaceExp1)
        sys.stdout.write(line)

def simulation_years(scenarioid):
    if scenarioid == 'piControl':
        y1, y2 = [1661,1861,2006], [1860,2005,2099]
    elif scenarioid == 'historical':
        y1, y2 = [1861], [2005]
    elif scenarioid == 'rcp26':
        y1, y2 = [2006,2100], [2099,2299]
    else:
        y1, y2 = [2006], [2099]

    return y1, y2

def format_nc4_par():
    index = range(0, len(full_lake_list) * 21 * 10)
    columns = ['lake', 'model', 'scenario', 'Calibration'] + ['%s.txt' % vari[:-4] for vari in output_variables] + [
        '%s.nc4' % vari[:-4] for vari in output_variables]
    tableau = pd.DataFrame(index=index, columns=columns)
    index = 0
    commandall = "{ "
    #for lake in full_lake_list:
    #    format_nc41(lake)
    Parallel(n_jobs=num_cores, verbose=10)(delayed(format_nc41)(lake) for lake in full_lake_list)

def format_nc41(lake):
    #output_variables = ["lakeicefrac.csv"]
    index = range(0, len(full_lake_list) * 21 * 10)
    columns = ['lake', 'model', 'scenario','date', 'Calibration'] + ['%s.txt' % vari[:-4] for vari in output_variables] + [
        '%s.nc4' % vari[:-4] for vari in output_variables]
    tableau = pd.DataFrame(index=index, columns=columns)

    tableau2 = [['lake', 'model', 'scenario','variable',"file","levlak_min","levlak_max",'lat','lon','time_start','time_end','variable_min',"variable_max"]]

    index = 0
    commandall = "{ "
    #for lake in full_lake_list:
    if 1==1:
        corrected_names = ["Allequash_Lake", "Big_Muskellunge_Lake", "Black_Oak_Lake", "Burley_Griffin", "Crystal_Bog",
                           "Crystal_Lake",
                           "Dickie_Lake", "Eagle_Lake", "Ekoln_basin_of_Malaren", "Esthwaite_Water",
                           "Falling_Creek",
                           "Fish_Lake", "Great_Pond", "Green_Lake", "Harp_Lake", "Laramie_Lake", "Lower_Zurich",
                           "Mt_Bold",
                           "Nohipalo_Mustjarv", "Nohipalo_Valgejarv", "Okauchee_Lake", "Rappbode_Reservoir",
                           "Sau_Reservoir",
                           "Sparkling_Lake", "Toolik_Lake", "Trout_Bog", "Trout_Lake", "Two_Sisters_Lake"]

        f_lake = lake


        reg = None
        for region in regions:
            if lake in regions[region]:
                reg = region
                break
        output = r"D:/output/%s/%s" % (reg, lake)

        outputnc = r"D:\final_files"

        for modelid in models:
            for scenarioid in scenarios:

                print(lake, modelid, scenarioid)
                # path = os.path.join(output, "%s/%s/%s/%s/RunComplete" % (reg, lake, modelid, scenarioid))
                if os.path.exists(os.path.join(output, "%s/%s/RunComplete1" % ( modelid, scenarioid))):
                    tableau.loc[index, 'lake'] = f_lake
                    tableau.loc[index, 'model'] = modelid
                    tableau.loc[index, 'scenario'] = scenarioid

                    tableau.loc[index, 'Calibration'] = 'Done'
                    if (modelid == "EWEMBI" and scenarioid == "historical") or modelid != "EWEMBI":

                        if modelid == "EWEMBI":
                            y11, y21 = [1979], [2016]

                        elif modelid == "GFDL-ESM2M" and scenarioid == 'piControl':
                            y11, y21 = [1661, 1861, 2006], [1860, 2005, 2099]
                        elif modelid == "GFDL-ESM2M" and scenarioid == 'rcp26':
                            y11, y21 = [2006], [2099]
                        elif modelid == "IPSL-CM5A-LR" and scenarioid == 'rcp85':
                            y11, y21 = [2006, 2100], [2099, 2299]
                        else:
                            y11, y21 = simulation_years(scenarioid)
                        startyear = y11[0]
                        nbrfile = len(y11)
                        for vari in output_variables:
                            for i in range(0,nbrfile):
                                y1,y2 = y11[i],y21[i]
                                listtableau2 = []

                                # if os.path.exists(os.path.join(output, "%s/%s/%s/%s" % (reg, lake, modelid, scenarioid), vari)):
                                #    data = pd.read_csv(os.path.join(output, "%s/%s/%s/%s" % (reg, lake, modelid, scenarioid), vari),
                                #                       header=None)
                                variable = vari[:-4]
                                model_name = "MyLake"
                                gcm_observation = "GCM"
                                bias = modelid
                                climate = scenarioid
                                socio = "nosoc"
                                sens = "co2"
                                region = "local"
                                timestep = "daily"
                                unit = output_unit.get(vari)
                                increment = "day"
                                searchstart = 0
                                searchend = 0
                                end = False
                                if bias == "HadGEM2-ES" and climate == "rcp26":
                                    print("stop")
                                y3,y4=y1, y2
                                # for y3 in range((math.floor(y1 / 10) * 10) + 1, (math.floor(y2 / 10) * 10) + 11, 10):
                                #     y4 = y3 + 9
                                #     if y1 > y3:
                                #         y3 = y1
                                #     if y2 < y4:
                                #         y4 = y2

                                if 1==1:#if y2 - y1 > 100:

                                    file_name = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                        model_name, bias, climate, socio, sens, variable, region, timestep, y3, y4)

                                    file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                        model_name, bias, climate, socio, sens, variable, lake, timestep, y3, y4)
                                    if variable == "lakeicefrac":
                                        file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                            model_name, bias, climate, socio, sens, "icetick", lake, timestep, y3, y4)
                                        unit= "m"

                                    print(os.path.join(output, file_name))
                                    file_namel = file_name2.lower()
                                    if bias == "HadGEM2-ES" and climate=="rcp26":
                                        print("stop")
                                    #run_myLake_ISIMIP.outputfile(y3, y4, r"D:\output\US\Mendota\EWEMBI\historical")
                                    if os.path.exists(os.path.join(output, "%s.txt" % file_name)) or os.path.exists(os.path.join(outputnc, "%s.nc4" % (file_namel))) or os.path.exists(os.path.join(outputnc, "%s.nc" % (file_namel))):
                                        print("a;;p")
                                                        #command = comm.split(' ')
                                                       # if os.path.exists(os.path.join(output, "%s.txt" % file_name)):
                                                       #     tableau.loc[index, '%s.txt' % variable] = 'Exists'

                                                            #if not os.path.exists(os.path.join(outputnc, "%s.nc4" % (file_namel))) and not os.path.exists(os.path.join(outputnc, "%s.nc" % (file_namel))) :
                                                                #if 1==1:#try:
                                                                    # creation = txt_to_netcdf.netcdf(output, "%s.txt"%( file_name), "%s.nc4"%file_namel, y3,y4, float(run_myLake_ISIMIP.get_latitude(lake, "D:/forcing_data", "EWEMBI",
                                                                    #                             "historical")), float(run_myLake_ISIMIP.get_longitude(lake, "D:/forcing_data", "EWEMBI",
                                                                    #                             "historical")),unit, variable)
                                                                    # if creation == 1:
                                                                    #     tableau.loc[index, '%s.nc4' % variable] = 'Exists'
                                                                    #     if os.path.exists(os.path.join(output, "%s.txt" % file_name)):
                                                                    #         os.remove(os.path.join(output, "%s.txt" % file_name))
                                                                    # else:
                                                                    #     tableau.loc[index, '%s.nc4' % variable] = 'Not created yet'

                                                                #except:
                                                                #     tableau.loc[index, '%s.nc4' % variable] = 'Bug Not created yet'

                                                                # print(command)
                                                                # commandall = commandall + command.lower() + " && "
                                                                # #os.system(command)

                                                        # else:
                                                        #     tableau.loc[index, '%s.txt' % variable] = 'doesnt Exist'
                                                        #     tableau.loc[index, '%s.nc4' % variable] = 'Exists'

                                                    # else:
                                                    #     tableau.loc[index, '%s.txt' % variable] = 'Not created'

                                                    #if os.path.exists(os.path.join(output, "%s.txt" % file_name)) and (os.path.exists(os.path.join(outputnc, "%s.nc4" % (file_namel))) or os.path.exists(os.path.join(outputnc, "%s.nc" % (file_namel)))):
                                                    #    os.remove(os.path.join(output, "%s.txt" % file_name))

                                    if 1==1:# os.path.exists(
                                            #os.path.join(outputnc, "%s.nc4" % (file_namel))) or os.path.exists(
                                            #os.path.join(outputnc, "%s.nc" % (file_namel))):

                                        listtableau2.append(f_lake)
                                        listtableau2.append(modelid)
                                        listtableau2.append(scenarioid)
                                        if variable == "lakeicefrac":
                                            variable = "icetick"
                                        listtableau2.append(variable)
                                        if os.path.exists("observations/{}/{}/{}_hypsometry_modified.csv".format(reg, lake, lake)):
                                            hypso = pd.read_csv("observations/{}/{}/{}_hypsometry_modified.csv".format(reg, lake, lake))
                                            maxdepth = len(hypso)

                                        validationresult = txt_to_netcdf.validationnetcdf("%s.nc4"%file_namel, variable, run_myLake_ISIMIP.get_latitude(lake, "D:/forcing_data","EWEMBI","historical"),  run_myLake_ISIMIP.get_longitude(f_lake, "D:/forcing_data","EWEMBI","historical"), maxdepth, y3, y4)
                                        for result in validationresult:
                                            listtableau2.append(result)
                                        tableau2.append(listtableau2)

                                # except:
                                #     print("bug")



                else:
                    tableau.loc[index, 'lake'] = f_lake
                    tableau.loc[index, 'model'] = modelid
                    tableau.loc[index, 'scenario'] = scenarioid
                    tableau.loc[index, 'Calibration'] = 'Not Done'

                index += 1



    # tableau.to_csv(r"all_variable_lakes_combinaison_update121.csv", index=False)
    # with open(r"validation_result2020.csv", 'w', newline='') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(tableau2)
    if len(tableau2) != 1:
        df = pd.DataFrame.from_records(tableau2)
        df.to_csv("validation_list%s.csv"%lake, index=False)
    else:
        df = pd.DataFrame.from_records(tableau2)
        df.to_csv("validation_list_empty%s.csv" % lake, index=False)

def format_nc4(lake):
    #output_variables = ["lakeicefrac.csv"]
    index = range(0, len(full_lake_list) * 21 * 10)
    columns = ['lake', 'model','date', 'scenario', 'Calibration'] + ['%s.nc4' % vari[:-4] for vari in output_variables]
    tableau = pd.DataFrame(index=index, columns=columns)

    tableau2 = [['lake', 'model', 'scenario','variable',"file","levlak_min","levlak_max",'lat','lon','time_start','time_end','variable_min',"variable_max"]]

    index = 0
    commandall = "{ "
    for lake in full_lake_list:
    #if 1==1:
        corrected_names = ["Allequash_Lake", "Big_Muskellunge_Lake", "Black_Oak_Lake", "Burley_Griffin", "Crystal_Bog",
                           "Crystal_Lake",
                           "Dickie_Lake", "Eagle_Lake", "Ekoln_basin_of_Malaren", "Esthwaite_Water",
                           "Falling_Creek",
                           "Fish_Lake", "Great_Pond", "Green_Lake", "Harp_Lake", "Laramie_Lake", "Lower_Zurich",
                           "Mt_Bold",
                           "Nohipalo_Mustjarv", "Nohipalo_Valgejarv", "Okauchee_Lake", "Rappbode_Reservoir",
                           "Sau_Reservoir",
                           "Sparkling_Lake", "Toolik_Lake", "Trout_Bog", "Trout_Lake", "Two_Sisters_Lake"]

        f_lake = lake


        reg = None
        for region in regions:
            if lake in regions[region]:
                reg = region
                break
        output = r"D:/output/%s/%s" % (reg, lake)

        outputnc = r"D:\final_files"

        for modelid in models:
            if modelid != "EWEMBI":
                for scenarioid in scenarios:

                    print(lake, modelid, scenarioid)
                    # path = os.path.join(output, "%s/%s/%s/%s/RunComplete" % (reg, lake, modelid, scenarioid))


                    if modelid == "EWEMBI":
                        y11, y21 = [1979], [2016]

                    elif modelid == "GFDL-ESM2M" and scenarioid == 'piControl':
                        y11, y21 = [1661, 1861, 2006], [1860, 2005, 2099]
                    elif modelid == "GFDL-ESM2M" and scenarioid == 'rcp26':
                        y11, y21 = [2006], [2099]
                    elif modelid == "IPSL-CM5A-LR" and scenarioid == 'rcp85':
                        y11, y21 = [2006, 2100], [2099, 2299]
                    else:
                        y11, y21 = simulation_years(scenarioid)
                    startyear = y11
                    subindex = index + 1
                    nbrfile = len(y11)

                    for i in range(0, nbrfile):
                        if os.path.exists(os.path.join(output, "%s/%s/RunComplete1" % (modelid, scenarioid))):
                            tableau.loc[index, 'lake'] = f_lake
                            tableau.loc[index, 'model'] = modelid
                            tableau.loc[index, 'scenario'] = scenarioid

                            tableau.loc[index, 'Calibration'] = 'Done'
                        else:
                            tableau.loc[index, 'lake'] = f_lake
                            tableau.loc[index, 'model'] = modelid
                            tableau.loc[index, 'scenario'] = scenarioid
                            tableau.loc[index, 'Calibration'] = 'not Done'
                        y1, y2 = y11[i], y21[i]
                        tableau.loc[index, 'date'] = '%s_%s'%(y1,y2)
                        for vari in output_variables:


                            # if 1==1:#try:
                            variable = vari[:-4]
                            model_name = "MyLake"
                            bias = modelid
                            climate = scenarioid
                            socio = "nosoc"
                            sens = "co2"
                            region = "local"
                            timestep = "daily"
                            unit = output_unit.get(vari)
                            increment = "day"
                            searchstart = 0
                            searchend = 0
                            end = False
                            file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                y2)

                            file_name = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                    model_name, bias, climate, socio, sens, variable, region, timestep, y1, y2)


                            print(os.path.join(output, file_name))
                            file_namel = file_name2.lower()

                            if os.path.exists(os.path.join(output, "%s.txt" % file_name)):
                                #tableau.loc[index, '%s.txt' % variable] = 'Exists'

                                if  os.path.exists(os.path.join(outputnc,"%s.nc4"%file_namel)):
                                    tableau.loc[index, '%s.nc4' % variable] = 'Exist'
                                elif os.path.exists(os.path.join(outputnc, "%s.nc" % (file_namel))):
                                    tableau.loc[index, '%s.nc4' % variable] = 'Exist'
                                else:
                                    tableau.loc[index, '%s.nc4' % variable] = 'Not created yet'

                            else:
                                #tableau.loc[index, '%s.txt' % variable] = 'doesnt Exist'

                                if  os.path.exists(os.path.join(outputnc,"%s.nc4"%file_namel)):
                                    tableau.loc[index, '%s.nc4' % variable] = 'Exist'
                                elif os.path.exists(os.path.join(outputnc, "%s.nc" % (file_namel))):
                                    tableau.loc[index, '%s.nc4' % variable] = 'Exist'
                                else:
                                    tableau.loc[index, '%s.nc4' % variable] = 'Not created yet'
                        index += 1

    tableau.to_csv(r"all_variable_lakes_combinaison_update2020.csv", index=False)

def format_parallel():
    for lake in full_lake_list:
       format(lake)

    # full_lake_list = ['Annie', 'Allequash_Lake', 'Alqueva', 'Annecy', 'Argyle',
    #                   'Biel', 'Big_Muskellunge_Lake', 'Black_Oak_Lake', 'Bourget', 'Burley_Griffin',
    #                   'Crystal_Bog', 'Crystal_Lake', 'Delavan', 'Dickie_Lake', 'Eagle_Lake',
    #                   'Ekoln_basin_of_Malaren', 'Erken', 'Esthwaite_Water', 'Falling_Creek_Reservoir', 'Feeagh',
    #                   'Fish_Lake', 'Geneva', 'Great_Pond', 'Green_Lake', 'Harp_Lake',
    #                   'Kilpisjarvi', 'Kinneret', 'Kivu', 'Kuivajarvi','Langtjern',
    #                   'Laramie_Lake', 'Lower_Zurich','Mendota', 'Monona', 'Mozaisk',
    #                   'Mt_Bold', 'Mueggelsee', 'Neuchatel', 'Nohipalo_Mustjarv',
    #                   'Nohipalo_Valgejarv', 'Okauchee_Lake', 'Paajarvi', 'Rappbode_Reservoir', 'Rimov',
    #                   'Rotorua', 'Sammamish', 'Sau_Reservoir', 'Sparkling_Lake', 'Stechlin',
    #                   'Sunapee', 'Tahoe', 'Tarawera', 'Toolik_Lake', 'Trout_Bog',
    #                   'Trout_Lake', 'Two_Sisters_Lake', 'Vendyurskoe', 'Vortsjarv', 'Washington',
    #                   'Windermere', 'Wingra']
    #Parallel(n_jobs=num_cores, verbose=10)(delayed(format)(lake) for lake in full_lake_list)

def format(lake):

    #output_variables = ["lakeicefrac.csv"]
    index = range(0, len(full_lake_list) * 21 * 10)
    columns = ['lake', 'model', 'scenario', 'Calibration', "Date"] + [vari[:-4] for vari in output_variables]
    tableau = pd.DataFrame(index=index, columns=columns)
    index = 0
    # for lake in (full_lake_list):
    f_lake = lake

    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    if reg != None:
        for modelid in models:
            for scenarioid in scenarios:
                if  modelid != "EWEMBI":
                    print(lake, modelid, scenarioid)

                    if os.path.exists(os.path.join(output, "%s/%s/%s/%s" % (reg, lake, modelid, scenarioid))):
                        if os.path.exists(
                                os.path.join(output, "%s/%s/%s/%s/RunComplete1" % (reg, lake, modelid, scenarioid))):

                            if modelid == "EWEMBI":
                                y11, y21 = [1979], [2016]

                            elif modelid == "GFDL-ESM2M" and scenarioid == 'piControl':
                                y11, y21 = [1661,1861,2006], [1860,2005,2099]
                            elif modelid == "GFDL-ESM2M" and scenarioid == 'rcp26':
                                y11, y21 = [2006], [2099]
                            elif modelid == "IPSL-CM5A-LR" and scenarioid == 'rcp85':
                                y11, y21 = [2006,2100], [2099,2299]
                            else:
                                y11, y21 = simulation_years(scenarioid)
                            startyear = y11
                            subindex = index + 1
                            nbrfile = len(y11)
                            for vari in output_variables:
                                for i in range(0,nbrfile):
                                    y1,y2 = y11[i],y21[i]


                                    #if 1==1:#try:
                                    variable = vari[:-4]
                                    model_name = "MyLake"
                                    bias = modelid
                                    climate = scenarioid
                                    socio = "nosoc"
                                    sens = "co2"
                                    region = "local"
                                    timestep = "daily"
                                    unit = output_unit.get(vari)
                                    increment = "day"
                                    searchstart = 0
                                    searchend = 0
                                    end = False
                                    file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                        model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                        y2)

                                    file_namel = file_name2.lower()
                                    #outputnc = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"
                                    outputnc = r"D:\final_files"
                                    if not (os.path.exists(os.path.join(r"D:\ready","%s.nc4" % file_namel))) and not (os.path.exists(os.path.join(r"D:\ready","%s.nc" % file_namel))) and \
                                        not (os.path.exists(os.path.join(r"D:\move","%s.nc" % file_namel)))and not (os.path.exists(os.path.join(r"D:\move","%s.nc4" % file_namel)))and \
                                        not (os.path.exists(os.path.join(r"D:\remove","%s.nc" % file_namel)))and not (os.path.exists(os.path.join(r"D:\remove","%s.nc4" % file_namel))):
                                        if not(os.path.exists(os.path.join("%s.nc4" % file_namel))) and not(os.path.exists(os.path.join("%s.nc" % file_namel))):
                                                outputnc = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"
                                                if y21[-1] - y11[0] > 100:

                                                    data_set = pd.DataFrame()

                                                    if y1 == 1661:
                                                        yinit1 = [1661,1761]
                                                        yend1 = [1760,1860]
                                                    elif y1 == 1861:
                                                        if y21[-1] == 2005:
                                                            yinit1 = [1861, 1961]
                                                            yend1 = [1960, 2005]
                                                        else:
                                                            yinit1 = [1861,1961]
                                                            yend1 = [1960,2060]
                                                    elif y1 == 2006:
                                                        if y21[-1] == 2099:
                                                            if i!= 0:
                                                                yinit1 = [1961,2061]
                                                                yend1 = [2060,2099]

                                                        elif y21[-1] == 2299:
                                                            if y11[0] == 2006:
                                                                yinit1 = [2006,2011]
                                                                yend1 = [2010,2110]
                                                            elif y11[0] == 1661:
                                                                yinit1=[1961,2061]
                                                                yend1=[2060,2160]
                                                    elif y1 == 2100:
                                                        if y11[0] == 2006:
                                                            yinit1 = [2011,2111,2211]
                                                            yend1 = [2110,2210,2299]
                                                        elif y11[0] == 1661:
                                                            yinit1=[2061,2161,2261]
                                                            yend1=[2160,2260,2299]

                                                    for j in range(0,len(yinit1)):
                                                        yinit = yinit1[j]
                                                        yend = yend1[j]
                                                        outputdir = os.path.join(output, "%s/%s/%s/%s/%s_%s" % (
                                                                reg, lake, modelid, scenarioid, yinit, yend))


                                                        if not os.path.exists(os.path.join(outputdir, vari)) and os.path.exists(os.path.join(outputdir,'Tzt.csv')):
                                                            run_myLake_ISIMIP.outputfile(yinit, yend, outputdir)
                                                        # elif not os.path.exists(os.path.join(outputdir,'Tzt.csv')):
                                                        #     run_myLake_ISIMIP.run_myLake( os.path.join(output, "%s/%s/%s/%s" % (reg, lake, modelid, scenarioid)), reg, lake, modelid, scenarioid)

                                                        ###Main program###
                                                        print(os.path.join(outputdir, vari))
                                                        if os.path.exists(os.path.join(outputdir, vari)):
                                                            #data = pd.read_csv(os.path.join(outputdir, vari), header=None)

                                                            y3, y4 = yinit, yend

                                                            tableau.loc[subindex, 'lake'] = f_lake
                                                            tableau.loc[subindex, 'model'] = modelid
                                                            tableau.loc[subindex, 'scenario'] = scenarioid
                                                            tableau.loc[subindex, 'Calibration'] = "Done"
                                                            tableau.loc[subindex, 'Date'] = "%s_%s" % (y3, y4)
                                                            file_name = "%s/%s/%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                                reg, lake, model_name, bias, climate, socio, sens,
                                                                variable, region, timestep, y1, y2)
                                                            outdir1 = os.path.join(output1, "%s/%s/" % (reg, lake))
                                                            file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                                model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                                                y2)

                                                            #file_namel = file_name2.lower()
                                                            outputnc = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"

                                                            if not (os.path.exists(os.path.join("%s.nc4" % file_name2.lower())))or not(os.path.exists(os.path.join("%s.nc" % file_name2.lower()))):

                                                                data = pd.read_csv(os.path.join(outputdir,vari),header=None)
                                                                #data = data.fillna("1.e+20f")

                                                                if data.iloc[0,0] != '%s-01-01, 00:00:00'%y3:
                                                                    r = range((datetime(y4, 12, 31) - datetime(y3, 1, 1)).days + 1)
                                                                    dates = [datetime(y3, 1, 1) + timedelta(days=x)for x in r]
                                                                    data[0] = pd.Series(dates)
                                                                if y3 >= y1:
                                                                    if y4 <= y2:
                                                                        data = data.set_index(data[0])
                                                                        if y3==y1:
                                                                            data_set= data[0:]
                                                                        else:
                                                                            data_set = data_set.append(data[0:],ignore_index=True)
                                                                            data_set = data_set.set_index(data_set[0])

                                                                    else:
                                                                        data = data.set_index(data[0])
                                                                        data = data.loc[:'%s-01-01'%(y2+1)]
                                                                        data_set = data_set.append(data,ignore_index=True)
                                                                        data_set = data_set.set_index(data_set[0])
                                                                else:
                                                                    data = data.set_index(data[0])
                                                                    data_set=data.loc['%s-01-01'%(y1):]




                                                        else:
                                                            print(os.path.join(outputdir, vari, " doesn't exist"))
                                                            tableau.loc[index, 'lake'] = f_lake
                                                            tableau.loc[index, 'model'] = modelid
                                                            tableau.loc[index, 'scenario'] = scenarioid
                                                            tableau.loc[index, 'Calibration'] = "Done"
                                                            tableau.loc[index, vari[:-4]] = "csv is missing"
                                                            break


                                                else:
                                                    data_set = pd.DataFrame()


                                                    outputdir = os.path.join(output, "%s/%s/%s/%s" % (reg, lake, modelid, scenarioid))

                                                    if not os.path.exists(os.path.join(outputdir, vari)) and os.path.exists(os.path.join(outputdir, 'Tzt.csv')):
                                                        run_myLake_ISIMIP.outputfile(y1, y2, outputdir)
                                                    ###Main program###
                                                    if os.path.exists(os.path.join(outputdir, vari)):
                                                        file_name = "%s/%s/%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                            reg, lake, model_name, bias, climate, socio, sens,
                                                            variable, region, timestep, y1, y2)
                                                        outdir1 = os.path.join(output1, "%s/%s/" % (reg, lake))
                                                        file_name2 = "%s_%sfile_name2%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                            model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                                            y2)

                                                        file_namel = file_name2.lower()
                                                        outputnc = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"

                                                        if not(os.path.exists(os.path.join("%s.nc4" % file_namel)))or not(os.path.exists(os.path.join("%s.nc" % file_namel))):
                                                            data = pd.read_csv(os.path.join(outputdir, vari), header=None)
                                                            data = data.fillna(method='ffill')
                                                            y3, y4 = y1, y2

                                                            tableau.loc[subindex, 'lake'] = f_lake
                                                            tableau.loc[subindex, 'model'] = modelid
                                                            tableau.loc[subindex, 'scenario'] = scenarioid
                                                            tableau.loc[subindex, 'Calibration'] = "Done"
                                                            tableau.loc[subindex, 'Date'] = "%s_%s" % (y3, y4)


                                                            data = data.set_index(data[0])
                                                            data_set = data_set.append(data[0:])


                                                    else:
                                                        print(os.path.join(outputdir, vari, " doesn't exist"))
                                                        tableau.loc[index, 'lake'] = f_lake
                                                        tableau.loc[index, 'model'] = modelid
                                                        tableau.loc[index, 'scenario'] = scenarioid
                                                        tableau.loc[index, 'Calibration'] = "Done"
                                                        tableau.loc[index, vari[:-4]] = "csv is missing"
                                                        break


                                                file_name = "%s/%s/%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                    reg, lake, model_name, bias, climate, socio, sens,
                                                    variable, region, timestep, y1, y2)

                                                outdir1 = os.path.join(output1, "%s/%s/" % (reg, lake))

                                                if variable == "lakeicefrac":
                                                    variable = "icetick"
                                                file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                    model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                                    y2)

                                                file_namel = file_name2.lower()
                                                outputnc = r"D:\final_files"

                                                if (not os.path.exists(os.path.join(output1, "%s.txt" % file_name))):

                                                    if not os.path.exists(outdir1):
                                                        os.makedirs(outdir1)
                                                    if len(data_set) != 0:
                                                        if len(data_set) == ((datetime(y2, 12, 31) - datetime(y1, 1, 1)).days + 1):
                                                            data_set.to_csv(os.path.join(output1, "%s.txt" % file_name),header=None, index=None, sep=' ', mode='w')
                                                        else:
                                                            print('problem!')
                                                            #x = 5 / 0

                                                    else:
                                                        print("data is empty!!!")
                                                        #x=5/0

                                                    tableau.loc[subindex, vari[:-4]] = "Done"



                                                if os.path.exists( os.path.join(output1, "%s.txt" % (file_name))):
                                                    if (not os.path.exists(os.path.join(r"D:\ready", "%s.nc4" % (file_namel))) and not os.path.exists(os.path.join(r"D:\ready", "%s.nc" % (file_namel)))) and \
                                                            (not os.path.exists(os.path.join(r"D:\final_files", "%s.nc4" % (file_namel))) and not os.path.exists(os.path.join(r"D:\final_files", "%s.nc" % (file_namel))))and \
                                                            (not os.path.exists(os.path.join(r"D:\move","%s.nc" % file_namel)) and  not os.path.exists(os.path.join(r"D:\move","%s.nc4" % file_namel)))and \
                                                            (not os.path.exists(os.path.join(r"D:\remove","%s.nc" % file_namel)) and  not os.path.exists(os.path.join(r"D:\remove","%s.nc4" % file_namel))):
                                                        lat,lon= float(run_myLake_ISIMIP.get_latitude(lake, "D:/forcing_data", "EWEMBI","historical")), float(run_myLake_ISIMIP.get_longitude(lake, "D:/forcing_data", "EWEMBI","historical"))
                                                        creation = txt_to_netcdf.netcdf(output1, "%s.txt"%( file_name), "%s.nc4"%file_namel, y1,y2, lat, lon,unit, variable)

                                                        if creation == 1:

                                                           if os.path.exists(os.path.join(output1, "%s.txt" % file_name)):
                                                                os.remove(os.path.join(output1, "%s.txt" % file_name))
                                                           if os.path.exists(os.path.join(output1, "%s.txt" % file_name)):
                                                               os.remove(os.path.join(output1, "%s.txt" % file_name))

                                                        if (os.path.exists(os.path.join(r"D:\ready", "%s.nc4" % (file_namel))) or os.path.exists(os.path.join(r"D:\ready","%s.nc" % (file_namel)))) \
                                                                or (os.path.exists(os.path.join(r"D:\final_files","%s.nc4" % (file_namel))) or  os.path.exists(os.path.join(r"D:\final_files","%s.nc" % (file_namel)))) \
                                                                or (os.path.exists(os.path.join(r"D:\move","%s.nc" % file_namel)) or os.path.exists(os.path.join(r"D:\move", "%s.nc4" % file_namel)))\
                                                                or (os.path.exists(os.path.join("%s.nc" % file_namel)) or os.path.exists(os.path.join( "%s.nc4" % file_namel)))\
                                                                or (os.path.exists(os.path.join(r"D:\remove","%s.nc" % file_namel)) or os.path.exists(os.path.join(r"D:\remove", "%s.nc4" % file_namel))):
                                                            if os.path.exists(os.path.join(output1, "%s.txt" % file_name)):
                                                                os.remove(os.path.join(output1, "%s.txt" % file_name))
                                                            if os.path.exists(os.path.join(output1, "%s.txt" % file_name)):
                                                                os.remove(os.path.join(output1, "%s.txt" % file_name))
                                                        # if creation == 0:
                                                        #     if os.path.exists(
                                                        #             r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\%s.nc4" % file_name):
                                                        #         os.remove(
                                                        #             r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\%s.nc4" % file_name)
                                                        #     if os.path.exists(
                                                        #             r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\%s.nc" % file_name):
                                                        #         os.remove(
                                                        #             r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\%s.nc" % file_name)

                                                #if os.path.exists(os.path.join(output1, "%s.txt" % file_name)):
                                                #    if os.path.exists(os.path.join(outputnc, "%s.nc4" % (file_namel))) or os.path.exists(os.path.join(outputnc, "%s.nc" % (file_namel))):
                                                #        os.remove(os.path.join(output1, "%s.txt" % file_name))



                                                else:
                                                    print("file exists")
                                                    tableau.loc[subindex, vari[:-4]] = "Done"



                                                subindex += 1

                                    #except:
                                    #    print("bg!! %s %s %s"%(lake,modelid,scenarioid))

                            index = subindex

                        else:
                            tableau.loc[index, 'lake'] = (f_lake)
                            tableau.loc[index, 'model'] = (modelid)
                            tableau.loc[index, 'scenario'] = (scenarioid)
                            tableau.loc[index, 'Calibration'] = "not Done"
                            index += 1
                    else:
                        print(os.path.join(output1, "%s/%s/%s/%s doesn't exist" % (reg, lake, modelid, scenarioid)))
                        tableau.loc[index, 'lake'] = f_lake
                        tableau.loc[index, 'model'] = modelid
                        tableau.loc[index, 'scenario'] = scenarioid
                        tableau.loc[index, 'Calibration'] = "folder is missing"
                        index += 1

    else:
        print("Lake is not in regions")
    tableau.to_csv(r"all_variable_lakes_combinaisonfinall.csv", index=False)
    format_nc41(lake)

def format_V1(lake):
    # models = [ "EWEMBI",
    #
    #           ]
    # scenarios = ["historical",
    #
    #              ]
    output_variables = ["lakeicefrac.csv"]
    index = range(0, len(full_lake_list) * 21*10)
    columns = ['lake', 'model', 'scenario','Calibration', "Date"]+[vari[:-4] for vari in output_variables ]
    tableau = pd.DataFrame(index=index, columns=columns)
    index = 0
    #for lake in (full_lake_list):
    f_lake = lake

    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    #createpersogrid(r"D:\output_txt\grid", lake, "-179.75", "%s"%(run_myLake_ISIMIP.get_longitude(lake, "D:/forcing_data","EWEMBI","historical")), "89.75", "%s" %(run_myLake_ISIMIP.get_latitude(lake, "D:/forcing_data","EWEMBI","historical")))

    
    if reg != None:
        for modelid in models:
            for scenarioid in scenarios:
                if (modelid == "EWEMBI" and scenarioid == "historical") or modelid != "EWEMBI":
                    print(lake, modelid, scenarioid)

                    if os.path.exists(os.path.join(output, "%s/%s/%s/%s" % (reg, lake, modelid, scenarioid))):
                        if os.path.exists(os.path.join(output, "%s/%s/%s/%s/RunComplete1" % (reg, lake, modelid, scenarioid))):

                            if modelid == "EWEMBI":
                                y1, y2 = 1979, 2016
                            elif modelid == "GFDL-ESM2M" and scenarioid == 'piControl':
                                y1, y2 = 1661, 2099
                            elif modelid == "GFDL-ESM2M" and scenarioid == 'rcp26':
                                y1, y2 = 2006, 2099
                            elif modelid == "IPSL-CM5A-LR" and scenarioid == 'rcp85':
                                y1, y2 = 2006, 2299
                            else:
                                y1, y2 = simulation_years(scenarioid)
                            startyear = y1
                            subindex = index +1

                            if y2 - y1 > 100:
                                data_set = pd.DataFrame()
                                data_set1 = pd.DataFrame()
                                years = [y1]
                                if y1 == 2006:
                                    yinit = 2011
                                else:
                                    yinit = y1 + 100
                                years = years + (list(range(yinit, y2, 100)))
                                years.append(y2)
                                all_files = []
                                yrange = range(0, len(years) - 1)

                                for i in range(0, len(years) - 1):
                                    test = (years[i])
                                    subindex = index

                                    if i + 1 != len(years) - 1:
                                        yinit = years[i]
                                        yend = years[i + 1] - 1

                                        print(yinit, yend)
                                    else:
                                        yinit = years[i]
                                        yend = years[i + 1]
                                        print(yinit, yend)

                                    outputdir = os.path.join(output, "%s/%s/%s/%s/%s_%s" % (reg, lake, modelid, scenarioid, yinit, yend))
                                    run_myLake_ISIMIP.outputfile(yinit, yend, outputdir)

                            else:
                                outputdir = os.path.join(output, "%s/%s/%s/%s" % (
                                reg, lake, modelid, scenarioid))
                                run_myLake_ISIMIP.outputfile(y1, y2, outputdir)

                            for vari in output_variables:

                                variable = vari[:-4]
                                model_name = "MyLake"
                                bias = modelid
                                climate = scenarioid
                                socio = "nosoc"
                                sens = "co2"
                                region = "local"
                                timestep = "daily"
                                unit = output_unit.get(vari)
                                increment = "day"
                                searchstart=0
                                searchend=0
                                end = False

                                if y2 - y1 > 100:

                                    data_set = pd.DataFrame()
                                    data_set1 = pd.DataFrame()
                                    years = [y1]
                                    if y1 == 2006:
                                        yinit = 2011
                                    else:
                                        yinit = y1 + 100
                                    years = years + (list(range(yinit, y2, 100)))
                                    years.append(y2)
                                    all_files = []
                                    yrange = range(0, len(years) - 1)

                                    for i in range(0, len(years) - 1):
                                        test = (years[i])
                                        subindex = index

                                        if i + 1 != len(years) - 1:
                                            yinit = years[i]
                                            yend = years[i + 1] - 1

                                            print(yinit, yend)
                                        else:
                                            yinit = years[i]
                                            yend = years[i + 1]
                                            print(yinit, yend)


                                        ###Main program###
                                        if os.path.exists(os.path.join(outputdir, vari)):
                                            data = pd.read_csv(os.path.join(outputdir, vari), header=None)

                                            y3,y4=yinit,yend
                                            # for y3 in range((math.floor(yinit / 10) * 10) + 1,(math.floor(yend / 10) * 10) + 11, 10):
                                            #     y4 = y3 + 9
                                            #     if yinit > y3:
                                            #         y3 = yinit
                                            #     if yend < y4:
                                            #         y4 = yend
                                            tableau.loc[subindex, 'lake'] = f_lake
                                            tableau.loc[subindex, 'model'] = modelid
                                            tableau.loc[subindex, 'scenario'] = scenarioid
                                            tableau.loc[subindex, 'Calibration'] = "Done"
                                            tableau.loc[subindex, 'Date'] = "%s_%s"%(y3,y4)
                                            file_name = "%s/%s/%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                reg, lake, model_name, bias, climate, socio, sens,
                                                variable, region, timestep, y1, y2)
                                            outdir1 = os.path.join(output1, "%s/%s/" % (reg, lake))
                                            file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                                y2)

                                            file_namel = file_name2.lower()
                                            outputnc = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"

                                            if (not os.path.exists(os.path.join(output1, "%s.txt" % file_name))):

                                                #data = pd.read_csv(os.path.join(outputdir,vari),header=None)
                                                if y3 == y1:
                                                    data = data.set_index(data[0])
                                                    data_set=data[1:]
                                                    data_set1=data
                                                else:
                                                    data = data.set_index(data[0])
                                                    data_set = data_set.append(data[1:])
                                                    data_set1 = data_set1.append(data)
                                        else:
                                            print(os.path.join(outputdir, vari, " doesn't exist"))
                                            tableau.loc[index, 'lake'] = f_lake
                                            tableau.loc[index, 'model'] = modelid
                                            tableau.loc[index, 'scenario'] = scenarioid
                                            tableau.loc[index, 'Calibration'] = "Done"
                                            tableau.loc[index, vari[:-4]] = "csv is missing"
                                            break
                                        # try:
                                        #     for row in range(searchstart, len(data.index)):
                                        #
                                        #         year = int(data.iloc[row][0][0:4])
                                        #
                                        #         if year >= y3 and year <= y4:
                                        #             searchend += 1
                                        #             if year == y2:
                                        #                 end = True
                                        #         else:
                                        #
                                        #             data_set = data.iloc[searchstart:searchend][1]
                                        #             data_set1 = data.iloc[:][searchstart:searchend]
                                        #             searchstart = searchend

                                else:
                                    data_set = pd.DataFrame()
                                    data_set1 = pd.DataFrame()

                                    outputdir = os.path.join(output, "%s/%s/%s/%s" % (reg, lake, modelid, scenarioid))

                                    ###Main program###
                                    if os.path.exists(os.path.join(outputdir, vari)):
                                        file_name = "%s/%s/%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                            reg, lake, model_name, bias, climate, socio, sens,
                                            variable, region, timestep, y1, y2)
                                        outdir1 = os.path.join(output1, "%s/%s/" % (reg, lake))
                                        file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                            model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                            y2)

                                        file_namel = file_name2.lower()
                                        outputnc = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"

                                        if (not os.path.exists(os.path.join(output1, "%s.txt" % file_name))) :

                                            data = pd.read_csv(os.path.join(outputdir, vari), header=None)

                                            y3,y4=y1,y2
                                            # for y3 in range((math.floor(yinit / 10) * 10) + 1,(math.floor(yend / 10) * 10) + 11, 10):
                                            #     y4 = y3 + 9
                                            #     if yinit > y3:
                                            #         y3 = yinit
                                            #     if yend < y4:
                                            #         y4 = yend
                                            tableau.loc[subindex, 'lake'] = f_lake
                                            tableau.loc[subindex, 'model'] = modelid
                                            tableau.loc[subindex, 'scenario'] = scenarioid
                                            tableau.loc[subindex, 'Calibration'] = "Done"
                                            tableau.loc[subindex, 'Date'] = "%s_%s"%(y3,y4)



                                            #data = pd.read_csv(os.path.join(outputdir,vari),header=None)
                                            data = data.set_index(data[0])
                                            data_set=data.drop(0,axis=1)
                                            data_set1=data

                                    else:
                                        print(os.path.join(outputdir, vari, " doesn't exist"))
                                        tableau.loc[index, 'lake'] = f_lake
                                        tableau.loc[index, 'model'] = modelid
                                        tableau.loc[index, 'scenario'] = scenarioid
                                        tableau.loc[index, 'Calibration'] = "Done"
                                        tableau.loc[index, vari[:-4]] = "csv is missing"
                                        break
                                        # try:
                                        #     for row in range(searchstart, len(data.index)):
                                        #
                                        #         year = int(data.iloc[row][0][0:4])
                                        #
                                        #         if year >= y3 and year <= y4:
                                        #             searchend += 1
                                        #             if year == y2:
                                        #                 end = True
                                        #         else:
                                        #
                                        #             data_set = data.iloc[searchstart:searchend][1]
                                        #             data_set1 = data.iloc[:][searchstart:searchend]
                                        #             searchstart = searchend


                                file_name = "%s/%s/%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                    reg, lake, model_name, bias, climate, socio, sens,
                                    variable, region, timestep, y1, y2)

                                outdir1 = os.path.join(output1, "%s/%s/" % (reg, lake))
                                file_name2 = "%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                    model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                    y2)

                                file_namel = file_name2.lower()
                                outputnc = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"

                                if (not os.path.exists(os.path.join(output1, "%s.txt" % file_name))):

                                    if not os.path.exists(outdir1):
                                        os.makedirs(outdir1)
                                    if len(data_set) != 0:
                                        data_set.fillna(method='ffill')
                                        if len(data_set) == ((datetime(y2,12,31)-datetime(y1,1,1)).days+1):

                                            data_set.to_csv(
                                                os.path.join(output1, "%s.txt" % file_name),
                                                header=None, index=None, sep=' ', mode='w')
                                        else:
                                            print('problem!')
                                            x=5/0
                                    else:
                                        print("data is empty!!!")
                                        x=5/0

                                    tableau.loc[subindex, vari[:-4]] = "Done"

                                    # data_set1.to_csv(os.path.join(output1,
                                    #                               "%s_data.txt" % file_name),
                                    #                  header=None, index=None, sep=' ',
                                    #                  mode='w')

                                    command = "cdo --history -f nc4c -z zip -setmissval,1e+20 -setunit,\"%s\" -setname," \
                                              "%s -setreftime,1661-01-01,00:00:00,1%s -settaxis," \
                                              "%s-01-01,00:00:00,1%s -input,%s %s.nc4 < %s.txt" % (
                                                  unit, variable, increment, startyear,
                                                  increment, grid,
                                                  os.path.join(output, file_name),
                                                  os.path.join(output, file_name))

                                    if  os.path.exists(
                                            os.path.join(output1,
                                                         "%s.txt" % (file_name))):
                                        print(command)
                                        # rootgrp = Dataset("%s.nc4" % file_name2, "w", format="NETCDF4")
                                        # print(rootgrp.data_model)
                                        # level = rootgrp.createDimension("levlak", len(data_set.columns))
                                        # time = rootgrp.createDimension("time", None)
                                        # lat = rootgrp.createDimension("lat", 1)
                                        # lon = rootgrp.createDimension("lon",1)
                                        # times = rootgrp.createVariable("time", "f8", ("time",))
                                        # levels = rootgrp.createVariable("level", "f8", ("levlak",))
                                        # latitudes = rootgrp.createVariable("lat", "f8", ("lat",))
                                        # longitudes = rootgrp.createVariable("lon", "f8", ("lon",))
                                        #
                                        # temp = rootgrp.createVariable("%s"%variable, "f8",
                                        #                                    ("time", "levlak", "lat", "lon",))
                                        #
                                        # print()
                                        # latitudes.units = "degrees north"
                                        #
                                        # longitudes.units = "degrees east"
                                        # levels.units = "m"
                                        # levels.axis = "Z"
                                        # temp.units = "%s" % unit
                                        # times.units = "days since 1661-01-01,00:00:00"
                                        # times.calendar = "proleptic_gregorian"
                                        # latitudes = run_myLake_ISIMIP.get_latitude(lake,"D:/forcing_data","EWEMBI","historical")
                                        # longitudes= run_myLake_ISIMIP.get_longitude(lake,"D:/forcing_data","EWEMBI","historical")
                                        # rootgrp.close()
                                        # os.system(command)

                                else:
                                    print("file exists")
                                    tableau.loc[subindex, vari[:-4]] = "Done"

#                                                    break

                                                # if end is True:
                                                #     data_set = data.iloc[:][searchstart:]
                                                #
                                                #     file_name = "%s/%s/%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                #         reg, lake, model_name, bias, climate,
                                                #         socio, sens, variable, region, timestep, y3, y4)
                                                #     data_set.to_csv(os.path.join(output1, "%s.txt" % file_name),
                                                #                     header=None, index=None, sep=' ', mode='w')
                                                #     tableau.loc[subindex, vari[:-4]] = "Done"
                                                #
                                                #     unit = output_unit.get(vari)
                                                #     increment = "day"
                                                #
                                                #     command = "cdo --history -f nc4c -z zip -setmissval,1e+20 -setunit,\"%s\" -setname," \
                                                #               "%s -setreftime,1661-01-01,00:00:00,1%s -settaxis," \
                                                #               "%s-01-01,00:00:00,1%s -input,%s %s.nc4 < %s.txt" % (
                                                #                   unit, variable, increment, startyear,
                                                #                   increment, grid,
                                                #                   os.path.join(output, file_name),
                                                #                   os.path.join(output, file_name))
                                                #
                                                #     if not os.path.exists(os.path.join(output1, "%s.nc4" % (
                                                #     file_name))) and os.path.exists(os.path.join(output1, "%s.txt" % (file_name))):
                                                #         print(command)
                                                #
                                                #         # os.system(command)


                                            # except:
                                            #
                                            #     print("bug with csv file for %s" % variable)
                                            #     tableau.loc[subindex, vari[:-4]] = "not created; Bug"

                                subindex += 1

                                        # else:
                                        #     print(os.path.join(outputdir, vari, " doesn't exist"))
                                        #     tableau.loc[index, 'lake'] = f_lake
                                        #     tableau.loc[index, 'model'] = modelid
                                        #     tableau.loc[index, 'scenario'] = scenarioid
                                        #     tableau.loc[index, 'Calibration'] = "Done"
                                        #     tableau.loc[index, vari[:-4]] = "csv is missing"
                                            #run_myLake_ISIMIP.outputfile(yinit, yend, outputdir)

                                # else:
                                #     ###Main program###
                                #     outputdir = os.path.join(output, "%s/%s/%s/%s" % (reg, lake, modelid, scenarioid))
                                #     if os.path.exists(os.path.join(outputdir, vari)):
                                #         data = pd.read_csv(os.path.join(output, "%s/%s/%s/%s" % (reg, lake, modelid, scenarioid), vari),header=None)
                                #         subindex=index
                                #         y3,y4=y1,y2
                                #         # for y3 in range((math.floor(y1/10)*10)+1, (math.floor(y2/10)*10)+11, 10):
                                #         #     y4 = y3 + 9
                                #         #     if y1 > y3:
                                #         #         y3 = y1
                                #         #     if y2 < y4:
                                #         #         y4 = y2
                                #
                                #         tableau.loc[subindex, 'lake'] = f_lake
                                #         tableau.loc[subindex, 'model'] = modelid
                                #         tableau.loc[subindex, 'scenario'] = scenarioid
                                #         tableau.loc[subindex, 'Calibration'] = "Done"
                                #         tableau.loc[subindex, 'Date'] = "%s_%s"%(y3,y4)
                                #
                                #         try:
                                #             for row in range(searchstart,len(data.index)):
                                #
                                #                 year = int(data.iloc[row][0][0:4])
                                #
                                #                 if year >= y3 and year <= y4:
                                #                         searchend +=1
                                #                         if year == y2:
                                #                             end = True
                                #                 else:
                                #
                                #                     data_set = data.iloc[searchstart:searchend][1]
                                #                     data_set1 = data.iloc[:][searchstart:searchend]
                                #                     searchstart = searchend
                                #
                                #                     file_name="%s/%s/%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                #                     reg, lake,  model_name,bias,climate,socio,sens,variable,region,timestep,y3,y4)
                                #                     outdir1 = os.path.join(output1, "%s/%s/" % (reg, lake))
                                #
                                #                     if not os.path.exists(os.path.join(output1,"%s.txt"%file_name )):
                                #                         if not os.path.exists(outdir1):
                                #                             os.makedirs(outdir1)
                                #
                                #                         data_set.to_csv(os.path.join(output1,"%s.txt"%file_name ), header=None, index=None, sep=' ',mode='w')
                                #
                                #                         tableau.loc[subindex, vari[:-4]] = "Done"
                                #
                                #
                                #                         data_set1.to_csv(os.path.join(output1, "%s_data.txt" % file_name),
                                #                                         header=None, index=None, sep=' ',
                                #                                         mode='w')
                                #
                                #                         command = "cdo --history -f nc4c -z zip -setmissval,1e+20 -setunit,\"%s\" -setname," \
                                #                                   "%s -setreftime,1661-01-01,00:00:00,1%s -settaxis," \
                                #                                   "%s-01-01,00:00:00,1%s -input,%s %s.nc4 < %s.txt" % (
                                #                                   unit, variable, increment, startyear, increment, grid,
                                #                                   os.path.join(output, file_name),
                                #                                   os.path.join(output, file_name))
                                #
                                #                         if not os.path.exists(os.path.join(output1, "%s.nc4" % (file_name))) and os.path.exists(os.path.join(output1,"%s.txt" % (file_name))):
                                #                             print(command)
                                #                             #os.system(command)
                                #
                                #                     else:
                                #                         print("file exists")
                                #                         tableau.loc[subindex, vari[:-4]] = "Done"
                                #
                                #
                                #                     break
                                #
                                #             if end is True:
                                #                 data_set = data.iloc[:][searchstart:][1]
                                #
                                #                 file_name = "%s/%s/%s_%s_ewembi_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                #                     reg, lake, model_name, bias, climate,
                                #                     socio, sens, variable, region, timestep, y3, y4)
                                #                 data_set.to_csv(os.path.join(output1, "%s.txt" % file_name), header=None, index=None, sep=' ',mode='w')
                                #                 tableau.loc[subindex, vari[:-4]] = "Done"
                                #
                                #                 unit = output_unit.get(vari)
                                #                 increment = "day"
                                #
                                #                 command = "cdo --history -f nc4c -z zip -setmissval,1e+20 -setunit,\"%s\" -setname," \
                                #                           "%s -setreftime,1661-01-01,00:00:00,1%s -settaxis," \
                                #                           "%s-01-01,00:00:00,1%s -input,%s %s.nc4 < %s.txt" % (
                                #                               unit, variable, increment, startyear, increment, grid,
                                #                               os.path.join(output, file_name),
                                #                               os.path.join(output, file_name))
                                #
                                #                 if not os.path.exists(os.path.join(output1, "%s.nc4" % (file_name))) and os.path.exists(os.path.join(output1, "%s.txt" % (file_name))):
                                #                     print(command)
                                #                     #os.system(command)
                                #
                                #
                                #         except:
                                #             print("bug with csv file for %s"%variable)
                                #             tableau.loc[subindex, vari[:-4]] = "not created; Bug"
                                #
                                #         subindex += 1
                                #
                                #
                                #     else:
                                #         print(os.path.join(outputdir, vari, " doesn't exist"))
                                #         tableau.loc[index, 'lake'] = f_lake
                                #         tableau.loc[index, 'model'] = modelid
                                #         tableau.loc[index, 'scenario'] = scenarioid
                                #         tableau.loc[index, 'Calibration'] = "Done"
                                #         tableau.loc[index, vari[:-4]] = "csv is missing"
                                #         # run_myLake_ISIMIP.outputfile(yinit, yend, outputdir)

                            index= subindex

                        else:
                            tableau.loc[index, 'lake'] = (f_lake)
                            tableau.loc[index, 'model'] = (modelid)
                            tableau.loc[index, 'scenario'] = (scenarioid)
                            tableau.loc[index,'Calibration'] = "not Done"
                            index+=1
                    else:
                        print(os.path.join(output1, "%s/%s/%s/%s doesn't exist" % (reg, lake, modelid, scenarioid)))
                        tableau.loc[index, 'lake'] = f_lake
                        tableau.loc[index, 'model'] = modelid
                        tableau.loc[index, 'scenario'] = scenarioid
                        tableau.loc[index, 'Calibration'] = "folder is missing"
                        index += 1

    else:
        print("Lake is not in regions")
    tableau.to_csv(r"all_variable_lakes_combinaison0129.csv", index=False)

def revision():

    index = range(0,len(full_lake_list)*21)
    columns = ['lake', 'model', 'scenario','nbr_forcing_data','forcing_data_empty','hyspometry','daily_obs','daily_format','init','input','par','calibration']
    tableau = pd.DataFrame(index=index, columns=columns)
    index = 0
    for lake in (full_lake_list):
        if lake == "Eagle_Lake":
            test=1
        corrected_names = ["Allequash_Lake", "Big_Muskellunge_Lake", "Black_Oak_Lake", "Burley_Griffin", "Crystal_Bog",
                           "Crystal_Lake",
                           "Dickie_Lake", "Eagle_Lake", "Ekoln_basin_of_Malaren", "Esthwaite_Water",
                           "Falling_Creek",
                           "Fish_Lake", "Great_Pond", "Green_Lake", "Harp_Lake", "Laramie_Lake", "Lower_Zurich",
                           "Mt_Bold",
                           "Nohipalo_Mustjarv", "Nohipalo_Valgejarv", "Okauchee_Lake", "Rappbode_Reservoir",
                           "Sau_Reservoir",
                           "Sparkling_Lake", "Toolik_Lake", "Trout_Bog", "Trout_Lake", "Two_Sisters_Lake"]

        f_lake = lake
        # for name in corrected_names:
        #     if lake == "Crystal":
        #         f_lake = "Crystal_Lake"
        #         break
        #     elif lake == "Trout":
        #         f_lake = "Trout_Lake"
        #         break
        #     elif lake == "TroutBog":
        #         f_lake = "Trout_Bog"
        #         break
        #     elif lake == "Two_Sisters":
        #         f_lake = "Two_Sisters_Lake"
        #         break
        #     elif lake == "Toolik":
        #         f_lake = "Toolik_Lake"
        #         break
        #
        #     if lake in name.replace("_", ''):
        #         f_lake = name
        #         break
        reg ='None'
        for region in regions:
            if lake in regions[region]:
                reg = region
                break
        if reg != "None":
            for model in models:
                for scenario in scenarios:
                    print(lake,model,scenario)
                    if (model == "EWEMBI" and scenario == "historical") or model !="EWEMBI":
                        tableau.loc[index,'lake'] = f_lake
                        tableau.loc[index, 'model'] = model
                        tableau.loc[index, 'scenario'] = scenario

                        count = 0
                        empty= 0
                        for vari in input_variables:
                            if os.path.exists(os.path.join(r"D:\forcing_data","%s_%s_%s_%s.allTS.nc" % (vari,model,scenario,f_lake))):
                                count +=1
                            try:
                                ncdf_file = ncdf.Dataset(os.path.join(r"D:\forcing_data","%s_%s_%s_%s.allTS.nc" % (vari,model,scenario,f_lake)), "r", format="NETCDF4")
                                if vari == "tas":  # converting from Kelvins to Celsius
                                    temp = float(ncdf_file.variables[vari][1]) - 273.15
                            except:
                                empty += 1

                        tableau.loc[index, 'nbr_forcing_data'] = count
                        tableau.loc[index, 'forcing_data_empty'] = empty


                        print("observations/{}/{}/{}_hypsometry_modified.csv".format(reg, lake, lake))
                        if os.path.exists("observations/{}/{}/{}_hypsometry_modified.csv".format(reg, lake, lake)):
                            with open("observations/{}/{}/{}_hypsometry_modified.csv".format(reg, lake, lake)) as obs:
                                reader = list(csv.reader(obs))
                                prefix = lake[:3]
                                if lake == "Nohipalo_Valgejarv":
                                    prefix = 'NoV'
                                elif lake == "Nohipalo_Mustjarv":
                                    prefix = 'NoM'
                                elif lake == "Crystal_Bog":
                                    prefix = 'CrB'
                                elif lake == "Great_Pond":
                                    prefix = 'GrP'
                                elif lake == "Trout_Bog":
                                    prefix = 'TrB'
                                elif lake == "Mt_Bold":
                                    prefix = 'MtB'
                            tableau.loc[index, 'hyspometry'] = "Exist"
                            if not os.path.exists("{}/Calibration_Complete.txt".format("D:\output/{}/{}/EWEMBI/historical".format(reg,lake))):
                                eee=1#myLake_post.run_optimization_Mylake_save(lake, "input/{}/{}".format(reg, prefix),
                                 #                                            "D:\output/{}/{}/EWEMBI/historical".format(reg,lake),r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations{}/{}".format(reg, lake), reg, model, scenario)


                            outputdir= run_myLake_ISIMIP.init_info(f_lake, "observations/{}/{}".format(reg, lake))
                            if os.path.exists(outputdir.get("outdir")+"/{}_{}_{}_input".format( prefix, model, scenario)):



                                tableau.loc[index, 'input'] = "Exist"
                            else:
                                tableau.loc[index, 'input'] = "Does not exist"
                            if os.path.exists(outputdir.get("outdir")+"/{}_init".format( prefix)):
                                tableau.loc[index, 'init'] = "Exist"
                            else:
                                tableau.loc[index, 'init'] = "Does not exist"
                            if os.path.exists(outputdir.get("outdir")+"/{}_par".format( prefix)):
                                tableau.loc[index, 'par'] = "Exist"
                            else:
                                tableau.loc[index, 'par'] = "Does not exist"
                        else:
                            tableau.loc[index, 'hyspometry'] = "Does not exist"

                        if (os.path.exists(r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations/{}/{}\{}_temp_daily.csv".format(reg, lake, lake))):
                            tableau.loc[index, 'daily_obs'] = "Exist"
                        else:
                            tableau.loc[index, 'daily_obs'] = "Does not exist"

                        if (os.path.exists(
                                r"D:\output/{}/{}/{}/{}/Observed_Temperatures.csv".format(reg, lake, model, scenario))):
                            tableau.loc[index, 'daily_format'] = "Exist"
                        else:
                            tableau.loc[index, 'daily_format'] = "Does not exist"

                        if os.path.exists("D:\output/{}/{}/EWEMBI/historical/Calibration_Complete".format(reg,lake)):
                            tableau.loc[index, 'calibration'] = "Exist"
                        else:
                            tableau.loc[index, 'calibration'] = "Does not exist"

                        index +=1

    tableau.to_csv(r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\recapitulatif.csv",index=False)

def input_files_parallel():
    with open(report, 'w') as f:
        f.write('\nrunning _parallel\n' )
        f.close()

    print("start")
    #for lake in full_lake_list:
    #    input_files_loop(lake)
    Parallel(n_jobs=num_cores, verbose=10)(delayed(input_files_loop)(lake) for lake in full_lake_list)

def input_files_loop(lake):


    corrected_names = ["Allequash_Lake", "Big_Muskellunge_Lake", "Black_Oak_Lake", "Burley_Griffin", "Crystal_Bog",
                       "Crystal_Lake",
                       "Dickie_Lake", "Eagle_Lake", "Ekoln_basin_of_Malaren", "Esthwaite_Water",
                       "Falling_Creek",
                       "Fish_Lake", "Great_Pond", "Green_Lake", "Harp_Lake", "Laramie_Lake", "Lower_Zurich", "Mt_Bold",
                       "Nohipalo_Mustjarv", "Nohipalo_Valgejarv", "Okauchee_Lake", "Rappbode_Reservoir",
                       "Sau_Reservoir",
                       "Sparkling_Lake", "Toolik_Lake", "Trout_Bog", "Trout_Lake", "Two_Sisters_Lake"]


    print("lake", lake)
    f_lake = lake
    prefix = f_lake[:3]
    if lake == "Nohipalo_Valgejarv":
        prefix = 'NoV'
    elif lake == "Nohipalo_Mustjarv":
        prefix = 'NoM'
    elif lake == "Crystal_Bog":
        prefix = 'CrB'
    elif lake == "Great_Pond":
        prefix = 'GrP'
    elif lake == "Trout_Bog":
        prefix = 'TrB'
    elif lake == "Mt_Bold":
        prefix = 'MtB'
    # for name in corrected_names:
    #     if lake == "Crystal":
    #         f_lake = "Crystal_Lake"
    #         break
    #
    #
    #
    #
    #     elif lake == "Trout":
    #         f_lake = "Trout_Lake"
    #         break
    #     elif lake == "TroutBog":
    #         f_lake = "Trout_Bog"
    #         break
    #     elif lake == "Two_Sisters":
    #         f_lake = "Two_Sisters_Lake"
    #         break
    #     elif lake == "Toolik":
    #         f_lake = "Toolik_Lake"
    #         break
    #
    #     if lake in name.replace("_", ''):
    #         f_lake = name
    #         break
    try:

        with open(report, 'a') as f:
            f.write('running lake %s \n'% (lake))
            f.close()
        print("download for %s"%lake)
        download_forcing_data(f_lake)
        with open(report, 'a') as f:
            f.write('download for %s completed\n'% (lake))
            f.close()
            print('download for %s completed'% lake)
    except:
        with open(report, 'a') as f:
            f.write('unable download for %s\n'% (lake))
            f.close()
        print('unable to download of %s' % lake)

    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break
    try:
        print("D:/forcing_data/hurs_EWEMBI_historical_{}.allTS.nc".format( lake))
        ncdf_file = ncdf.Dataset( "D:/forcing_data/hurs_EWEMBI_historical_{}.allTS.nc".format( lake), "r",
            format="NETCDF4")

        if reg == None:
            print("Cannot find {}'s region".format(lake))
            return None
        else:
            print("observations/{}/{}/{}_hypsometry_modified.csv".format(reg, lake, lake))
            with open("observations/{}/{}/{}_hypsometry_modified.csv".format(reg, lake, lake)) as obs:
                reader = list(csv.reader(obs))

        outputdir = run_myLake_ISIMIP.init_info(lake, "observations/{}/{}".format(reg, lake))
        for model in models:
            for scenario in scenarios:

                #if os.path.exists("D:\output/{}/{}/{}/{}/RunComplete".format(reg, lake, model, scenario)):
                #    print("{} {} {} Run is already completed.\n".format(lake, model, scenario))

                if not os.path.exists(
                        "D:\output/{}/{}/EWEMBI/historical/Calibration_Complete1.txt".format(reg, lake)):# and os.path.exists(
                        #"input/{}/{}/{}_{}_{}_input".format(reg, prefix, prefix, model, scenario)):

                    if 1==1:#try:
                        if (model == "EWEMBI" and scenario == "historical") or model !="EWEMBI":
                            print(outputdir.get("outdir")+r"\{}_{}_{}_input".format(prefix, model, scenario))
                            print(outputdir.get("outdir")+ "\{}_par".format( prefix))

                            #if 1==1:
                            if not (os.path.exists( outputdir.get("outdir")+ r"\{}_{}_{}_input".format(prefix, model, scenario))) \
                                    or not (os.path.exists(outputdir.get("outdir")+r"\{}_par".format(prefix))) \
                                    or not (os.path.exists(outputdir.get("outdir")+r"\{}_init".format(prefix))):
                                if 1==1:#try:

                                    run_myLake_ISIMIP.generate_input_files("observations/{}/{}".format(reg, lake), lake, f_lake,
                                                               "D:/forcing_data", run_myLake_ISIMIP.get_longitude(f_lake, "D:/forcing_data",model,scenario),
                                                               run_myLake_ISIMIP.get_latitude(f_lake, "D:/forcing_data",model,scenario), model, scenario,reg)
                                # except:
                                #     print("missing data")
                                if not (os.path.exists( r"D:\output/{}/{}/{}/{}/Observed_Temperatures.csv".format(reg, lake, "EWEMBI","historical"))):
                                    if (os.path.exists( r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations/{}/{}\{}_temp_daily.csv".format(reg, lake, lake))):
                                        if 1==1:#try:
                                            myLake_post.temperatures_by_depth("observations/{}/{}".format(reg, lake),
                                                                              lake,
                                                                              "D:\output/{}/{}/{}/{}".format(reg, lake,
                                                                                                             'EWEMBI',
                                                                                                             "historical"),
                                                                              'EWEMBI',
                                                                              "historical")
                                            print("obsertvation done")
                                        # except:
                                        #     print("missing data")
                                    else:
                                        print('no daily data for %s' % lake)


                            else:
                                print('Already done')

                                if not (os.path.exists(
                                        r"D:\output/{}/{}/{}/{}/Observed_Temperatures.csv".format(reg, lake, "EWEMBI",
                                                                                                  "historical"))):
                                    if (os.path.exists(
                                            r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations/{}/{}\{}_temp_daily.csv".format(
                                                    reg, lake, lake))):
                                        try:
                                            myLake_post.temperatures_by_depth("observations/{}/{}".format(reg, lake),
                                                                              lake,
                                                                              "D:\output/{}/{}/{}/{}".format(reg, lake,
                                                                                                             'EWEMBI',
                                                                                                             "historical"),
                                                                              'EWEMBI',
                                                                              "historical")
                                            print("obsertvation done")
                                        except:
                                            print("missing data")
                                    else:
                                        print('no daily data for %s' % lake)
                        else:
                            if not model == "EWEMBI":
                                try:
                                    run_myLake_ISIMIP.generate_input_files("observations/{}/{}".format(reg, lake), lake, f_lake,
                                                                       "D:/forcing_data",
                                                                       run_myLake_ISIMIP.get_longitude(f_lake, "D:/forcing_data",model,scenario),
                                                                       run_myLake_ISIMIP.get_latitude(f_lake, "D:/forcing_data",model,scenario),
                                                                       model, scenario)
                                except:
                                    print("missing data")
                    # except:
                    #    print("problem when doing %s %s %s" % (lake, model, scenario))

                    if model == "EWEMBI":
                        if scenario == "historical":
                            if not os.path.exists(outputdir.get("outdir")+"/{}_{}_{}_input".format(prefix,model,scenario)) or not os.path.exists(outputdir.get("outdir")+"/{}_init".format(prefix)) or not os.path.exists(outputdir.get("outdir")+"/{}_par".format(prefix)):
                                print("not all initial files existing for %s %s %s" % ( model, scenario,lake))
                    else:
                        if not os.path.exists(outputdir.get("outdir")+"/{}_{}_{}_input".format(prefix,model,scenario)) or not os.path.exists(outputdir.get("outdir")+"/{}_init".format(prefix)) or not os.path.exists(outputdir.get("outdir")+"/{}_par".format(prefix)):
                            print("not all initial files existing for %s %s %s" % ( model, scenario,lake))
            #for model in models:
            #    for scenario in scenarios:
            #        for var in input_variables:
            #            os.remove("forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, f_lake))
    except:
        print("missing file climatic probably or orthers...")

def download_forcing_data(lake):
    """
    A function to download forcing data from dkrz server.
    :param lake:
    :return:
    """
    countto = 0
    for model in models:
        for scenario in scenarios:
            if not os.path.exists(os.path.join("D:/forcing_data\\{}\\Complete_Download_{}_{}_{}.txt".format(lake, model, scenario, lake))):
                done = 0
                if model == "EWEMBI":
                    if scenario == "historical":

                        with pysftp.Connection('mistralpp.dkrz.de', username='b380750', password='TwopFaP5') as sftp:
                            sftp.cwd(
                                "/mnt/lustre01/work/bb0820/ISIMIP/ISIMIP2b/InputData/OBS_atmosphere/local_lakes/EWEMBI/historical")
                            print(sftp.listdir())
                            for var in input_variables:

                                if not os.path.exists("D:/forcing_data\\{}_EWEMBI_historical_{}.allTS.nc".format(var, lake)):
                                    print("start scenario EWE histo")
                                    try:
                                        sftp.get("{}/{}_EWEMBI_historical_{}.allTS.nc".format(lake, var, lake),
                                                 localpath="D:/forcing_data\\{}_EWEMBI_historical_{}.allTS.nc".format(var,
                                                                                                                      lake))
                                        print("end")
                                        done += 1

                                    except:
                                        print("enable to get {}/{}_EWEMBI_historical_{}.allTS.nc".format(lake, var, lake))
                                else:
                                    done += 1
                                    with open(report, 'a') as f:
                                        f.write('download already done %s \n' % (lake))
                                        f.close()
                                    print('download already done %s \n' % (lake))

                else:

                    with pysftp.Connection('mistralpp.dkrz.de', username='b380750', password='TwopFaP5') as sftp:
                        sftp.cwd( "/mnt/lustre01/work/bb0820/ISIMIP/ISIMIP2b/InputData/GCM_atmosphere/biascorrected/local_lakes")
                        print(sftp.listdir())
                        for var in input_variables:
                            print(done)
                            if not os.path.exists("D:/forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, lake)):
                                print("start scenario %s"%(scenario))
                                try:
                                    sftp.get("{}/{}_{}_{}_{}.allTS.nc".format(lake, var, model, scenario, lake), localpath="D:/forcing_data\\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, lake))
                                    print("end")
                                    done += 1
                                except:
                                    print("enable to get {}/{}_{}_{}_{}.allTS.nc".format(lake, var, model, scenario, lake))
                            else:
                                done += 1
                                with open(report, 'a') as f:
                                    f.write('download already done %s \n' % (lake))
                                    f.close()
                                print('download already done %s \n' % (lake))

                if done == 6:
                    outdirl = os.path.join("D:/forcing_data\\{}".format(lake))
                    if not os.path.exists(outdirl):
                        os.makedirs(outdirl)
                    with open(os.path.join("{}\\Complete_Download_{}_{}_{}.txt".format(outdirl, model, scenario, lake)), 'w') as f:
                        f.write("Done")
                    print("Done")
                    countto += 1
            else:
                countto += 1
    if countto == 21:
        outdirl = os.path.join("D:/forcing_data\\{}".format(lake))
        with open(os.path.join("{}\\Complete_Download_all_{}.txt".format(outdirl, lake)),'w') as f:
            f.write("Done")
            print("Done")

def mylake_parallel():

    #for lake in full_lake_list:
    #    model_scenario_loop(lake)
    Parallel(n_jobs=num_cores, verbose=10)(delayed(model_scenario_loop)(lake) for lake in full_lake_list)

def model_scenario_loop(lake):


    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    if reg == None:
        print("Cannot find {}'s region".format(lake))
        return None
    else:
        prefix = lake[:3]
        if lake == "Nohipalo_Valgejarv":
            prefix = 'NoV'
        elif lake == "Nohipalo_Mustjarv":
            prefix = 'NoM'
        elif lake == "Crystal_Bog":
            prefix = 'CrB'
        elif lake == "Great_Pond":
            prefix = 'GrP'
        elif lake == "Trout_Bog":
            prefix = 'TrB'
        elif lake == "Mt_Bold":
            prefix = 'MtB'
        if prefix == "Bmu":
            prefix = "Big"
        elif prefix == "GRN":
            prefix = "Gre"
        elif prefix == "Mug":
            prefix = "Mue"
        try:
            with open("observations/{}/{}/{}_hypsometry_modified.csv".format(reg,lake, lake)) as obs:
                reader = list(csv.reader(obs))

                getprefix=True
        except:
            getprefix = False
        outputdir = run_myLake_ISIMIP.init_info(lake, "observations/{}/{}".format(reg, lake))

        if os.path.exists("D:\output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(reg, lake)):

            for model in models:
                if not model == "EWEMBI":
                    for scenario in scenarios:

                        print(outputdir.get("outdir")+"/{}_{}_{}_input".format(prefix,model,scenario))
                        if os.path.exists("D:\output/{}/{}/{}/{}/RunComplete1".format(reg, lake, model, scenario)):
                             print("{} {} {} Run is already completed.\n".format(lake, model, scenario))

                        elif os.path.exists("D:\output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(reg, lake)) :
                            if os.path.exists(outputdir.get("outdir")+"/{}_{}_{}_input".format(prefix,model,scenario)):
                                if 1==1:#try:
                                    print("start")
                                    run_myLake_ISIMIP.run_myLake(run_myLake_ISIMIP.init_info(lake, "observations/{}/{}".format(reg, lake)).get("outdir"), reg, lake, model, scenario)
                                    print("Run of {} {} {} Completed.\n".format(lake, model, scenario))
                                #except:
                                    print("problem with {} {} {}.\n".format(lake, model, scenario))
                            else:
                                print("input doesnt exist")
                        else:
                            print("{} Calibration have not been done.\n".format(lake))

def make_parameters_file_parallel():
    """
    Parallelise the function get_best_parameters.
    """
    Parallel(n_jobs=num_cores, verbose=10)(delayed(get_best_parameters(lake)) for lake in lake_list)

def get_best_parameters(lake):
    """
    Looks for the results of calibration for a lake and returns the value for the parameters.
    :param lake:
    :return:
    """
    reg = ''
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    with open("D:\output/{}/{}/GFDL-ESM2M/historical".format(reg, lake)) as results:
        pass

def calibration_parallel():
    """
    Simple function to call a parallel calibration of all lakes.
    :return:
    """
    print(num_cores)
    # with open(report, 'w') as f:
    #     f.write('\nrunning _parallel\n' )
    #     f.close()
    # for lake in full_lake_list:
    #      run_calibrations(lake)
    #full_lake_list=["Burley_Griffin","Laramie_Lake","Tarawera"]
    Parallel(n_jobs=num_cores, verbose=10)(delayed(run_calibrations)(lake) for lake in full_lake_list)

def run_calibrations(lake):
    """
    Intermediary function to call nelder-mead optimisation function for a single lake.

    :param lake: Type string. The name of the lake to calibrate.
    :return: If Calibration_Complete file is found, returns None. Else, return the nelder-mead optimisation function
    from myLake_post module for the given lake.
    """
    print(lake)

    file = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"
    reg = None
    for region in regions:
        if lake in regions[region]:
            reg = region
            break

    if reg == None:
        print("Cannot find {}'s region".format(lake))
        return None
    else:
        with open("{}/observations/{}/{}/{}_hypsometry_modified.csv".format(file,reg, lake, lake)) as obs:
            reader = list(csv.reader(obs))

    optimisation_completed_unsuccessfull = []
    prefix = lake[:3]
    if lake == "Nohipalo_Valgejarv":
        prefix = 'NoV'
    elif lake == "Nohipalo_Mustjarv":
        prefix = 'NoM'
    elif lake == "Crystal_Bog":
        prefix = 'CrB'
    elif lake == "Great_Pond":
        prefix = 'GrP'
    elif lake == "Trout_Bog":
        prefix = 'TrB'
    elif lake == "Mt_Bold":
        prefix = 'MtB'
    for region in regions:
        if lake in regions[region]:
            outputdir = run_myLake_ISIMIP.init_info(lake, "observations/{}/{}".format(region, lake))
            if os.path.exists("D:\output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(region, lake)):
                print("Calibration for {} is already complete.\n".format(lake))
                return None

            elif not os.path.exists(outputdir.get("outdir")+"/{}_EWEMBI_historical_input".format(prefix)):
                print("Calibration can't {} be completed. input file doesn't exist.\n".format(lake))
                return None
            elif not os.path.exists(outputdir.get("outdir")+"/{}_EWEMBI_historical_input".format(prefix)) or not os.path.exists(outputdir.get("outdir")+"/{}_init".format(prefix)) or not os.path.exists(outputdir.get("outdir")+"/{}_par".format(prefix)):
                print("not all initial files existing for %s" % ( lake))
            elif os.path.exists("D:\output/{}/{}/EWEMBI/historical/Calibration_problem.txt".format(region, lake)):
                print("Unable to calibration {}.\n".format(lake))
                return None

            else:

                return myLake_post.optimize_differential_evolution(lake, "observations/{}/{}".format(reg,lake),
                                                     "input/{}/{}".format(region, prefix), region,
                                                     "D:\output/{}/{}/{}/{}".format(region, lake, "EWEMBI","historical"),"EWEMBI","historical")


    print("Cannot find {}'s region".format(lake))

if __name__ == "__main__":
    #
    #input_files_parallel()
    # lake ="BurleyGriffin"
    # reg = "AU"
    # model="EWEMBI"
    # scenario= "historical"
    #for lake in full_lake_list:
    #    input_files_loop(lake)
       # myLake_post.temperatures_by_depth(r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations/{}/{}/".format(reg, lake), lake,
       #                                   os.path.join(r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\output",reg, lake),model,scenario)
    #input_files_loop("Washington")
    # #Parallel(n_jobs=num_cores)(delayed(run_calibrations(lake)) for lake in full_lake_list)
    #for lake in ['Allequash_Lake', 'Alqueva', 'Annecy', 'Annie', 'Argyle', 'Biel', 'Big_Muskellunge_Lake', 'Black_Oak_Lake', 'Bourget', 'Burley_Griffin', 'Crystal_Bog', 'Crystal_Lake', 'Delavan', 'Dickie_Lake', 'Eagle_Lake', 'Ekoln_basin_of_Malaren', 'Erken', 'Esthwaite_Water', 'Falling_Creek_Reservoir', 'Feeagh', 'Fish_Lake', 'Geneva', 'Great_Pond', 'Green_Lake', 'Harp_Lake', 'Kilpisjarvi', 'Kinneret', 'Kivu', 'Klicava', 'Kuivajarvi', 'Langtjern', 'Laramie_Lake', 'Lower_Zurich', 'Mendota', 'Monona', 'Mozaisk', 'Mt_Bold', 'Mueggelsee', 'Neuchatel', 'Ngoring', 'Nohipalo_Mustjarv', 'Nohipalo_Valgejarv', 'Okauchee_Lake', 'Paajarvi', 'Rappbode_Reservoir', 'Rimov', 'Rotorua', 'Sammamish', 'Sau_Reservoir', 'Sparkling_Lake', 'Stechlin', 'Sunapee', 'Tahoe', 'Tarawera', 'Toolik_Lake', 'Trout_Bog', 'Trout_Lake', 'Two_Sisters_Lake', 'Vendyurskoe', 'Victoria', 'Vortsjarv', 'Washington', 'Windermere', 'Wingra', 'Zlutice']:
    #    download_forcing_data(lake)
    #input_files_parallel()

    #for lake in ["Geneva","Muggelsee","Eagle_lake","Mozhaysk"]:
    #    download_forcing_data(lake)

    #revision()

    #calibration_parallel()
    #input_files_loop("Burley_Griffin")
    from transfertreal import transfert

    lake = "Tahoe"
    # reg = "AU"
    # model="EWEMBI"
    # scenario="historical"
    # run_myLake_ISIMIP.run_myLake(
    #     run_myLake_ISIMIP.init_info(lake, "observations/{}/{}".format(reg, lake)).get("outdir"), reg, lake, model,
    #     scenario)

    #input_files_loop(lake)
    # run_calibrations(lake)
    # model_scenario_loop(lake)
    # # #
    # # #
    # format(lake)
    # format_nc41(lake)
    #
    # try:
    #     Parallel(n_jobs=num_cores, verbose=10)(delayed(transfert)(folder) for folder in [ 'future_extended', 'pre-industrial','historical','future',])
    # except:
    #     print('bug1')
    # #for lake in full_lake_list:
    #    format(lake)
    #format_nc41('Annecy')
    #format_nc4("lake").
    #format_nc4("lake")
    #input_files_parallel()
    #mylake_parallel()
    format_parallel()
    #format_nc4_par()

    #run_calibrations(full_lake_list[1])
    #revision()

    #plot_calibrations_par()
    #
    #input_files_parallel()

    #calibration_parallel()
    #
