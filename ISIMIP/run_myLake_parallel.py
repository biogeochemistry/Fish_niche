"""
run_mylake_parallel.py

Module that allows running multiple simulations or calibrations in parallel.
"""
import run_myLake_ISIMIP
import myLake_post
import csv
import sys
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

num_cores = mp.cpu_count()-2
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
full_lake_list =["Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal",
                 "CrystalBog", "Delavan","FallingCreek", "Fish", "Great_Pond",
                 "Green_lake", "Laramie", "Mendota", "Monona","Okauchee",
                 "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik",
                 "Trout", "TroutBog", "Two_Sisters","Washington", "Wingra",
                 "Biel", "LowerZurich", "Neuchatel","Alqueva","Annecy",
                 "Bourget", "Geneva","Argyle", "BurleyGriffin", "MtBold",
                 "Dickie", "Eagle_lake", "Harp","Ekoln", "Erken",
                 "EsthwaiteWater", "Windermere","Feeagh","Kilpisjarvi", "Kuivajarvi",
                 "Paajarvi","Kinneret","Kivu","Klicava", "Rimov",
                 "Zlutice","Langtjern","Mozhaysk", "Vendyurskoe","Muggelsee",
                 "Rappbode", "Stechlin","Ngoring","NohipaloMustjarv", "NohipaloValgejarv",
                 "Vortsjarv","Sau","Rotorua", "Tarawera", "Taupo",
                 "Waahi"]

#"Ngoring","Klicava",
full_lake_list =["Argyle","Alqueva","Rotorua","Muggelsee","BigMuskellunge",
                 "Vendyurskoe",
                 "EsthwaiteWater","Allequash", "Annie",  "BlackOak", "Crystal",
                 "CrystalBog", "Delavan","Falling_Creek", "Fish", "Great_Pond",
                 "Green_lake", "Laramie", "Mendota", "Okauchee",
                  "Sparkling", "Sunapee", "Toolik",
                 "Trout", "TroutBog", "Two_Sisters_Lake", "Wingra",
                 "Biel", "LowerZurich", "Neuchatel","Annecy",
                 "Bourget",  "BurleyGriffin", "MtBold",
                 "Dickie", "Eagle_lake", "Harp","Ekoln", "Erken",
                 "EsthwaiteWater", "Windermere","Feeagh","Kilpisjarvi", "Kuivajarvi",
                 "Paajarvi","Kinneret","Kivu", "Rimov",
                 "Langtjern","Mozhaysk","Rappbode", "Stechlin","NohipaloMustjarv", "NohipaloValgejarv",
                 "Vortsjarv","Sau", "Tarawera", "Washington","Sammamish", "Geneva","Tahoe"]

full_lake_list=['Langtjern','Annecy','Argyle','Crystal','Dickie',
                'Ekoln','Erken','EsthwaiteWater','Feeagh','Fish',
                'Harp','Kilpisjarvi','Kinneret','Kivu','LowerZurich',
                'Mendota','Neuchatel','Okauchee','Rimov','Rotorua',
                'Sammamish','Sau','Sparkling','Stechlin','Sunapee',
                'Vendyurskoe','Vortsjarv','Windermere']



#'Ngoring',

full_lake_list= ['Annie', 'Allequash_Lake', 'Alqueva', 'Annecy', 'Argyle',
                  'Biel', 'Big_Muskellunge_Lake', 'Black_Oak_Lake', 'Bourget', 'Burley_Griffin',
                  'Crystal_Bog', 'Crystal_Lake', 'Delavan', 'Dickie_Lake', 'Eagle_Lake',
                  'Ekoln_basin_of_Malaren', 'Erken', 'Esthwaite_Water', 'Falling_Creek_Reservoir', 'Feeagh',
                  'Fish_Lake', 'Geneva', 'Great_Pond', 'Green_Lake', 'Harp_Lake',
                  'Kilpisjarvi', 'Kinneret', 'Kivu', 'Kuivajarvi','Langtjern',
                  'Laramie_Lake', 'Lower_Zurich', 'Mendota', 'Monona','Mozaisk',
                  'Mt_Bold', 'Mueggelsee', 'Neuchatel', 'Nohipalo_Mustjarv',
                  'Nohipalo_Valgejarv', 'Okauchee_Lake', 'Paajarvi', 'Rappbode_Reservoir','Rimov',
                  'Rotorua', 'Sammamish', 'Sau_Reservoir', 'Sparkling_Lake','Stechlin',
                  'Sunapee', 'Tahoe', 'Tarawera', 'Toolik_Lake','Trout_Bog',
                  'Trout_Lake', 'Two_Sisters_Lake', 'Vendyurskoe', 'Vortsjarv', 'Washington',
                  'Windermere', 'Wingra']

full_lake_list1=["Annie","Two_Sisters_Lake"]
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

models = ["EWEMBI","GFDL-ESM2M","MIROC5", "IPSL-CM5A-LR","HadGEM2-ES"]

scenarios = ["historical","rcp85","rcp26","piControl","rcp60"]

input_variables = ["hurs",
                   "pr",
                   "ps",
                   "rsds",
                   "sfcWind",
                   "tas"
                    ]
output_variables = ["strat.csv", "watertemp.csv", "thermodepth.csv", "ice.csv", "lakeicefrac.csv",
                    "snowtick.csv", "sensheatf.csv", "latentheatf.csv", "lakeheatf.csv", "albedo.csv",
                    "turbdiffheat.csv", "sedheatf.csv"]

output_variables_depths = [ "watertemp.csv",
                    "turbdiffheat.csv", "sedheatf.csv"]
output_unit={"strat.csv":"unitless", "watertemp.csv":"K", "thermodepth.csv":"m", "ice.csv":"unitless",
             "lakeicefrac.csv":"unitless","snowtick.csv":"m", "sensheatf.csv":"W m-2",
             "latentheatf.csv":"W m-2", "lakeheatf.csv":"W m-2", "albedo.csv":"unitless",
             "turbdiffheat.csv":"m2 s-1", "sedheatf.csv":"W m-2"}
params_0 = np.array([0, 0.3, 0.55, 1, 0, 2.5, 1])
report = 'report.txt'
grid = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\grid.txt"
output = r"D:/output"
output1 = r"D:/output"

from subprocess import Popen, PIPE



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
        y1, y2 = 1661, 2299
    elif scenarioid == 'historical':
        y1, y2 = 1861, 2005
    elif scenarioid == 'rcp26':
        y1, y2 = 2006, 2299
    else:
        y1, y2 = 2006, 2099

    return y1, y2


def format_nc4():
    index = range(0, len(full_lake_list) * 21 * 10)
    columns = ['lake', 'model', 'scenario', 'Calibration'] + ['%s.txt'%vari[:-4] for vari in output_variables]+['%s.nc4'%vari[:-4] for vari in output_variables]
    tableau = pd.DataFrame(index=index, columns=columns)
    index = 0
    commandall = "{ "
    for lake in (full_lake_list):
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

        outputnc = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"
        grid = 'D:/output_txt/grid_%s.txt' % ( lake)
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

                        for vari in output_variables:
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
                            y3,y4=y1, y2
                            # for y3 in range((math.floor(y1 / 10) * 10) + 1, (math.floor(y2 / 10) * 10) + 11, 10):
                            #     y4 = y3 + 9
                            #     if y1 > y3:
                            #         y3 = y1
                            #     if y2 < y4:
                            #         y4 = y2

                            if 1==1:#if y2 - y1 > 100:

                                file_name = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                    model_name, bias, climate, socio, sens, variable, region, timestep, y3, y4)
                                file_name2 = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                    model_name, bias, climate, socio, sens, variable, lake, timestep, y3, y4)
                                print(os.path.join(output, file_name))
                                file_namel = file_name2.lower()
                                if os.path.exists(os.path.join(output, "%s.txt" % file_name)) or os.path.exists(os.path.join(outputnc, "%s.nc4" % (file_namel))) or os.path.exists(os.path.join(outputnc, "%s.nc" % (file_namel))):
                                    tableau.loc[index, '%s.txt'%variable] = 'Exists'
                                    command = './cdo.exe --history -f nc4c -z zip -setmissval,1e+20 -setunit,\"%s\" -setname,' \
                                           '%s -setreftime,1661-01-01,00:00:00,1%s -settaxis,' \
                                              '%s-01-01,00:00:00,1%s -input,%s %s.nc4 < %s.txt' % \
                                              (unit, variable, increment, startyear, increment, grid,
                                               "%s/%s"%(outputnc, file_name2),"%s/%s"%(output, file_name))




                                    #command = comm.split(' ')

                                    if not os.path.exists(os.path.join(outputnc, "%s.nc4" % (file_namel))) and not os.path.exists(os.path.join(outputnc, "%s.nc" % (file_namel))) :
                                        try:
                                            creation = txt_to_netcdf.netcdf(output, "%s.txt"%( file_name), "%s.nc4"%file_namel, y3,y4, float(run_myLake_ISIMIP.get_latitude(lake, "D:/forcing_data", "EWEMBI",
                                                                        "historical")), float(run_myLake_ISIMIP.get_longitude(lake, "D:/forcing_data", "EWEMBI",
                                                                        "historical")),unit, variable)
                                            if creation == 1:
                                                tableau.loc[index, '%s.nc4' % variable] = 'Exists'
                                            else:
                                                tableau.loc[index, '%s.nc4' % variable] = 'Not created yet'
                                        except:
                                            tableau.loc[index, '%s.nc4' % variable] = 'Bug Not created yet'

                                        print(command)
                                        commandall = commandall + command.lower() + " && "
                                        #os.system(command)

                                    else:
                                        tableau.loc[index, '%s.nc4' % variable] = 'Exists'
                                else:
                                    tableau.loc[index, '%s.txt' % variable] = 'Not created'



                            # except:
                            #     print("bug")



                else:
                    tableau.loc[index, 'lake'] = f_lake
                    tableau.loc[index, 'model'] = modelid
                    tableau.loc[index, 'scenario'] = scenarioid
                    tableau.loc[index, 'Calibration'] = 'Not Done'

                index += 1

    commandall = commandall + "; } &"
    with open(os.path.join( 'listcommand_over100.txt'), 'w') as f:
        f.write(commandall)
    tableau.to_csv(r"all_variable_lakes_combinaison_update.csv", index=False)


def format_parallel():
    # for lake in full_lake_list:
    #     format(lake)
    Parallel(n_jobs=num_cores, verbose=10)(delayed(format)(lake) for lake in full_lake_list)

def format(lake):
    # models = [ "EWEMBI",
    #
    #           ]
    # scenarios = ["historical",
    #
    #              ]
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

                                        outputdir = os.path.join(output, "%s/%s/%s/%s/%s_%s" % (reg, lake, modelid, scenarioid, yinit, yend))

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
                                            file_name = "%s/%s/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                reg, lake, model_name, bias, climate, socio, sens,
                                                variable, region, timestep, y1, y2)
                                            outdir1 = os.path.join(output1, "%s/%s/" % (reg, lake))
                                            file_name2 = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                                model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                                y2)

                                            file_namel = file_name2.lower()
                                            outputnc = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"

                                            if (not os.path.exists(os.path.join(output1, "%s.txt" % file_name))and (not os.path.exists(os.path.join(outputnc, "%s.nc4" % (file_namel)))
                                                or not os.path.exists(os.path.join(outputnc, "%s.nc" % (file_namel))))) or vari in output_variables_depths:

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
                                        file_name = "%s/%s/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                            reg, lake, model_name, bias, climate, socio, sens,
                                            variable, region, timestep, y1, y2)
                                        outdir1 = os.path.join(output1, "%s/%s/" % (reg, lake))
                                        file_name2 = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                            model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                            y2)

                                        file_namel = file_name2.lower()
                                        outputnc = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"

                                        if (not os.path.exists(os.path.join(output1, "%s.txt" % file_name)) and (
                                                not os.path.exists(os.path.join(outputnc, "%s.nc4" % (file_namel)))
                                                or not os.path.exists(os.path.join(outputnc, "%s.nc" % (
                                        file_namel))))) or vari in output_variables_depths:

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


                                file_name = "%s/%s/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                    reg, lake, model_name, bias, climate, socio, sens,
                                    variable, region, timestep, y1, y2)

                                outdir1 = os.path.join(output1, "%s/%s/" % (reg, lake))
                                file_name2 = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                                    model_name, bias, climate, socio, sens, variable, lake, timestep, y1,
                                    y2)

                                file_namel = file_name2.lower()
                                outputnc = r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP"

                                if (not os.path.exists(os.path.join(output1, "%s.txt" % file_name)) and (
                                        not os.path.exists(os.path.join(outputnc, "%s.nc4" % (file_namel)))
                                        or not os.path.exists(os.path.join(outputnc, "%s.nc" % (
                                file_namel))))) or vari in output_variables_depths:

                                    if not os.path.exists(outdir1):
                                        os.makedirs(outdir1)

                                    data_set.to_csv(
                                        os.path.join(output1, "%s.txt" % file_name),
                                        header=None, index=None, sep=' ', mode='w')

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

                                    if (not os.path.exists(os.path.join(output1,
                                                                       "%s.nc4" % (
                                                                       file_name))) and os.path.exists(
                                            os.path.join(output1,
                                                         "%s.txt" % (file_name)))) or vari in output_variables_depths:
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
                                                #     file_name = "%s/%s/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
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
                                #                     file_name="%s/%s/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
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
                                #                 file_name = "%s/%s/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
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


                        print("observations/{}/{}/{}_hypsometry2.csv".format(reg, lake, lake))
                        if os.path.exists("observations/{}/{}/{}_hypsometry2.csv".format(reg, lake, lake)):
                            with open("observations/{}/{}/{}_hypsometry2.csv".format(reg, lake, lake)) as obs:
                                reader = list(csv.reader(obs))
                                prefix = f_lake[:3]
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

                        if os.path.exists("D:\output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(reg,lake)):
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
    for lake in full_lake_list:
        input_files_loop(lake)
    #Parallel(n_jobs=num_cores, verbose=10)(delayed(input_files_loop)(lake) for lake in full_lake_list)

def input_files_loop(lake):


    corrected_names = ["Allequash_Lake", "Big_Muskellunge_Lake", "Black_Oak_Lake", "Burley_Griffin", "Crystal_Bog",
                       "Crystal_Lake",
                       "Dickie_Lake", "Eagle_Lake", "Ekoln_basin_of_Malaren", "Esthwaite_Water",
                       "Falling_Creek",
                       "Fish_Lake", "Great_Pond", "Green_Lake", "Harp_Lake", "Laramie_Lake", "Lower_Zurich", "Mt_Bold",
                       "Nohipalo_Mustjarv", "Nohipalo_Valgejarv", "Okauchee_Lake", "Rappbode_Reservoir",
                       "Sau_Reservoir",
                       "Sparkling_Lake", "Toolik_Lake", "Trout_Bog", "Trout_Lake", "Two_Sisters_Lake"]


    f_lake = lake
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
        ncdf_file = ncdf.Dataset( "D:/forcing_data/hurs_{}_{}_{}.allTS.nc".format("EWEMBI", "historical", lake), "r",
            format="NETCDF4")

        if reg == None:
            print("Cannot find {}'s region".format(lake))
            return None
        else:
            with open("observations/{}/{}/{}_hypsometry2.csv".format(reg, lake, lake)) as obs:
                reader = list(csv.reader(obs))
                prefix = reader[1][0][3:]
        outputdir = run_myLake_ISIMIP.init_info(lake, "observations/{}/{}".format(reg, lake))
        for model in models:
            for scenario in scenarios:

                if os.path.exists("D:\output/{}/{}/{}/{}/RunComplete".format(reg, lake, model, scenario)):
                    print("{} {} {} Run is already completed.\n".format(lake, model, scenario))

                elif not os.path.exists(
                        "D:\output/{}/{}/EWEMBI/historical/Calibration_Complete1.txt".format(reg, lake)):# and os.path.exists(
                        #"input/{}/{}/{}_{}_{}_input".format(reg, prefix, prefix, model, scenario)):

                    try:
                        if (model == "EWEMBI" and scenario == "historical") or model !="EWEMBI":
                            print(outputdir.get("outdir")+r"\{}_{}_{}_input".format(reg,lake[:3], model, scenario))
                            print(outputdir.get("outdir")+ "\{}_par".format( lake[:3]))

                            #if 1==1:
                            if not (os.path.exists( outputdir.get("outdir")+ r"\{}_{}_{}_input".format(lake[:3], model, scenario))) \
                                    or not (os.path.exists(outputdir.get("outdir")+r"\{}_par".format(lake[:3]))) \
                                    or not (os.path.exists(outputdir.get("outdir")+r"\{}_init".format(lake[:3]))):
                                try:

                                    run_myLake_ISIMIP.generate_input_files("observations/{}/{}".format(reg, lake), lake, f_lake,
                                                               "D:/forcing_data", run_myLake_ISIMIP.get_longitude(f_lake, "D:/forcing_data",model,scenario),
                                                               run_myLake_ISIMIP.get_latitude(f_lake, "D:/forcing_data",model,scenario), model, scenario,reg)
                                except:
                                    print("missing data")
                                if not (os.path.exists( r"D:\output/{}/{}/{}/{}/Observed_Temperatures.csv".format(reg, lake, model,scenario))):
                                    if (os.path.exists( r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations/{}/{}\{}_temp_daily.csv".format(reg, lake, lake))):
                                        try:
                                            myLake_post.temperatures_by_depth("observations/{}/{}".format(reg, lake),
                                                                              lake,
                                                                              "D:\output/{}/{}/{}/{}".format(reg, lake,
                                                                                                             model,
                                                                                                             scenario),
                                                                              model, scenario)
                                            print("obsertvation done")
                                        except:
                                            print("missing data")
                                    else:
                                        print('no daily data for %s' % lake)


                            else:
                                print('Already done')


                                if 1==1:# not (os.path.exists(r"D:\output/{}/{}/{}/{}/Observed_Temperatures.csv".format(reg, lake, model, scenario))):
                                    if (os.path.exists(
                                            r"C:\Users\macot620\Documents\GitHub\Fish_niche\ISIMIP\observations/{}/{}\{}_temp_daily.csv".format(
                                                reg, lake, lake))):
                                        try:
                                            myLake_post.temperatures_by_depth("observations/{}/{}".format(reg, lake), lake,
                                                                  "D:\output/{}/{}/{}/{}".format(reg, lake, model, scenario),model,scenario)
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
                    except:
                       print("problem when doing %s %s %s" % (lake, model, scenario))

                    if model == "EWEMBI":
                        if scenario == "historical":
                            if not os.path.exists(outputdir.get("outdir")+"/{}_{}_{}_input".format(lake[:3],model,scenario)) or not os.path.exists(outputdir.get("outdir")+"/{}_init".format(lake[:3])) or not os.path.exists(outputdir.get("outdir")+"/{}_par".format(lake[:3])):
                                print("not all initial files existing for %s %s %s" % ( model, scenario,lake))
                    else:
                        if not os.path.exists(outputdir.get("outdir")+"/{}_{}_{}_input".format(lake[:3],model,scenario)) or not os.path.exists(outputdir.get("outdir")+"/{}_init".format(lake[:3])) or not os.path.exists(outputdir.get("outdir")+"/{}_par".format(lake[:3])):
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

    for lake in full_lake_list:
        model_scenario_loop(lake)
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
        try:
            with open("observations/{}/{}/{}_hypsometry2.csv".format(reg,lake, lake)) as obs:
                reader = list(csv.reader(obs))
                prefix = reader[1][0][3:]
                getprefix=True
        except:
            getprefix = False
        outputdir = run_myLake_ISIMIP.init_info(lake, "observations/{}/{}".format(reg, lake))
        if getprefix is True:
            if os.path.exists("D:\output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(reg, lake)):

                for model in models:
                    for scenario in scenarios:

                        print(outputdir.get("outdir")+"/{}_{}_{}_input".format(prefix,model,scenario))
                        if os.path.exists("D:\output/{}/{}/{}/{}/RunComplete2".format(reg, lake, model, scenario)):
                             print("{} {} {} Run is already completed.\n".format(lake, model, scenario))

                        elif os.path.exists("D:\output/{}/{}/EWEMBI/historical/Calibration_Complete.txt".format(reg, lake)) :
                            if os.path.exists(outputdir.get("outdir")+"/{}_{}_{}_input".format(prefix,model,scenario)):
                                try:
                                    print("start")
                                    run_myLake_ISIMIP.run_myLake(run_myLake_ISIMIP.init_info(lake, "observations/{}/{}".format(reg, lake)).get("outdir"), reg, lake, model, scenario)
                                    print("Run of {} {} {} Completed.\n".format(lake, model, scenario))
                                except:
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
    with open(report, 'w') as f:
        f.write('\nrunning _parallel\n' )
        f.close()

    run_calibrations(full_lake_list[0])
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
        with open("{}/observations/{}/{}/{}_hypsometry2.csv".format(file,reg, lake, lake)) as obs:
            reader = list(csv.reader(obs))
            prefix = reader[1][0][3:]
    optimisation_completed_unsuccessfull = []
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
    #mylake_parallel()
    #format_parallel()
    format_nc4()

    #run_calibrations("Langtjern")
    #run_calibrations(full_lake_list[1])
    #revision()
    #calibration_parallel()
    #input_files_loop("Langtjern")
    #input_files_parallel()

    #calibration_parallel()
    #
