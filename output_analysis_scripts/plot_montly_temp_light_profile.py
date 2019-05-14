# !/usr/bin/env python3
"""
    File name: plot_montly_temp_light_profile.py
    Author: Marianne Cote
    Python Version: 3.6


"""

import multiprocessing
from statistics import mean
from os import path
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter, DayLocator, WeekdayLocator, MONDAY
# import matplotlib as mpl
from datetime import date, datetime
import csv
import pandas as pd
import rpy2.robjects as ro
import matplotlib.pyplot as plt
num_cores = multiprocessing.cpu_count()  # needs to be modified if you want to choose the number of cores used.


outputfolder = 'E:\output-30-03-2019'
timeformat = '%Y-%m-%d %H:%M:%S'
variables = ['clt', 'hurs', 'tas', 'rsds', 'ps', 'pr', 'sfcWind']
models = {1: ('ICHEC-EC-EARTH', 'r1i1p1_KNMI-RACMO22E_v1_day'),
          2: ('ICHEC-EC-EARTH', 'r3i1p1_DMI-HIRHAM5_v1_day'),
          3: ('MPI-M-MPI-ESM-LR', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day'),
          4: ('MOHC-HadGEM2-ES', 'r1i1p1_SMHI-RCA4_v1_day'),
          5: ('IPSL-IPSL-CM5A-MR', 'r1i1p1_IPSL-INERIS-WRF331F_v1_day'),
          6: ('CNRM-CERFACS-CNRM-CM5', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day')}
scenarios = {1: ('historical', 1971, 'historical', 1976),
             2: ('historical', 2001, 'rcp45', 2006),
             3: ('rcp45', 2031, 'rcp45', 2036),
             4: ('rcp45', 2061, 'rcp45', 2066),
             5: ('rcp45', 2091, 'rcp45', 2096),
             6: ('rcp85', 2031, 'rcp85', 2036),
             7: ('rcp85', 2061, 'rcp85', 2066),
             8: ('rcp85', 2091, 'rcp85', 2096)}
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
def thermocline(temp,depths):
    r=ro.r
    r.source(r'montly_temperature_light_profiles.R' )
    p=r.calculThermocline(temp,depths)
    return p
def montly_temp_light_profile(listofmodels, listofscenarios, lakelistfile):    

    for model in listofmodels:
        for scenario in listofscenarios:
            exA, y1A, exB, y1B = scenarios[scenario]
            m1, m2 = models[model]

            with open(lakelistfile, 'rU') as f:
                lines = f.readlines()
                nlines = len(lines)
                ii = range(1, nlines)
            listlaketempepday,listlaketemphyday,listlakeIzepday,listlakeIzhyday = [],[],[],[]
            for i in ii:
                
                lake_id, subid, name, eh, area, depth, longitude, latitude, volume, mean_depth, sediment, mean_calculated = lines[i].strip().split(',')
            
                listtempepi,listtemphypo,listIzepi,listIzhypo= [lake_id],[lake_id],[lake_id],[lake_id]
                eh = eh[2:] if eh[:2] == '0x' else eh
                while len(eh)< 6:
                    eh = '0' + eh
                
                d1, d2, d3 = eh[:2], eh[:4], eh[:6]
                outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y1B+4))
                
                print(lake_id)
                lambdazt = path.join(outdir, 'lambdazt.csv')
                inputlake = path.join(outdir,'2017input')
                His = path.join(outdir,'His.csv')
                Tzt = path.join(outdir, 'Tzt.csv')
                zlen = int(float(depth))
                dp = list(range(1,zlen+1))
                dp = str(dp)
                with open(Tzt, 'rU') as tt:
                    liness = tt.readlines()
                    nliness = len(liness)
                    jj = range(0, nliness)
                with open(lambdazt, 'rU') as tlambda:
                    linesl = tlambda.readlines()
                Glo = pd.read_csv(inputlake,sep="\t", header= 1) 
                Ice = pd.read_csv(His,sep="\t", header= None) 
                for j in jj:
                    
                    wts = ''.join(liness[j])
                    dp1 = ''.join(dp)[1:-1]
                    depththermo = (((str(thermocline(wts,dp1))).strip().split(' '))[1].strip().split('.')[0])
                    print(depththermo) 
                    
                    if depththermo.isdigit():
                        depththermo = int(depththermo)
                        listtemp = liness[j].strip().split(',')
                        tempfloat = [float(n) for n in listtemp if n]
                        tempavgep = mean(tempfloat[:depththermo])
                        tempavghy = mean(tempfloat[depththermo:])
                        listlambda = linesl[j].strip().split(',')
                        floatlambda = [float(nl) for nl in listlambda if nl]
                        lambdaavg = mean(floatlambda)
                        floatIz = [Glo.iloc[j,3]**(-lambdaavg*x)(1-Ice.iloc[j,6])for x in range(1,int(depth)+1)]
                        Izavgep = mean(floatIz[:depththermo])
                        Izavghy = mean(floatIz[depththermo:])
                                      
                    else:
                        tempavgep = tempfloat[0]
                        tempavghy = tempfloat[-1]
                        Izavgep = floatIz[0]
                        Izavghy = floatIz[-1]
                    
                    
                    listtempepi.append(tempavgep)
                    listtemphypo.append(tempavghy)
                    listIzepi.append(Izavgep)
                    listIzhypo.append(Izavghy)
                    
                listlaketempepday.append(listtempepi)
                listlaketemphyday.append(listtemphypo)
                listlakeIzepday.append(listIzepi)
                listlakeIzhyday.append(listIzhypo)
            
            lakems = 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %( m1, exA, exB, m2, y1A, y1B+4)
            
            with open("%s/output_%s_temp_epi.csv"%(outputfolder, lakems), "w",newline='') as f:
                writer = csv.writer(f) 
                writer.writerows(listlaketempepday)
            with open("%s/output_%s_temp_hypo.csv"%(outputfolder,lakems), "w",newline='') as f1:
                writer = csv.writer(f1)
                writer.writerows(listlaketemphyday)
            with open("%s/output_%s_Iz_hypo.csv"%(outputfolder,lakems), "w",newline='') as f2:
                writer = csv.writer(f2)
                writer.writerows(listlakeIzhyday) 
            with open("%s/output_%s_Iz_epi.csv"%(outputfolder,lakems), "w",newline='') as f2:
                writer = csv.writer(f2)
                writer.writerows(listlakeIzepday) 
           
def price(x):
    return '$%1.2f' % x                
def plot_montly_temp_light_profile(listofmodels, listofscenarios, lakelistfile): 
    months = MonthLocator()
    datasheet1 = pd.read_csv('E:\output-30-03-2019\mean_temp_hypo.csv')
    datasheet2 = pd.read_csv('E:\output-30-03-2019\mean_temp_epi.csv')
    datasheet3 = pd.read_csv('E:\output-30-03-2019\mean_Iz_hypo.csv')
    datasheet4 = pd.read_csv('E:\output-30-03-2019\mean_Iz_epi.csv')                     
    datasheet1['Date'] = pd.to_datetime(datasheet1['Date'], format="%Y-%m-%d")
    datasheet2['Date'] = pd.to_datetime(datasheet2['Date'], format="%Y-%m-%d")
    datasheet3['Date'] = pd.to_datetime(datasheet3['Date'], format="%Y-%m-%d")
    datasheet4['Date'] = pd.to_datetime(datasheet4['Date'], format="%Y-%m-%d")
    datasheet1.set_index('Date', inplace=True)
    datasheet2.set_index('Date', inplace=True)
    datasheet3.set_index('Date', inplace=True)
    datasheet4.set_index('Date', inplace=True)
    
    tt = pd.date_range(start='2000-01-01', end='2000-12-31')
    meantemph = datasheet1.groupby([datasheet1.index.month, datasheet1.index.day]).mean()
    meantempe = datasheet2.groupby([datasheet2.index.month, datasheet2.index.day]).mean()
    meanlamh =  datasheet3.groupby([datasheet3.index.month, datasheet3.index.day]).mean()
    meanlame =  datasheet4.groupby([datasheet4.index.month, datasheet4.index.day]).mean()
    stdtemph =  datasheet1.groupby([datasheet1.index.month, datasheet1.index.day]).std()
    stdtempe =  datasheet2.groupby([datasheet2.index.month, datasheet2.index.day]).std()
    stdlamh =   datasheet3.groupby([datasheet3.index.month, datasheet3.index.day]).std()
    stdlame =   datasheet4.groupby([datasheet4.index.month, datasheet4.index.day]).std()
    
#    medtemph = datasheet1.groupby([datasheet1.index.month, datasheet1.index.day]).quantile(0.5)
#    medtempe = datasheet2.groupby([datasheet2.index.month, datasheet2.index.day]).quantile(0.5)
#    medlamh =  datasheet3.groupby([datasheet3.index.month, datasheet3.index.day]).quantile(0.5)
#    medlame =  datasheet4.groupby([datasheet4.index.month, datasheet4.index.day]).quantile(0.5)
#    
#    mintemph =  datasheet1.groupby([datasheet1.index.month, datasheet1.index.day]).quantile(0.25)
#    mintempe =  datasheet2.groupby([datasheet2.index.month, datasheet2.index.day]).quantile(0.25)
#    minlamh =   datasheet3.groupby([datasheet3.index.month, datasheet3.index.day]).quantile(0.25)
#    minlame =   datasheet4.groupby([datasheet4.index.month, datasheet4.index.day]).quantile(0.25)
#    
#    maxtemph =  datasheet1.groupby([datasheet1.index.month, datasheet1.index.day]).quantile(0.75)
#    maxtempe =  datasheet2.groupby([datasheet2.index.month, datasheet2.index.day]).quantile(0.75)
#    maxlamh =   datasheet3.groupby([datasheet3.index.month, datasheet3.index.day]).quantile(0.75)
#    maxlame =   datasheet4.groupby([datasheet4.index.month, datasheet4.index.day]).quantile(0.75)
    
    for i in range(0,210):
        fig1 = plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(211)
        plt.plot_date(tt, meantempe.iloc[:,i],'-',color='red',label='Epilimnion')
        plt.fill_between(tt, meantempe.iloc[:,i] + stdtempe.iloc[:,i], meantempe.iloc[:,i] - stdtempe.iloc[:,i], color='red', alpha='0.5')
        #plt.fill_between(tt, meantempe.iloc[:,i], meantempe.iloc[:,i], color='red', alpha='0.5')
        
        plt.plot_date(tt, meantemph.iloc[:,i],'-',color='blue',label='Hypolimnion')
        plt.fill_between(tt, meantemph.iloc[:,i] + stdtemph.iloc[:,i], meantemph.iloc[:,i] - stdtemph.iloc[:,i], color='blue', alpha='0.5')
        #plt.fill_between(tt, meantemph.iloc[:,i], meantemph.iloc[:,i], color='blue', alpha='0.5')
        #print(meantemph.iloc[:,i])
    
        plt.ylim(-1, 20)
        
        ax1.set_xlim([datetime(2000, 1, 1), datetime(2000, 12, 31)])
        ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_locator(months)
        ax1.xaxis.set_minor_locator(mondays)
        ax1.xaxis.set_major_formatter(weekFormatter)
        ax1.fmt_ydata = price
        ax1.yaxis.grid(True)
        plt.ylabel("Temp")
        
        plt.xlabel("Date")
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.tight_layout()
        plt.legend()
        
        fig1.savefig("E:\output-30-03-2019\Figure1_mean_temp_%s.png" %(meantemph.columns.get_values()[i]))
        
        
        fig2 = plt.figure(figsize=(20, 10))
        ax2 = plt.subplot(211)
        plt.plot_date(tt, meanlame.iloc[:,i],'-',color='red',label='Epilimnion')
        plt.fill_between(tt, meanlame.iloc[:,i] + stdlame.iloc[:,i], meanlame.iloc[:,i] - stdlame.iloc[:,i], color='red', alpha='0.5')
        #plt.fill_between(tt, meanlame.iloc[:,i], meanlame.iloc[:,i], color='red', alpha='0.5')
        plt.plot_date(tt, meanlamh.iloc[:,i],'-',color='blue',label='Hypolimnion')
        plt.fill_between(tt, meanlamh.iloc[:,i] + stdlamh.iloc[:,i], meanlamh.iloc[:,i] - stdlamh.iloc[:,i], color='blue', alpha='0.5')
        #plt.fill_between(tt, meanlamh.iloc[:,i], meanlamh.iloc[:,i], color='blue', alpha='0.5')
        #print(meanlamh.iloc[:,i])
    
        plt.ylim(-1, 20)
        
        ax2.set_xlim([datetime(2000, 1, 1), datetime(2000, 12, 31)])
        ax2.fmt_xdata = DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_locator(months)
        ax2.xaxis.set_minor_locator(mondays)
        ax2.xaxis.set_major_formatter(weekFormatter)
        ax2.fmt_ydata = price
        ax2.yaxis.grid(True)
        plt.ylabel("PAR")
        
        plt.xlabel("Date")
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.tight_layout()
        plt.legend()
        print(meanlamh.columns.get_values()[i])
        fig2.savefig("E:\output-30-03-2019\Figure1_mean_par_%s.png" %(meanlamh.columns.get_values()[i]))
    plt.show()
                
if __name__ == '__main__':
    plot_montly_temp_light_profile([2], [2], 'D:\\Fish_niche\\lakes\\2017SwedenList.csv')