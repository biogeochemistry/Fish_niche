# !/usr/bin/env python3
"""
    File name: plot_montly_temp_light_profile.py
    Author: Marianne Cote
    Python Version: 3.6


"""

import multiprocessing
from matplotlib.gridspec import GridSpec
from statistics import mean
from os import path
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter, DayLocator, WeekdayLocator, MONDAY
import matplotlib as mpl
from datetime import datetime
import csv
import numpy as np
import pandas as pd
#import rpy2.robjects as ro
from math import exp
import matplotlib.pyplot as plt
num_cores = multiprocessing.cpu_count()  # needs to be modified if you want to choose the number of cores used.


outputfolder = 'E:\output-05-23-2019'
outputfolder = r'C:\Users\macot620\Documents\GitHub\Fish_niche\output'
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
                qst = path.join(outdir,'Qst.csv')
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
                if lake_id == "727":
                    print("test")
                Glo = pd.read_csv(inputlake,sep="\t", header= 1) 
                Glo = Glo.iloc[730:,:]
                Ice = pd.read_csv(His,header=None) 
                Qst = pd.read_csv(qst,header=None) 
                for j in jj:
                    
                    wts = ''.join(liness[j])
                    dp1 = ''.join(dp)[1:-1]
                    depththermo = (((str(thermocline(wts,dp1))).strip().split(' '))[1].strip().split('.')[0])
                    #print(depththermo) 
                    listtemp = liness[j].strip().split(',')
                    tempfloat = [float(n) for n in listtemp if n]
                        
                    listlambda = linesl[j].strip().split(',')
                    floatlambda = [float(nl) for nl in listlambda if nl]
                    lambdaavg = mean(floatlambda)
                    floatIz=[((Qst.iloc[j,0]*exp(-lambdaavg*x))*(1-int(Ice.iloc[j,6])))for x in range(1,int(float(depth))+1)]
                    
                    #depththermo = "Na"
                    if depththermo.isdigit():
                        depththermo = int(depththermo)
                        #listtemp = liness[j].strip().split(',')
                        #tempfloat = [float(n) for n in listtemp if n]
                        tempavgep = mean(tempfloat[:depththermo])
                        tempavghy = mean(tempfloat[depththermo:])
                        #listlambda = linesl[j].strip().split(',')
                        #floatlambda = [float(nl) for nl in listlambda if nl]
                        #lambdaavg = mean(floatlambda)
                        
                            #floatIz=[(Glo.iloc[j,3]**(-lambdaavg*x))*(1-int(Ice.iloc[j,6]))for x in range(1,int(float(depth))+1)]
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

def plot_montly_temp_light_profile(modelid, scenarioid, lakelistfile): 
    plt.rcParams.update({'font.size': 20})
    exA, y1A, exB, y1B = scenarios[scenarioid]
    m1, m2 = models[modelid]
    y2B = y1B + 4
    with open(lakelistfile, 'rU')as f:
        lakes = f.readlines()
        nlakes = len(lakes)
    
    for lakenum in np.arange(1, nlakes):
        lake_id, subid, name, eh, area, depth, longitude, latitude, volume, mean_depth, sediment,mean_calculated \
        = lakes[lakenum].strip().split(',')
        
        # getOutputPathFromEbHex
        eh = eh[2:] if eh[:2] == '0x' else eh
        
        while len(eh)< 6:
            eh = '0' + eh
        
        d1, d2, d3 = eh[:2], eh[:4], eh[:6]
        
        outdir = path.join(outputfolder, d1, d2, d3,'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
        print(outdir)

        # read *.cvs
        Par = pd.read_csv(path.join(outdir, 'Parzt.csv'), header=None)
        T = pd.read_csv(path.join(outdir, 'Tzt.csv'), header=None)
        
        months = MonthLocator()
        date = pd.date_range(start='2001-01-01', end='2010-12-31')
        
        Par['Date'] = date
        T['Date'] = date
        Par.set_index('Date', inplace=True)
        T.set_index('Date', inplace=True)
       
        #tt = pd.date_range(start='2000-01-01', end='2000-12-31')
        tt = pd.date_range('2000-01-01','2001-01-01' , freq='1M')-pd.offsets.MonthBegin(1)
        meantemp = T.groupby([T.index.month]).mean()
        meanpar =  Par.groupby([Par.index.month]).mean()
        stdtemp =  T.groupby([T.index.month]).std()
        stdpar =   Par.groupby([Par.index.month]).std()
        meantemp.to_csv("%s\%s_Monthly_Mean_T_2001-2010.csv"%(outputfolder,lake_id),header=False,index=False)
        meanpar.to_csv("%s\%s_Monthly_Mean_PPFD_2001-2010.csv"%(outputfolder,lake_id),header=False,index=False)
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
        
        
        fig1 = plt.figure(figsize=(22, 15))
        gs=GridSpec(2,30)
        x = np.arange(0, int(float(depth)))
        
        
        color_idx = np.linspace(0, 1, int(float(depth)))
        ax1 = fig1.add_subplot(gs[0,:28])
        for i,color_i in zip(x,color_idx):
            plt.plot_date(tt, meantemp.iloc[:,i],'-',color=plt.cm.coolwarm_r(color_i))
            
            
            #plt.fill_between(tt, meantemp + stdtemp, meantemp - stdtemp,  alpha='0.5')
        plt.text(0.01, 0.97,'Max depth: %s'%(depth),horizontalalignment='left',verticalalignment='top',transform = ax1.transAxes)
       
        ax1.set_xlim([datetime(2000, 1,1), datetime(2000, 12,1)])
        ax1.fmt_xdata = DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_locator(months)
        #ax1.xaxis.set_minor_locator(mondays)
        ax1.xaxis.set_major_formatter(weekFormatter)
        ax1.fmt_ydata = price
        ax1.yaxis.grid(True)
        
        plt.ylabel("Temperature [°C]")
                    
        #plt.xlabel("Date")
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.tight_layout()
        #plt.legend()
        
        ax2 = fig1.add_subplot(gs[1,:28])
        for i,color_i in zip(x,color_idx):
            plt.plot_date(tt, meanpar.iloc[:,i],'-',color=plt.cm.coolwarm_r(color_i))
            ax2.set_yscale('log')
           
        ax2.set_xlim([datetime(2000, 1, 1), datetime(2000, 12, 1)])
        ax2.fmt_xdata = DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_locator(months)
        #ax2.xaxis.set_minor_locator(mondays)
        ax2.xaxis.set_major_formatter(weekFormatter)
        ax2.fmt_ydata = price
        ax2.yaxis.grid(True)
        
        plt.ylabel("Photosynthetic photon flux density (PPFD)\n [μmol/m2/s]")
                
        plt.xlabel("Date")
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.tight_layout()
        #plt.legend()
        #fig1.savefig("%s\Figure_mean_temp_%s.png" %(outputfolder,lake_id))
        #plt.show()
#        
        #fig2 = plt.figure(figsize=(20, 10))
        
        #plt.fill_between(tt, meanpar + stdpar, meanpar - stdpar,  alpha='0.5')
        #plt.text(0.01, 0.97,'Max depth: %s'%(depth),horizontalalignment='left',verticalalignment='top',transform = ax2.transAxes)
        cmap = mpl.cm.get_cmap('coolwarm_r', int(float(depth)))
        
        norm = mpl.colors.BoundaryNorm(np.arange(0.5,int(float(depth))+1,1), cmap.N)
        ax3=fig1.add_subplot(gs[:,28])
        if int(float(depth))>20:
            steps = 20
        else: steps=int(float(depth))
        mpl.colorbar.ColorbarBase(ax3, cmap=cmap,norm=norm,orientation='vertical',format=mpl.ticker.FormatStrFormatter('%.d'),ticks=np.linspace( 1,int(float(depth)), steps, endpoint=True))  
        
        plt.ylabel("Depth(m)")
        print(lake_id)
        fig1.savefig("%s\Figure_%s_Montly_Mean_PPFD_T_2001-2010.png" %(outputfolder,lake_id))
        #plt.show()
              
def plot_montly_thermo_temp_light_profile(listofmodels, listofscenarios, lakelistfile): 
    months = MonthLocator()
    datasheet1 = pd.read_csv('E:\output-05-23-2019\mean_temp_hypo.csv')
    datasheet2 = pd.read_csv('E:\output-05-23-2019\mean_temp_epi.csv')
    datasheet3 = pd.read_csv('E:\output-05-23-2019\mean_Iz_hypo.csv')
    datasheet4 = pd.read_csv('E:\output-05-23-2019\mean_Iz_epi.csv')                     
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
        
        fig1.savefig("%s\Figure_mean_temp_%s.png" %(outputfolder,meantemph.columns.get_values()[i]))
        
        
        fig2 = plt.figure(figsize=(20, 10))
        ax2 = plt.subplot(211)
        plt.plot_date(tt, meanlame.iloc[:,i],'-',color='red',label='Epilimnion')
        plt.fill_between(tt, meanlame.iloc[:,i] + stdlame.iloc[:,i], meanlame.iloc[:,i] - stdlame.iloc[:,i], color='red', alpha='0.5')
        #plt.fill_between(tt, meanlame.iloc[:,i], meanlame.iloc[:,i], color='red', alpha='0.5')
        plt.plot_date(tt, meanlamh.iloc[:,i],'-',color='blue',label='Hypolimnion')
        plt.fill_between(tt, meanlamh.iloc[:,i] + stdlamh.iloc[:,i], meanlamh.iloc[:,i] - stdlamh.iloc[:,i], color='blue', alpha='0.5')
        #plt.fill_between(tt, meanlamh.iloc[:,i], meanlamh.iloc[:,i], color='blue', alpha='0.5')
        #print(meanlamh.iloc[:,i])
    
        #plt.ylim(-1, 20)
        
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
        fig2.savefig("%s\Figure_mean_Iz_%s.png" %(outputfolder,meanlamh.columns.get_values()[i]))
    plt.show()
                
if __name__ == '__main__':
    plot_montly_temp_light_profile(2, 2, r'C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList.csv')