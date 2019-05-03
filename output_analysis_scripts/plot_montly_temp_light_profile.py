# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:24:05 2019

@author: Administrateur
"""


import multiprocessing
import datetime
import time
from os import path
import pandas as pd
import csv
import rpy2.robjects as ro
num_cores = multiprocessing.cpu_count()  # needs to be modified if you want to choose the number of cores used.

cordexfolder = 'G:cordex'  # needs to be changed depending where the meteorological files are.
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
            listlaketempepday,listlaketemphyday,listlakelamepday,listlakelamhyday = [],[],[],[]
            for i in ii:
                listtempepi,listtemphypo,listlamepi,listlamhypo = [],[],[],[]
                lake_id, subid, name, eh, area, depth, longitude, latitude, volume, mean_depth, sediment, mean_calculated = lines[i].strip().split(',')
            
                eh = eh[2:] if eh[:2] == '0x' else eh
                while len(eh)< 6:
                    eh = '0' + eh
                
                d1, d2, d3 = eh[:2], eh[:4], eh[:6]
                outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y1B+4))
                print(outdir)

                lambdazt = path.join(outdir, 'lambdazt.csv')
                Tzt = path.join(outdir, 'Tzt.csv')
                zlen = int(float(depth))
                wtr = range(1,zlen)
                wtr = str(wtr)
                with open(Tzt, 'rU') as tt:
                    liness = tt.readlines()
                    nliness = len(liness)
                    jj = range(1, nliness)
                with open(lambdazt, 'rU') as tlambda:
                    linesl = tlambda.readlines()
                    
                for j in jj:
                    print(wtr)
                    print(j)
                    depththermo = thermocline(wtr, j)
                    depththermo = int(((str(depththermo).strip().split(' '))[1].strip().split('.'))[0])
                     
                    listtemp = liness[j].strip().split(',')
                    tempfloat = [float(n) for n in listtemp if n]
                    tempavgep = sum(tempfloat[:depththermo])/len(tempfloat[:depththermo]) if tempfloat else '-'
                    tempavghy = sum(tempfloat[depththermo:])/len(tempfloat[:depththermo]) if tempfloat else '-'
                     
                    listlam = linesl[j].strip().split(',')
                    floatlam = [float(nl) for nl in listlam if nl]
                    lamavgep = sum(floatlam[:depththermo])/len(floatlam[:depththermo]) if floatlam else '-'
                    lamavghy = sum(floatlam[depththermo:])/len(floatlam[:depththermo]) if floatlam else '-'
                     
                    listtempepi.append(tempavgep)
                    listtemphypo.append(tempavghy)
                    listlamepi.append(lamavgep)
                    listlamhypo.append(lamavghy)
                listlaketempepday.append(listtempepi)
                listlaketemphyday.append(listtemphypo)
                listlakelamepday.append(listlamepi)
                listlakelamhyday.append(listlamhypo)
            
            lakems = '%s_%s_%s_EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(d1, d2, d3, m1, exA, exB, m2, y1A, y1B)
            
            with open("%s/output_%s_temp_epi.csv"%(outputfolder.lakems), "wb") as f:
                writer = csv.writer(f)
                writer.writerows(listlaketempepday)
            with open("%s/output_%s_temp_hypo.csv"%(outputfolder.lakems), "wb") as f1:
                writer = csv.writer(f1)
                writer.writerows(listlaketemphyday)
            with open("%s/output_%s_lam_epi.csv"%(outputfolder.lakems), "wb") as f2:
                writer = csv.writer(f2)
                writer.writerows(listlakelamepday) 
            with open("%s/output_%s_lam_hypo.csv"%(outputfolder.lakems), "wb") as f3:
                writer = csv.writer(f3)
                writer.writerows(listlakelamhyday)
                
if __name__ == '__main__':
    montly_temp_light_profile([2], [2], 'D:\\Fish_niche\\lakes\\2017SwedenList111.csv')