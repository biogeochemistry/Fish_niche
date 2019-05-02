# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:24:05 2019

@author: Administrateur
"""
from test import thermocline
import math
from joblib import Parallel, delayed
import multiprocessing
import datetime
import time
from os import path
import pandas as pd
num_cores = multiprocessing.cpu_count()  # needs to be modified if you want to choose the number of cores used.

cordexfolder = 'G:cordex'  # needs to be changed depending where the meteorological files are.
outputfolder = '../output'
timeformat = '%Y-%m-%d %H:%M:%S'
variables = ['clt', 'hurs', 'tas', 'rsds', 'ps', 'pr', 'sfcWind']
models = {1: ('ICHEC-EC-EARTH', 'r1i1p1_KNMI-RACMO22E_v1_day'),  # MC 2018-05-16
          2: ('ICHEC-EC-EARTH', 'r3i1p1_DMI-HIRHAM5_v1_day'),
          3: ('MPI-M-MPI-ESM-LR', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day'),
          4: ('MOHC-HadGEM2-ES', 'r1i1p1_SMHI-RCA4_v1_day'),
          5: ('IPSL-IPSL-CM5A-MR', 'r1i1p1_IPSL-INERIS-WRF331F_v1_day'),
          6: ('CNRM-CERFACS-CNRM-CM5', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day')}
scenarios = {1: (('historical', '19710101', '19751231'), ('historical', '19760101', '19801231')),
             2: (('historical', '20010101', '20051231'), ('rcp45', '20060101', '20101231')),
             3: (('rcp45', '20310101', '20351231'), ('rcp45', '20360101', '20401231')),
             4: (('rcp45', '20610101', '20651231'), ('rcp45', '20660101', '20701231')),
             5: (('rcp45', '20910101', '20951231'), ('rcp45', '20960101', '21001231')),
             6: (('rcp85', '20310101', '20351231'), ('rcp85', '20360101', '20401231')),
             7: (('rcp85', '20610101', '20651231'), ('rcp85', '20660101', '20701231')),
             8: (('rcp85', '20910101', '20951231'), ('rcp85', '20960101', '21001231'))}

def montly_temp_light_profile(listofmodels, listofscenarios, lakelistfile):    
    nbrsuccess = 0
    nbrtested = 0
    st = datetime.datetime.fromtimestamp(time.time()).strftime(timeformat)
    with open(lakelistfile, 'rU') as f:
        lines = f.readlines()
        nlines = len(lines)

    for model in listofmodels:
        for scenario in listofscenarios:
            m0, m1 = models[model]
            s0, s1 = scenarios[scenario]
            
            with open(csvf, 'rU') as f:
                lines = f.readlines()
                nlines = len(lines)
                ii = range(1, nlines)
            listlaketempepday,listlaketemphyday,listlakelamepday,listlakelamhyday = [],[],[],[]
            for i in ii:
                listtempepi,listtemphypo,listlamepi,listlamhypo = [],[],[],[]
                lake_id, subid, name, ebh, area, depth, longitude, latitude, volume, mean_depth, sediment, mean_calculated = lines[i].strip().split(',')
            
                eh = eh[2:] if eh[:2] == '0x' else eh
                while len(eh)< 6:
                    eh = '0' + eh
                
                d1, d2, d3 = eh[:2], eh[:4], eh[:6]

                outdir = path.join(outputfolder, d1, d2, d3,
                                 'EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(m1, exA, exB, m2, y1A, y2B))
                print(outdir)

                lambdazt = pd.read_csv(path.join(outdir, 'lambdazt.csv'), header=None)
                Tzt = pd.read_csv(path.join(outdir, 'Tzt.csv'), header=None)
                zlen = int(depth)
                wtr = range(1,zlen)
                wtr = str(wtr)
                with open(Tzt, 'rU') as tt:
                    liness = tt.readlines()
                    nliness = len(liness)
                    jj = range(1, nliness)
                with open(lambdazt, 'rU') as tlambda:
                    linesl = tlambda.readlines()
                    
                for j in jj:
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
            
            lakems = '%s_%s_%s_EUR-11_%s_%s-%s_%s_%s0101-%s1231' %(d1, d2, d3, m1, exA, exB, m2, y1A, y2B))
            
            with open("output_%s_temp_epi.csv"%(lakems), "wb") as f:
                writer = csv.writer(f)
                writer.writerows(listlaketempepday)
            with open("output_%s_temp_hypo.csv"%(lakems), "wb") as f1:
                writer = csv.writer(f1)
                writer.writerows(listlaketemphyday)
            with open("output_%s_lam_epi.csv"%(lakems), "wb") as f2:
                writer = csv.writer(f2)
                writer.writerows(listlakelamepday)
            with open("output_%s_lam_hypo.csv"%(lakems), "wb") as f3:
                writer = csv.writer(f3)
                writer.writerows(listlakelamhyday)
                
if __name__ == '__main__':
    montly_temp_light_profile([2], [2], lakelistfile)