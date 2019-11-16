# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:07:38 2019

@author: maria
"""
import os
import pandas as pd
import run_mylakeGoran
import math
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()-6 
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



outputfolder = r"D:\Fish_niche\output1"
lake_list = r"D:\Fish_niche\lakes\2017SwedenList.csv"

def loop_through_lakes_list(i, lines, modelid, scenarioid,swa_b1):
    """Changes value of some parameters and run run_mylakeGoran.runlake()

    Args:
        i: index of the list of lskes
        lines: list of lakes
        modelid: model id (one of the keys of the dictionary "models")
        scenarioid: scenario id (one of the keys of the dictionary "scenarios")
    """

    lake_id, subid, name, ebh, area, depth, longitude, latitude, volume, mean_depth, sediment, mean_calculated\
        = lines[i].strip().split(',')

    print('running lake %s' % ebh)
    if mean_depth != '' and (float(depth) - float(mean_depth)) > 0:
        #swa_b1 = math.exp(-0.95670 * math.log(float(mean_depth)) + 1.36359)
        # swa_b1 = OLD EQUATION: 3.6478*float(depthmean)**-0.945
        #                        0.796405*math.exp(-0.016045*float(depth))
        #                       -0.3284*math.log(float(depth)) + 1.6134
        #                       -0.8547*math.log(math.log(depth)) + 1.4708
        #                       3.0774*math.exp(-0.6432*math.log(float(depth)))

        k_BOD = math.exp(-0.25290 * float(mean_depth) - 1.36966)
        # k_BOD = OLD EQUATION: 0.268454*math.log(float(depth))**-3.980304
        #                     0.291694*math.log(float(depth))**-4.013598#0.2012186 * math.log(float(depth)) ** -3.664774
    else:
        #swa_b1 = math.exp(-0.95670 * math.log(float(volume)/float(area)) + 1.36359)
        k_BOD = math.exp(-0.25290 * (float(volume)/float(area)) - 1.36966)

    
    
    print('BOD %s' % k_BOD)
    k_SOD = math.exp(-0.06629 * float(depth) + 0.64826 * math.log(float(area)) - 3.13037)
    # k_SOD = OLD EQUATION: 13069.873528*math.exp(-1.760776*math.log(float(depth)))
    #                       11838.3*math.exp(-1.69321*math.log(float(depth)))
    print('swa %s, SOD %s' % (swa_b1, k_SOD))

    I_scDOC = math.log((swa_b1 + 0.727)/0.3208008880)/(2 * 0.1338538345)  # needs to be modified if swa_b0 changes
    print('IDOC %s' % I_scDOC)


    run_mylakeGoran.runlake(modelid, scenarioid, ebh.strip('"'), int(subid), float(depth), float(area),
                            float(longitude), float(latitude), k_BOD, swa_b1, k_SOD, I_scDOC)
    
    
 



if __name__ == '__main__':

    data_lake = pd.read_csv(lake_list,encoding='latin')
    list_selected_lake = data_lake["ebhex"].tolist()
    listmeanice = []
    model = 2
    scenario=1
    exA, y1A, exB, y1B = scenarios[scenario]
    m1, m2 = models[model]
    y2A = y1A + 4
    y2B = y1B + 4
    test = pd.DataFrame()
    
    
    
    
    with open(lake_list, 'rU') as f: 
        lines = f.readlines()
        nlines = len(lines)
        ii = range(1, nlines)
    data_lake = pd.read_csv(lake_list,encoding='latin')
        
    for j in [0.001,0.01,0.1,1.0,10,100]:
    #for i in ii:
    #    loop_through_lakes_list(i,lines,model_id,scenario_id)
        listmeanice=[]
        print(j)
        for i in ii:
            loop_through_lakes_list(i, lines, model, scenario,j)
        Parallel(n_jobs=num_cores)(delayed(loop_through_lakes_list)(i, lines, model, scenario,j) for i in ii)
        
        list_selected_lake = data_lake["ebhex"].tolist()
        for eh in list_selected_lake:
            eh = eh[2:] if eh[:2] == '0x' else eh
            while len ( eh ) < 6:
                eh = '0' + eh
            d1, d2, d3 = eh[:2], eh[:4], eh[:6]
            outdir = os.path.join ( outputfolder, d1, d2, d3,
                                'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (m1, exA, exB, m2, y1A, y2B) )
    
        
            ice = pd.read_csv(os.path.join ( outdir, 'His.csv' ),names=['1','2','3','4','5','6','7','8'])
            meanice = (ice['7'].sum())/10
            listmeanice.append(meanice)
            print(listmeanice)
        test['meanice_%s'%(j)] = listmeanice
    test.to_csv('testiceall.csv',index=False)
    
