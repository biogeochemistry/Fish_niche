# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:27:30 2019

@author: maria
"""
import os
import pandas as pd
import time


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



outputfolder = r"F:\output"#r"C:\Users\macot620\Documents\GitHub\Fish_niche\output"#
lake_list = r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList_only_validation_12lakes.csv"
sjolista = r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\sjolista.xlsx"

exA, y1A, exB, y1B = scenarios[2]
m1, m2 = models[2]
y2A = y1A + 4
y2B = y1B + 4

data_lake = pd.read_csv(lake_list,encoding='latin')
data_sjolista = pd.read_excel(sjolista)
data_sjolista.drop("X", axis=1, inplace=True)
data_sjolista.drop("Y", axis=1, inplace=True)
data_sjolista.drop("adjustment  factor", axis=1, inplace=True)
data_sjolista.drop("type of problem", axis=1, inplace=True)
data_sjolista.drop("Comment", axis=1, inplace=True)
data_sjolista.drop("endd", axis=1, inplace=True)
data_sjolista.drop("startd", axis=1, inplace=True)
data_sjo=data_sjolista.dropna()
lake_id = data_lake["ebhex"].tolist()
lake_id2 = data_sjo['ebhex'].tolist()

test = data_sjo.loc[data_sjo['ebhex'].isin(lake_id)]
test = test.sort_values(by=['ebhex'])
test2 =data_lake.loc[data_lake['ebhex'].isin(lake_id2)]
test2 = test2.sort_values(by=['ebhex'])


#test = test[test['end']>=1971]
#test = test[test['start']<1980]
print(test)  

list_selected_lake = test["ebhex"].tolist()
listmeanice = []
for eh in list_selected_lake:
        
    eh = eh[2:] if eh[:2] == '0x' else eh
    while len ( eh ) < 6:
        eh = '0' + eh
    d1, d2, d3 = eh[:2], eh[:4], eh[:6]
    outdir = os.path.join ( outputfolder, d1, d2, d3,
                            'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (
                                m1, exA, exB, m2, y1A, y2B),"calibration_result_old" )

    outdir
    try:
        ice = pd.read_csv(os.path.join ( outdir, 'His.csv' ),names=['1','2','3','4','5','6','7','8'])
        meanice = (ice['7'].sum())/10
        listmeanice.append(meanice)
        print(listmeanice)
    except:
        listmeanice.append("")
   
test['meanice'] = listmeanice
print(test2['area'])
test['area'] = test2['area'].tolist()
test['Mean'] = test2['Mean'].tolist()
test['volume'] = test2['volume'].tolist()

print(test)

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
test.to_csv(r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\icemodelvsdata1001old_%s.csv"%timestr)
print("eee")

list_selected_lake = lake_id
test = data_lake
listmeanice = []
for eh in list_selected_lake:

    eh = eh[2:] if eh[:2] == '0x' else eh
    while len(eh) < 6:
        eh = '0' + eh
    d1, d2, d3 = eh[:2], eh[:4], eh[:6]
    outdir = os.path.join(outputfolder, d1, d2, d3,
                          'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (
                              m1, exA, exB, m2, y1A, y2B), "calibration_result_old")

    try:
        ice = pd.read_csv(os.path.join(outdir, 'His.csv'), names=['1', '2', '3', '4', '5', '6', '7', '8'])
        meanice = (ice['7'].sum()) / 10
        listmeanice.append(meanice)
        print(listmeanice)
    except:
        listmeanice.append("")

test['meanice'] = listmeanice

print(test)


timestr = time.strftime("%Y%m%d-%H%M%S")
test.to_csv(r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\icemodelvsdata100111old_%s.csv" % timestr)
print("eee")