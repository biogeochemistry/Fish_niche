# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:48:45 2019

@author: macot620
"""


import pandas as pd
import os as path

with open(r'C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList.csv', 'rU')as f:

        lines = f.readlines()
        nlines = len(lines)
        ii = range(1, nlines)
    
for variable in ['T','PPFD']:
    writer = pd.ExcelWriter('Montly_Mean_%s_2001-2010.xlsx'%(variable),engine='xlsxwriter')
    for i in ii:
       lake_id, subid, name, ebh, area, depth, longitude, latitude, volume, mean_depth, sediment, mean_calculated\
       = lines[i].strip().split(',')
       print(lake_id)
       datafile = (r'C:\Users\macot620\Documents\Figure_Montly_Mean_PPFD_T_2001-2010/%s_Monthly_Mean_%s_2001-2010.csv'%(lake_id,variable))
       data = pd.read_csv(datafile, header=None)
       data.to_excel(writer,sheet_name=lake_id,index=False,header=False)
    writer.save()
        
            