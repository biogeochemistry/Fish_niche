# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:51:15 2019

@author: Administrateur
"""

import rpy2.robjects as ro
from statistics import mean

def thermocline(temp,depths):
    r=ro.r
    r.source(r'D:\Fish_niche\output_analysis_scripts\montly_temperature_light_profiles.R' )
    p=r.calculThermocline(temp,depths)
    return p



if __name__ == '__main__':
    wtr = "22.51, 22.42, 22.4, 22.4, 22.4, 22.36, 22.3, 22.21, 22.11, 21.23, 16.42, 15.15, 14.24, 13.35, 10.94, 10.43, 10.36, 9.94, 9.45, 9.1, 8.91, 8.58, 8.43"

    #A vector defining the depths
    #depths = "0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20"
    depths = '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18'
    wts = '0,2.857,3.467,3.721,4.068,4.068,4.068,4.066,4.062,4.059,4.056,4.054,4.052,4.052,4.051,4.051,4.051,4.051'
   
    td = ((str(thermocline(wts,depths)).strip().split(' '))[1].strip().split('.'))
    #test = td.strip().split(',')

    
    print(td)
    wts1 = wts.split(',')
    print(wts1)
    results = [float(i) for i in wts1] 
    td1 = int(td[0])
    print(td1)
    print(mean(results[td1:]))