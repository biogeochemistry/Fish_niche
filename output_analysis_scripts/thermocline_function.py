# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:51:15 2019

@author: Administrateur
"""

import rpy2.robjects as ro

def thermocline(temp,depths):
    r=ro.r
    r.source(r'C:\Users\Administrateur\Documents\GitHub\Fish_niche\output_analysis_scripts\montly_temperature_light_profiles.R' )
    p=r.calculThermocline(temp,depths)
    return p



if __name__ == '__main__':
    wtr = "22.51, 22.42, 22.4, 22.4, 22.4, 22.36, 22.3, 22.21, 22.11, 21.23, 16.42, 15.15, 14.24, 13.35, 10.94, 10.43, 10.36, 9.94, 9.45, 9.1, 8.91, 8.58, 8.43"

    #A vector defining the depths
    depths = "0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20"
    depths = '1,2,3,4,5,6,7'
    td = ((str(thermocline(wtr,depths)).strip().split(' '))[1].strip().split('.'))[0]
    test = depths.strip().split(',')
    print(test[4:])