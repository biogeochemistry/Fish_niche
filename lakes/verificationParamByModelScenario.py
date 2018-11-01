"""
Script verifies if all the h5 files needed of Mylake to run are present in the same directory.
With the giving directory (directory_where_h5_are), it look at each combination of model-scenario and verify if the 14 h5 files corresponding exist in it.
It return two csv file with index = model and columns = scenario: 'model_scenario_variables_list.csv'  if all files exist, data= OK, and if not, data= the number of missing files
                                                                  'model_scenario_variables_missing.csv' if all files exist, data = OK, and if not, data = list of missing variables (2 first letters + 1 or 2 depending if it is first or second part of scenario (XXX1-XXX5 or XXX6-XXX0))
"""

import os.path
import pandas as pd


directory_where_h5_are = r"G:\cordex"

scenarios = {1: (('historical', '19710101', '19751231'), ('historical', '19760101', '19801231')),
             2: (('historical', '20010101', '20051231'), ('rcp45', '20060101', '20101231')),
             3: (('rcp45', '20310101', '20351231'), ('rcp45', '20360101', '20401231')),
             4: (('rcp45', '20610101', '20651231'), ('rcp45', '20660101', '20701231')),
             5: (('rcp45', '20910101', '20951231'), ('rcp45', '20960101', '21001231')),
             6: (('rcp85', '20310101', '20351231'), ('rcp85', '20360101', '20401231')),
             7: (('rcp85', '20610101', '20651231'), ('rcp85', '20660101', '20701231')),
             8: (('rcp85', '20910101', '20951231'), ('rcp85', '20960101', '21001231'))}

models = {1: ('ICHEC-EC-EARTH', 'r1i1p1_KNMI-RACMO22E_v1_day'),  # MC 2018-05-16
          2: ('ICHEC-EC-EARTH', 'r3i1p1_DMI-HIRHAM5_v1_day'),
          3: ('MPI-M-MPI-ESM-LR', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day'),
          4: ('MOHC-HadGEM2-ES', 'r1i1p1_SMHI-RCA4_v1_day'),
          5: ('IPSL-IPSL-CM5A-MR', 'r1i1p1_IPSL-INERIS-WRF331F_v1_day'),
          6: ('CNRM-CERFACS-CNRM-CM5', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day')}

variables = ['clt', 'hurs', 'tas', 'rsds', 'ps', 'pr', 'sfcWind']

url2 = '_EUR-11_'
final = pd.DataFrame(index=[1,2,3,4,5,6],columns=[1,2,3,4,5,6,7,8])
final1 = pd.DataFrame(index=[1,2,3,4,5,6],columns=[1,2,3,4,5,6,7,8])
for model in [6]:
    for scenario in [6,7,8]:
        listvariable = ""
        j=0
        for v in variables:
            m0, m1 = models[model]
            s0, s1 = scenarios[scenario]
            urlsA = [
               '/Lakes_%s%s%s_%s_%s_%s-%s.h5' %  # MC 2017-05-16 add of "Lakes_" to be fix h5 filename on repertory, erase _day
               (v, url2, m0, s0[0], m1, s0[1], s0[2])]

            urlsB = ['/Lakes_%s%s%s_%s_%s_%s-%s.h5' %  # MC 2017-05-16 add of "Lakes_" to be fix h5 filename on repertory
                     (v, url2, m0, s1[0], m1, s1[1], s1[2]) ]
            urls = urlsA + urlsB

            ##for models 4, dates end by 30 and not 31 MC 2018-05-18
            new_urls = []
            for url in urls:
                if url.find ( "MOHC-HadGEM2-ES" ) != -1:
                    if url.find ( "20960101" ) != -1:  # 2018-07-31 MC for model 4, scenario 5, interval end = 2099-11-30, not 2100-12-31 and scenario 8 = 2099-12-30
                        if url.find ( "rcp45" ) != -1:
                           head, _sep, tail = url.rpartition ( '21001231' )
                           url = head + "20991130" + tail
                        elif url.find ( "rcp85" ) != -1:
                            head, _sep, tail = url.rpartition ( '21001231' )
                            url = head + "20991230" + tail
                    else:
                        head, _sep, tail = url.rpartition ( '1' )
                        url = head + "0" + tail
                new_urls.append ( url )
            urls = new_urls

            for file in urls:
                print(file)
                file_out = "%s%s" % (directory_where_h5_are, file)
                if not os.path.isfile ( file_out ):
                    listvariable= listvariable + "%s"%v[0:2]
                    test = file[-17]
                    if file[-17]=='1':
                        listvariable = listvariable+'1, '
                    elif file[-17] == '6':
                        listvariable = listvariable+'2, '
                    j+=1
        if listvariable == "":
            final.loc[model, scenario] ="OK"
            final1.loc[model, scenario] = "OK"
        else:
            final1.loc[model,scenario]=listvariable
            final.loc[model, scenario] = j
final.to_csv('model_scenario_variables_list.csv')
final1.to_csv('model_scenario_variable_missing.csv')

