#!/usr/bin/env python3
"""
    File name: run_parallel_mylake.py
    Author: Mariane Cote
    Date created: 08/28/2018
    Python Version: 3.6

    Launch of the large modelisation.

    usage: python runlakesGoran_par.py modeli scenarioi csvfile
        modeli:      modelid     (index into list in mylakeGoran.py)
        scearioi:    scenarioid  (index into list in mylakeGoran.py)
        csvfile:     filename of the csv with the list of lakes to run.
    example:
        python runlakesGoran_par.py 2 2 2017SwedenList.csv
        Or using Python IDE.
"""

import run_mylakeGoran
import math
from joblib import Parallel, delayed
import multiprocessing
import datetime
import time
import os
num_cores = multiprocessing.cpu_count()  # needs to be modified if you want to choose the number of cores used.

#cordexfolder = 'G:cordex'  # needs to be changed depending where the meteorological files are.
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
# The website for the meteorological variables generated by the combination of model-scenario:
# 'http://ns2806k.web.sigma2.no/'  # website updated: 5-9-2018 MC

# initial parameter:
#       I_scT :0, I_scDOC: 1, I_scO:1, k_BOD:0.01, k_SOD:100, theta_BOD:1.047, theta_SOD:1
#
# k_BOD: Biochemical oxygen demand coefficient
# swa_b1: PAR light atteneuation coefficient
# k_SOD: Sedimentary oxygen demand
# I_scDOC: scaling factor for inflow concentration of DOC


def loop_through_model_scenario(listofmodels, listofscenarios, lakelistfile, report=None):
    """loops through all possible combination of model and scenario.

    Tries with each combination of model and scenario to run the function runlakesGoran_par with the given lakes list.
    Creates a report of the run.

    Args:
        listofmodels: list of numbers (the keys of the dictionary "models")
        listofscenarios: list of numbers (the keys of the dictionary "scenarios")
        lakelistfile: filename of the CSV file containing a list of lakes
        report: Optional variable. If a report is wanted, it is the filename of the TXT file.
    """
    nbrsuccess = 0
    nbrtested = 0
    report = os.path.join(outputfolder, report)
    st = datetime.datetime.fromtimestamp(time.time()).strftime(timeformat)
    with open(lakelistfile, 'rU') as f:
        lines = f.readlines()
        nlines = len(lines)

    for model in listofmodels:
        for scenario in listofscenarios:
            m0, m1 = models[model]
            s0, s1 = scenarios[scenario]
            
            if report is not None:
                if not os.path.isfile(report): 
                    with open(report, 'w') as f:
                        f.write('\n-------Start-------\n%s\n'%(st))
                        f.close()
                else:
                    if model ==1 and scenario == 1:
                        with open(report, 'a') as f:
                            f.write('\n-------Start-------\n%s\n'%(st))
                            f.close()
                try:
                    
                    with open(report, 'a') as f:
                        times = datetime.datetime.fromtimestamp(time.time()).strftime(timeformat)
                        f.write('\nrunning _EUR-11_%s_%s_%s_%s-%s\n%s\n' % (m0, s0[0], m1, s0[1], s1[2],times))
                        f.close()
                    nbrtested += nlines
                    runlakesGoran_par(model, scenario, lakelistfile)
                    nbrsuccess += nlines
                    with open(report, 'a') as f:
                        f.write('run of mylake completed for _EUR-11_%s_%s_%s_%s-%s\n' % (m0, s0[0], m1, s0[1], s1[2]))
                        f.close()
                except:
                    with open(report, 'a') as f:
                        f.write('\nunable to run mylake for _EUR-11_%s_%s_%s_%s-%s\n' % (m0, s0[0], m1, s0[1], s1[2]))
                        f.close()

            else:
                try:
                    runlakesGoran_par(model, scenario, lakelistfile)
                    print('run of mylake completed for _EUR-11_%s_%s_%s_%s-%s' % (m0, s0[0], m1, s0[1], s1[2]))
                except:
                    print('unable to run mylake for _EUR-11_%s_%s_%s_%s-%s' % (m0, s0[0], m1, s0[1], s1[2]))
    if report is not None:
        end = datetime.datetime.fromtimestamp(time.time()).strftime(timeformat)
        interval = datetime.datetime.strptime(end, timeformat) - datetime.datetime.strptime(st, timeformat)
        with open(report, 'a') as f:
            f.write('\n-------End-------\n')
            f.write('\n-------Run Summary-------\n')
            f.write('\nStart : %s      End : %s       Run Time : %s '
                    '\nNumber of combination (lake,model,scenario) tested: %s '
                    '\nNumber of lake tested: %s '
                    '\nSuccessful combination : %s '
                    '\nUnsuccessful combination : %s\n'
                    % (st, end, interval, nbrtested, nlines, nbrsuccess, nbrtested-nbrsuccess))
            f.write('\n------------------------\n')
            f.close()


def runlakesGoran_par(model_id, scenario_id, csvf):
    """Runs in parallel loop_through_lake_list()

    For each lake in the lake list, it runs in parallel the function loop_through_lake_iist().

    Args:
        csvf: filename of the CSV file containing a list of lakes.
        model_id: model id (one of the keys of the dictionary "models")
        scenario_id: scenario id (one of the keys of the dictionary "scenarios")
    """
    with open(csvf, 'rU') as f:
        lines = f.readlines()
        nlines = len(lines)
        ii = range(1, nlines)

    #for i in ii:
    #    loop_through_lakes_list(i,lines,model_id,scenario_id)
    Parallel(n_jobs=num_cores)(delayed(loop_through_lakes_list)(i, lines, model_id, scenario_id) for i in ii)


def loop_through_lakes_list(i, lines, modelid, scenarioid):
    """Changes value of some parameters and run run_mylakeGoran.runlake()

    Args:
        i: index of the list of lskes
        lines: list of lakes
        modelid: model id (one of the keys of the dictionary "models")
        scenarioid: scenario id (one of the keys of the dictionary "scenarios")
    """

    lake_id, subid, name, ebh, area, depth, longitude, latitude, volume, mean_depth, sediment \
        = lines[i].strip().split(',')
    mean_calculated = mean_depth
    print('running lake %s' % ebh)
    if mean_depth != '' and (float(depth) - float(mean_depth)) > 0:
        swa_b1 = math.exp(-0.95670 * math.log(float(mean_depth)) + 1.36359)
        # swa_b1 = OLD EQUATION: 3.6478*float(depthmean)**-0.945
        #                        0.796405*math.exp(-0.016045*float(depth))
        #                       -0.3284*math.log(float(depth)) + 1.6134
        #                       -0.8547*math.log(math.log(depth)) + 1.4708
        #                       3.0774*math.exp(-0.6432*math.log(float(depth)))

        k_BOD = math.exp(-0.25290 * float(mean_depth) - 1.36966)
        # k_BOD = OLD EQUATION: 0.268454*math.log(float(depth))**-3.980304
        #                     0.291694*math.log(float(depth))**-4.013598#0.2012186 * math.log(float(depth)) ** -3.664774
    else:
        swa_b1 = math.exp(-0.95670 * math.log(float(mean_depth)) + 1.36359)
        k_BOD = math.exp(-0.25290 * float(mean_calculated) - 1.36966)

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

    # Script to run test the model with the command line MC-2018-11-02
    # modeli = int(sys.argv[1])
    # scenarioi = int(sys.argv[2])
    # csvf = sys.argv[3]
    # runlakesGoran_par(csvf, modeli, scenarioi)

    # Line used to run regional scale simulation with all model-scenario combinations
    loop_through_model_scenario ( [ 2], [ 2], r'2017SwedenList_only_validation_12lakes.csv', 'report_12.txt' )
    #loop_through_model_scenario ( [1,2,3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8], r'2017SwedenList.csv', 'report_end.txt' )