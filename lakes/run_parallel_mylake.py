"""
run_parallel_mylake.py
last modification 2018-08-29 MC

Launch of the large modelisation.

usage: python runlakesGoran_par.py modeli scenarioi csvfile
   modeli:   modelid       (index into list in mylakeGoran.py)
   scearioi: scenarioid    (index into list in mylakeGoran.py)
   csvfile:  the name of the csv with the list of lakes to run.
   example:
		python runlakesGoran_par.py 2 2 2017SwedenList.csv
   Or using Python IDE.
"""

import run_mylakeGoran
import sys
import math
from joblib import Parallel, delayed
import multiprocessing

#listofmodels = int ( sys.argv[1] )
#listofscenarios = int ( sys.argv[2] )
#csvf = sys.argv[3]

num_cores = multiprocessing.cpu_count () #to modified if you want to choose the number of cores used.

cordexfolder = 'G:cordex' #change depending where the climatic files are.
outputfolder = '../output'
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
url1 = 'http://ns2806k.web.sigma2.no/'  # 5-9-2018 MC
url2 = '_EUR-11_'
# initial parameter I_scT :0, I_scDOC: 1, I_scO:1,
    # k_BOD:0.01,k_SOD:100,theta_BOD:1.047,theta_SOD:1

def loop_throught_model_scenario(listofmodels,listofscenarios,csvf):
    """
        Launch the modelisation with each model and scenario for the list of lake given.
        Write a report of the run.
        :param listofmodels: list of the models by number
        :param listofscenarios: list of the scenarios by number
        :param csvf: list of the lakes

        :return: None
    """
    lakelistfile = csvf
    for model in listofmodels:
        for scenario in listofscenarios:
            m0, m1 = models[model]
            s0, s1 = scenarios[scenario]
            test = True
            with open ( report, 'w' ) as f:
               f.write('\nrunning _EUR-11_%s_%s_%s_%s-%s\n' % (m0, s0[0], m1, s0[1], s1[2]))
               f.close()

            try:
                with open ( report, 'a' ) as f:
                    f.write ( 'running mylake \n' )
                    f.close ()
                runlakesGoran_par ( lakelistfile, model, scenario )
                with open (report, 'a' ) as f:
                    f.write ( 'run of mylake completed\n' )
                    f.close ()

            except:
                with open ( report, 'a' ) as f:
                    f.write ( 'unable to run mylake for _EUR-11_%s_%s_%s_%s-%s\n' % (m0, s0[0], m1, s0[1], s1[2]) )
                    f.close ()

                print ( 'unable to run mylake for _EUR-11_%s_%s_%s_%s-%s' % (m0, s0[0], m1, s0[1], s1[2]) )

def runlakesGoran_par(csvf,modeli,scenarioi,k_BOD=0.01,swa_b1=1,k_SOD=100,I_scDOC=1):
    """
        Launch the modelisation with each model and scenario for each lake.
        :param modeli: model by number
        :param scenarioi: scenario by number
        :param csvf: list of the lakes
        :return: None
    """
    with open ( csvf, 'rU' ) as f:
        # throwaway = f.readline()     #This was here to skip the heading, but somehow that is done automatically, and this line if left here skips the first lake. I have no idea why.
        lines = f.readlines ()
        nlines = len ( lines )
        #nlines = 3
        ii = range ( 1, nlines )


    #for i in ii:
    #    loop_through_lake_list(i,lines,modeli,scenarioi)
    Parallel ( n_jobs=num_cores ) ( delayed ( loop_through_lake_list ) ( i,lines,modeli,scenarioi,k_BOD,swa_b1,k_SOD,I_scDOC) for i in ii)#series[i-1]

def loop_through_lake_list(i,lines,modeli,scenarioi,k_BOD=0.01,swa_b1=1,k_SOD=100,I_scDOC=1):
    """
    loop which treat each lake in file with the function mylakeGoran.runlake().
    :param i: line in the file which give the information about the lake analysed
    :param lines: list of lake and description
    :param modeli: model by number
    :param scenarioiL scenario by number
    :return: None
    """
    # 5-9-2018 MC
    lake_id, subid, name, ebh, area, depth, longitude, latitude, volume \
        = lines[i].strip ().split ( ',' )
    print ( 'running lake %s' % ebh )
    #swa_b1 = OLD EQUATION: 3.6478*float(depthmean)**-0.945 # 0.796405*math.exp(-0.016045*float(depth))#-0.3284*math.log(float(depth)) + 1.6134#-0.8547*math.log(math.log(depth)) + 1.4708#3.0774*math.exp(-0.6432*math.log(float(depth)))
    #k_SOD = OLD EQUATION: 13069.873528*math.exp(-1.760776*math.log(float(depth)))#11838.3*math.exp(-1.69321*math.log(float(depth)))
    print('swa %s, SOD %s'%(swa_b1calib,k_sod))
    #k_BOD = OLD EQUATION: 0.268454*math.log(float(depth))**-3.980304#0.291694*math.log(float(depth))**-4.013598#0.2012186 * math.log(float(depth)) ** -3.664774
    print ('BOD %s'% k_bod)
    #I_scDOC2 = OLD EQUATION: math.log((swa_b1calib+0.727)/0.3208008880)/(2*0.1338538345) #to modified if swa_b0 is modified
    print('IDOC %s'%I_scDOC2)

    run_mylakeGoran.runlake ( modeli, scenarioi, ebh.strip ( '"' ), int ( subid ), float ( depth ),
                    float ( area ), float ( longitude ), float ( latitude ),k_BOD,swa_b1,k_SOD,I_scDOC )


if __name__ == '__main__':
    listofscenarios = [1,2,3,4,5,6,7,8]
    listofmodels = [1,2,3,4,5,6]
    csvf = r'2017SwedenList_only_validation_lakes.csv'
    loop_throught_model_scenario ( listofmodels, listofscenarios, csvf )