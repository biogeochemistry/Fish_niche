from FishNiche_output_analysis_scripts2 import FishNiche_csv_result,FishNiche_plot_timeseries,FishNiche_graph_temp_complete_time,FishNiche_secchi_graph,FishNiche_graph_temp_time,FishNiche_mean_secchi_graph,FishNiche_validate_results,FishNiche_mean_k_BOD_graph, FishNiche_graph_oxy_time
from run_parallel_mylake import runlakesGoran_par

#import matplotlib.pyplot as plt
import sys
#sys.path.insert(0, 'G:')
#from unzip_functions import unzip,delete_h5
#from joblib import Parallel, delayed
import multiprocessing
#import os
import datetime

#modeli = int ( sys.argv[1] )
#scenarioi = int ( sys.argv[2] )
#csvf = sys.argv[3]
cordexfolder = 'G:cordex'
outputfolder = '../output'
num_cores = multiprocessing.cpu_count ()
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
# initial parameter I_scT :0, I_scDOC: 1, I_scO:1,
    # k_BOD:0.01,k_SOD:100,theta_BOD:1.047,theta_SOD:1

#Equation finale: sew = 7, sod= equation 3, bod= equation 2, iscdoc= equation 2

if __name__ == '__main__':

    lakelistfile = r'2017SwedenList_only_validation_lakes3.csv'
    lakelistfile2 = r'2017SwedenList_only_validation_lakes2.csv'
    for model in [2]:
        for scenario in [2]:
            m0, m1 = models[model]
            s0, s1 = scenarios[scenario]
            test=True
            #with open ( '%s/running_report.txt'%outputfolder, 'w' ) as f:
            #    f.write('\nrunning _EUR-11_%s_%s_%s_%s-%s\n' % (m0, s0[0], m1, s0[1], s1[2]))
            #    f.close()
            # unzip  h5 files:
            #try:
            #    Parallel ( n_jobs=num_cores) (
            #        delayed ( unzip ) ( v, model, scenario, cordexfolder,'%s/running_report.txt'%outputfolder ) for v in variables )
            #    with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
            #        f.write ( 'able to unzip h5 files\n' )
            #        f.close()
            #except:
            #    test=False
            #    with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
            #        f.write('unable to find or unzip _EUR-11_%s_%s_%s_%s-%s.h5.gz\n' % (m0, s0[0], m1, s0[1], s1[2]) )
            #        f.close()
            #    print ( 'unable to find or unzip _EUR-11_%s_%s_%s_%s-%s.h5.gz' % (m0, s0[0], m1, s0[1], s1[2]) )

            # run mylake
            #try:
            #    with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
            #        f.write ( 'running mylake \n' )
            #        f.close()
            # swa=0.42
            #
            # #runlakesGoran_par ( lakelistfile, model, scenario, swa )
            #
            # FishNiche_validate_results ( scenario, model, lakelistfile2, "2001_2010_swa_b1_%s" % (swa), swa*10 )
            # FishNiche_secchi_graph ( scenario, model, lakelistfile2, "2001_2010_swa_b1_%s" % (swa), swa*10 )

            for swa in ['equation1_basek']:
                with open ( '%s/running_report.txt' % outputfolder, 'w' ) as f:
                    f.write ( '/n start_swa_%s\n' % swa )
                    f.close ()
                runlakesGoran_par(lakelistfile, model, scenario,swa)
                with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
                    f.write ( datetime.datetime.now ().isoformat () )
                FishNiche_validate_results ( scenario, model, lakelistfile2, "2001_2010_swa_b1_%s"%(swa),100 )
                FishNiche_secchi_graph ( scenario, model, lakelistfile2, "2001_2010_swa_b1_%s"%(swa), 100 )
            #    with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
            #        f.write ( 'run of mylake completed\n' )
            #        f.close()
            #except:
            #    test=False
            #    with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
            #        f.write('unable to run mylake for _EUR-11_%s_%s_%s_%s-%s\n' % (m0, s0[0], m1, s0[1], s1[2]))
            #        f.close()
            #    print ( 'unable to run mylake for _EUR-11_%s_%s_%s_%s-%s' % (m0, s0[0], m1, s0[1], s1[2]) )

            #delete h5 file unzip
            #try:

            #    Parallel ( n_jobs=num_cores ) (
            #        delayed ( delete_h5 ) ( v, model, scenario, cordexfolder,'%s/running_report.txt'%outputfolder ) for v in variables )
            #except:
            #    test=False

            #if test:
            #    with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
            #        f.write('\nrun of _EUR-11_%s_%s_%s_%s-%s completed without Error \n' % (m0, s0[0], m1, s0[1], s1[2]))
            #        f.close()
    #with open ( '%s/running_report.txt' % outputfolder, 'a' ) as f:
    #    f.write('END')
    #    f.close()
    #print('end')