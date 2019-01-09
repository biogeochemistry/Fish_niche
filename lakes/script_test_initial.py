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
#cordexfolder = 'C:/Users/User/Documents/GitHub/Fish_niche/cordex' # neede to be change depending where de climatic files are.
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


if __name__ == '__main__':

    lakelistfile = r'2017SwedenList_test_initial.csv'
    k_BOD = 0.01
    swa_b1 = 1
    k_SOD = 100
    I_scDOC = 1
    model,scenario = 2,2
    runlakesGoran_par(lakelistfile, model, scenario,k_BOD,swa_b1,k_SOD,I_scDOC)
