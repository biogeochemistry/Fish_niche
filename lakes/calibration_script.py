from FishNiche_output_analysis_scripts2 import FishNiche_csv_result,FishNiche_graph_temp_complete_time,FishNiche_secchi_graph,FishNiche_graph_temp_time,FishNiche_mean_secchi_graph,FishNiche_validate_results,FishNiche_mean_k_BOD_graph, FishNiche_graph_oxy_time
from run_parallel_mylake import runlakesGoran_par
import pandas as pd

# initial parameter I_scT :0, I_scDOC: 1, I_scO:1,
    # k_BOD:0.01,k_SOD:100,theta_BOD:1.047,theta_SOD:1

if __name__ == '__main__':

    lakelistfile = r'2017SwedenList_only_validation_lakes3.csv'
    lakelistfile2 = r'2017SwedenList_only_validation_lakes2.csv'

    outputfolder = '../output'

    ki = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 950, 1000]
    jj = [1e-8, 5e-8, 7.5e-8, 1e-7, 2.5e-7, 5e-7, 7.5e-6, 1e-6, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002,
          0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
    ki=[50]
    jj=[0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
    #FishNiche_csv_result ( 2, 2, lakelistfile2, 10, 1e-8, 0, 0, ki, jj )


    # ki = [550]
    # jj = [ 0.01,  0.05]
    # for i in ['Equation2_I_scDOC']:
    #     m = 0
    #     print ( i )
    #     for k in ki:
    #         print ( k )
    #         n = 0
    #
    #         for j in jj:
    #             print ( j )
    #             runlakesGoran_par ( lakelistfile, 2, 2, j, k )
    #             FishNiche_csv_result ( 2, 2, lakelistfile2, k, j, m, n, ki, jj ,'pearson5','rmse5' )
    #
    #             n += 1
    #         m += 1




    ki = [10,20]
    jj = [ 7.5e-6,1e-6,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.005,0.006,0.007,0.01,0.05]
    for i in ['Equation_']:
        m = 1
        print ( i )
        for k in ki:
            print ( k )
            n = 0

            for j in jj:
                print ( j )
                runlakesGoran_par ( lakelistfile, 2, 2, j, k )
                FishNiche_csv_result ( 2, 2, lakelistfile2, k, j, m, n, ki, jj, 'pearson1', 'rmse1' )
                FishNiche_validate_results (2, 2, lakelistfile2, "Bob_%s_Sod_%s" % (j,k), 1000000*j*k )
                n += 1
            m += 1



