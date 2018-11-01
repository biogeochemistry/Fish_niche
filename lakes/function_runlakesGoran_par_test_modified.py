## usage: python runlakesGoran_par.py modeli scenarioi csvfile
##   modeli:   modelid       (index into list in mylakeGoran.py)
##   scearioi: scenarioid    (index into list in mylakeGoran.py)
##   csvfile:  the name of the csv with the list of lakes to run.
##  example:
##		python runlakesGoran_par.py 2 2 2017SwedenList.csv


import mylakeGoran
import sys
import math
import numpy as np
from joblib import Parallel, delayed
import multiprocessing


#modeli = int ( sys.argv[1] )
#scenarioi = int ( sys.argv[2] )
#csvf = sys.argv[3]
def runlakesGoran_pa(csvf,modeli,scenarioi,k_BOD,k_SOD,I_scDOC,swa_b1):
    with open ( csvf, 'rU' ) as f:
        # throwaway = f.readline()     #This was here to skip the heading, but somehow that is done automatically, and this line if left here skips the first lake. I have no idea why.
        lines = f.readlines ()

        nlines = len ( lines )

        ii = range ( 1, nlines )
    num_cores = multiprocessing.cpu_count ()-4 #modified to be able to run other programs
    lake = [1178685, 4935398, 4834100, 3858150, 4032446, 52161748, 34085962, 456332771, 1885977434]
    swa_b1series=[]
    for i in lake:
        swa_b1series.append(162279*math.log(i)**-4.648)
    arealake = [1057659999725.86,6.1376E+12,1.0577E+12,2.5725E+13,2.5253E+13,2.5103E+13,4.2755E+15,2.3542E+15,6.6229E+17,1.9464E+19]
    k_SODseries =[]
    for i in lake:
        k_SODseries.append(716441720496.523*np.exp(-1.54*math.log(i))) #4.5586E+39*math.log(i)**-25.47 10291000000*np.exp(-0.64*i)
    k_BODseries = []
    #for i in arealake[0:4]:
    #    k_BODseries.append ( 5.0528593E-33*math.log ( i )**21.2887635)#26983 * np.exp ( -0.435 * math.log ( i ) ) )
    for i in lake:#[5:]:
        k_BODseries.append(191718*np.exp(-0.985*math.log(i)))#26983*np.exp(-0.435*math.log(i)))#0.0016695*math.log ( i )**2 - 0.14410753*math.log ( i ) + 3.10722634)#307967*np.exp(-1.02*math.log(i)))#

    print(k_SODseries)
    print(k_BODseries)
    Parallel ( n_jobs=num_cores ) ( delayed ( loop_through_lake_list ) ( i,lines,modeli,scenarioi,k_BODseries[i-1],swa_b1series[i-1],k_SODseries[i-1],I_scDOC ) for i in ii )#k_SODseries[i-1]


def loop_through_lake_list(i,lines,modeli,scenarioi,k_BOD=0.01,swa_b1=1,k_SOD=100,I_scDOC=1):
    """
    loop which treat each lake in file with the function mylakeGoran.runlake().
    :param i: line in the file which give the information about the lake analysed
    :type i: int
    :return: function mylake.runlake()
    """
    # 5-9-2018 MC
    lake_id, subid, name, ebh, area, depth, longitude, latitude \
        = lines[i].strip ().split ( ',' )
    print ( 'running lake %s' % ebh )
    print(k_SOD)
    print(k_BOD)
    mylakeGoran.runlake (modeli, scenarioi, ebh.strip ( '"' ), int ( subid ), float ( depth ),float ( area ), float ( longitude ), float ( latitude ),k_BOD,swa_b1,k_SOD,I_scDOC )#add parameters


if __name__ == '__main__':
    #modeli = int ( sys.argv[1] )
    a=0
    #scenarioi = int ( sys.argv[2] )
    #csvf = sys.argv[3]
    #with open ( csvf, 'rU' ) as f:
        # throwaway = f.readline()     #This was here to skip the heading, but somehow that is done automatically, and this line if left here skips the first lake. I have no idea why.
     #   lines = f.readlines ()

      #  nlines = len ( lines )

       # ii = range ( 1, nlines )
    #num_cores = multiprocessing.cpu_count ()
    #Parallel ( n_jobs=num_cores ) ( delayed ( loop_through_lake_list ) ( i,lines,modeli,scenarioi ) for i in ii )

    #runlakesGoran_pa ( 'test.csv', 4, 2 )



