## usage: python runlakesGoran_par.py modeli scenarioi csvfile
##   modeli:   modelid       (index into list in mylakeGoran.py)
##   scearioi: scenarioid    (index into list in mylakeGoran.py)
##   csvfile:  the name of the csv with the list of lakes to run.
##  example:
##		python runlakesGoran_par.py 2 2 2017SwedenList.c
import sys 
import math
from joblib import Parallel, delayed
import multiprocessing

#modeli = int(sys.argv[1])
#scenarioi = int(sys.argv[2])
#csvf = sys.argv[3]
modeli = 1
scenarioi = 1
csvf = "2017SwedenList.csv"

with open(csvf, 'rU') as f:
    #throwaway = f.readline()     #This was here to skip the heading, but somehow that is done automatically, and this line if left here skips the first lake. I have no idea why.
    lines = f.readlines()
    
    nlines = len(lines)
    
    ii = range(1, nlines)
    num_cores = multiprocessing.cpu_count()

def loop_through_lake_list(i):
    """
    loop which treat each lake in file with the function mylakeGoran.runlake().
    :param i: line in the file which give the information about the lake analysed
    :type i: int
    :return: function mylake.runlake()
    """
    lake_id, subid, name, ebh, area, depth, longitude, latitude \
        = lines[i].strip().split(',')
    print('running lake %s' % ebh)
    mylakeGoran.runlake(modeli, scenarioi, ebh.strip('"'), int(subid), float(depth),
                        float(area), float(longitude), float(latitude))


if __name__ == '__main__':
 

    
    Parallel(n_jobs=num_cores)(delayed(loop_through_lake_list)(i) for i in ii)  
