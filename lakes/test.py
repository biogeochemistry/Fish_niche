
from os import path
import pandas as pd
filename = r'D:\Fish_niche\lakes\2017SwedenList.csv'
datafolder = r'E:\output-30-03-2019'
models = {1: ('ICHEC-EC-EARTH', 'r1i1p1_KNMI-RACMO22E_v1_day'),
          2: ('ICHEC-EC-EARTH', 'r3i1p1_DMI-HIRHAM5_v1_day'),
          3: ('MPI-M-MPI-ESM-LR', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day'),
          4: ('MOHC-HadGEM2-ES', 'r1i1p1_SMHI-RCA4_v1_day'),
          5: ('IPSL-IPSL-CM5A-MR', 'r1i1p1_IPSL-INERIS-WRF331F_v1_day'),
          6: ('CNRM-CERFACS-CNRM-CM5', 'r1i1p1_CLMcom-CCLM4-8-17_v1_day')}
scenarios = {1: ('historical', 1971, 'historical', 1976),
             2: ('historical', 2001, 'rcp45', 2006),
             3: ('rcp45', 2031, 'rcp45', 2036),
             4: ('rcp45', 2061, 'rcp45', 2066),
             5: ('rcp45', 2091, 'rcp45', 2096),
             6: ('rcp85', 2031, 'rcp85', 2036),
             7: ('rcp85', 2061, 'rcp85', 2066),
             8: ('rcp85', 2091, 'rcp85', 2096)}
with open(filename, 'rU')as f:
    lakes = f.readlines()
    nlakes = len(lakes)
j = 0
dataframe = []
for lakenum in range(1, nlakes):
    lake_id, subid, name, eh, area, depth, longitude, latitude, volume, meandepth, sedimentArea, meanCalculated \
        = lakes[lakenum].strip().split(',')

    eh = eh[2:] if eh[:2] == '0x' else eh
    while len(eh) < 6:
        eh = '0' + eh

    d1, d2, d3 = eh[:2], eh[:4], eh[:6]
    if j == 0:
        timeseries = pd.DataFrame()
        listtitle = ['lake']
        
       
    listlake = [lake_id]
        
    for model in [1,2,3,4,5,6]:
        for scenario in [1,2,3,4,5,6,7,8]:
            if j==0:
                listtitle.append('M%sS%s'%(model,scenario))
            exA, y1A, exB, y1B = scenarios[scenario]
            y2B = y1B + 4
            m1, m2 = models[model]
            outdir = path.join(datafolder, d1, d2, d3,
                                       'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (m1, exA, exB, m2, y1A, y2B),'Tzt.csv')
            outdir2 = path.join(datafolder, d1, d2, d3,
                                       'EUR-11_%s_%s-%s_%s_%s0101-%s1231' % (m1, exA, exB, m2, y1A, y2B),'2017init')
            
            if not path.exists(outdir):
                if not path.exists(outdir2):
                    listlake.append(0)
                else:
                    listlake.append(2)
            else:
                with open(outdir,'r') as file:
                    lfile=file.readline()
                    
                if not lfile.find('NA')==-1: #no data
                    if not path.exists(outdir2):
                        listlake.append(1)  
                    else:
                        listlake.append(3)
                else:
                    listlake.append(4)  
    print(lake_id)
    if j==0:
        j+=1
    else:
        dataframe.append(listlake) 
        j+=1
    
timeseries = pd.DataFrame(dataframe, columns = listtitle)
print('end')
timeseries.to_csv('out.csv')

