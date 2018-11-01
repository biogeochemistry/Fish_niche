
import dbf
import csv

table = dbf.Table('ecco-biwa_lakes_v.0.2.dbf')
table.open()
fdi = open('2017SwedenList_incomplete_data.csv', 'r')
fdo = open('2017SwedenList.csv', 'w')
try:
    line0 = fdi.readline()
    lines = fdi.readlines()
    lakeids = [int(line.strip().split(',')[0]) for line in lines]
    subids = [int(line.strip().split(',')[1]) for line in lines]
    names = [line.strip().split(',')[2] for line in lines]
    areas = [float(line.strip().split(',')[3]) for line in lines]
    depths = [float(line.strip().split(',')[4]) for line in lines]
    longitudes = [float(line.strip().split(',')[5]) for line in lines]
    latitudes = [float(line.strip().split(',')[6]) for line in lines]

    csvout = csv.writer(fdo, dialect='dbf')
    csvout.writerow(['lake_id', 'subid', 'name', 'ebhex' , 'area', 'depth', 'longitude', 'latitude'])
    for record in table:
        for ii in range(0,len(lakeids)):
            if record['lake_id'] == lakeids[ii]:
                fields = [lakeids[ii], subids[ii], names[ii], record['ebhex'].strip(), areas[ii], depths[ii], longitudes[ii], latitudes[ii]]
                csvout.writerow(fields)
                break
    
finally:
    table.close()
    fdo.close()
    fdi.close()