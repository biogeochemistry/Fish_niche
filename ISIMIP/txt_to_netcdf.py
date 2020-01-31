from netCDF4 import Dataset
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import run_myLake_ISIMIP
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num, date2index
import os
output_long={"strat":"Thermal_stratification", "watertemp":"Water_temperature", "thermodepth":"Depth_of_Thermocline", "ice":"Lake_ice_cover",
             "lakeicefrac":"Lake_layer_ice_mass_fraction","snowtick":"Snow_thickness", "sensheatf": "Sensible_heat_flux_at_the_lake-atmosphere_interface",
             "latentheatf":"Latent_heat_flux_at_the_lake-atmosphere_interface", "lakeheatf": "Downward_heat_flux_at_the_lake-atmosphere_interface",
             "albedo":"Surface_albedo",
             "turbdiffheat":"Turbulent_diffusivity_of_heat", "sedheatf":"Sediment_upward_heat_flux_at_the_lakesediment_interface"}
# datalake = pd.read_csv(r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList1.csv",encoding = "ISO-8859-1")
# dataalllake = pd.read_csv(r"T:\Usagers\macot620\Lake_IDs_2019\lake_identifier_registry.csv",encoding = "ISO-8859-1")
#
# dataid = dataalllake[dataalllake['ebhex'].isin(datalake['ebhex'])][['ebint','lake_id','ebhex']]
# dataid = dataid.drop_duplicates().set_index('ebhex')
# datalake = datalake.set_index('ebhex')
# datalake['ebint'] = dataid['ebint']
# datalake['volume']= datalake['volume'].astype(float)
# datalake['area']=datalake['area'].astype(float)
#
# datalake['secchi'] = 1.48/(np.exp(-0.95670 * np.log(datalake['volume']/datalake['area'] + 1.36359)))
#
# dataSrc = gpd.read_file(r'T:\Usagers\macot620\Lake_IDs_2019\Swedish_lakes\Swedish_lakes.shp')
# dataSrc = dataSrc[dataSrc['ebint'].isin(datalake['ebint'])]
# dataSrc['ebint1'] = dataSrc['ebint']
# dataSrc=dataSrc.set_index('ebint1')
# datalake=datalake.set_index('ebint')
# dataSrc['secchi'] = datalake['secchi']
# dataSrc.to_file(r'T:\Usagers\macot620\Lake_IDs_2019\Swedish_lakes\Swedish_lakes_210.shp')

# print(dataid.loc['0x57d70b',:],datalake.loc['0x57d70b',:])
#
# datalake.to_csv(r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList1.csv",index=False)


def netcdf(output,file_name_txt,file_name_nc,datestart,dateend,lats,lons,unit,variable):
    try:

        # code from tutorial.
        file_name_txt = "%s/%s" % (output, file_name_txt)
        data = pd.read_csv(file_name_txt,sep=" ",header=None)
        #print(np.array((2,3,1,1)).shape)
        data.round(-3)
        data_a = data.to_numpy()
        x = len(data)
        y = len(data.columns)

        # create a file (Dataset object, also the root group).
        #file_name_nc = "%s/%s"%(output,file_name_nc)
        rootgrp = Dataset(file_name_nc, 'w', format='NETCDF4')
        print(rootgrp.file_format)

        # walk the group tree using a Python generator.
        def walktree(top):
            values = top.groups.values()
            yield values
            for value in top.groups.values():
                for children in walktree(value):
                    yield  children
        print(rootgrp)
        for children in walktree(rootgrp):
            for child in children:
                print(child)

        # dimensions.
        if y != 1:
            level = rootgrp.createDimension('levlak',y)
        time = rootgrp.createDimension('time', None)
        lat = rootgrp.createDimension('lat', 1)
        lon = rootgrp.createDimension('lon', 1)

        print(rootgrp.dimensions)

        print(time)

        # variables.
        times = rootgrp.createVariable('time','f8',('time',))
        if y!=1:
            levels = rootgrp.createVariable('levlak','f8',('levlak',))
        latitudes = rootgrp.createVariable('lat','f4',('lat',))
        longitudes = rootgrp.createVariable('lon','f4',('lon',))
        if y!=1:
            temp = rootgrp.createVariable('%s'%variable,'f4',('time','levlak','lat','lon',))
        else:
            temp = rootgrp.createVariable('%s' % variable, 'f4', ('time',  'lat', 'lon',))
        print(temp)
        # create variable in a group using a path.



        # attributes.
        #import time
        rootgrp.contact = "Raoul-Marie Couture <Raoul.Couture@chm.ulaval.ca>"
        rootgrp.institution = "Universite Laval (Ulaval)"
        rootgrp.comment = "Data prepared for ISIMIP2b"

        latitudes.longname = "latitude"
        #time.longname = "time"
        longitudes.longname = "longitude"
        temp.longname = output_long.get(variable)
        latitudes.units = 'degrees_north'
        longitudes.units = 'degrees_east'
        latitudes.axis = "Y"
        longitudes.axis = "X"

        if y !=1:
            levels.units = 'm'
            levels.axis = "Z"
            levels.longname = "depth_below_water_surface"
            levels.positive = "down"
        temp.units = unit
        times.units = 'days since 1661-01-01 00:00:00'

        times.calendar = 'proleptic_gregorian'

        for name in rootgrp.ncattrs():
            print('Global attr', name, '=', getattr(rootgrp,name))


        rootgrp.close()

        # create some groups.
        rootgrp = Dataset(file_name_nc, 'a')
        #fcstgrp = rootgrp.createGroup('forecasts')
        #analgrp = rootgrp.createGroup('analyses')


        latitudes[:] = lats
        longitudes[:] = lons
        print('latitudes =\n',latitudes[:])
        print('longitudes =\n',longitudes[:])

        # append along two unlimited dimensions by assigning to slice.
        nlats = len(rootgrp.dimensions['lat'])
        nlons = len(rootgrp.dimensions['lon'])


        print('start')
        if y!=1:
            data_b = np.reshape(data_a, (x,y,1,1))
        else:
            data_b = np.reshape(data_a, (x, 1, 1))
        print(data_b.shape)
        print('temp shape before adding data = ',temp.shape)
        if y!=1:
            temp[0:x,0:y,:,:] = data_b
        else:
            temp[0:x, :, :] = data_b
        print('temp shape after adding data = ',temp.shape)
        # levels have grown, but no values yet assigned.
        if y!=1:
            print('levels shape after adding pressure data = ',levels.shape)
            levels[:] = [x for x in np.arange(0,y,1)]

        r = range((datetime(dateend,12,31)-datetime(datestart,1,1)).days+1)
        print(r)
        dates = [datetime(datestart,1,1) + timedelta(days=x) for x in r]
        times[:] = date2num(dates,units=times.units,calendar=times.calendar)
        print('time values (in units %s): ' % times.units+'\\n',times[:])
        dates = num2date(times[:],units=times.units,calendar=times.calendar)
        print('dates corresponding to time values:\\n',dates)

        rootgrp.close()

        if y!=1:
            print(file_name_nc[:-1])
            if not os.path.exists(os.path.join(file_name_nc[:-1])):
                os.system("ncks -h -7 -L 9 %s %s"%(file_name_nc,file_name_nc[:-1]))
                os.remove(file_name_nc)

        if os.path.exists(file_name_nc[:-1]) or os.path.exists(file_name_nc):
            os.remove(os.path.join(output,file_name_txt))
        return 1
    except:
        print("bug")
        if os.path.exists(file_name_nc):
            os.remove(file_name_nc)
        if os.path.exists(file_name_nc[-1]):
            os.remove(file_name_nc[-1])

        return 0




if __name__ == "__main__":

    datestart = 1979
    dateend = 2016
    lake = 'Annecy'
    output = 'D:\output\FR\Annecy'
    lats = float(run_myLake_ISIMIP.get_latitude(lake, "D:/forcing_data","EWEMBI","historical"))
    lons = float(run_myLake_ISIMIP.get_longitude(lake, "D:/forcing_data","EWEMBI","historical"))
    unit='unitless'
    variable = 'strat'
    file_name_txt = "MyLake_EWEMBI_historical_nosoc_co2_%s_local_daily_1979_2016.txt"%variable
    model_name = "MyLake"
    gcm_observation = "GCM"
    bias = 'EWEMBI'
    climate = 'historical'
    socio = "nosoc"
    sens = "co2"
    region = "local"
    timestep = "daily"
    increment = "day"

    file_name_nc = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.nc4" % (model_name, bias, climate, socio, sens, variable, lake, timestep, datestart, dateend)
    netcdf(output, file_name_txt, file_name_nc, datestart, dateend, lats, lons, unit,variable)