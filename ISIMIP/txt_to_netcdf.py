from netCDF4 import Dataset
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import run_myLake_ISIMIP
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num, date2index
import os
import shutil

output_long={"strat":"Thermal_stratification", "watertemp":"Water_temperature", "thermodepth":"Depth_of_Thermocline", "ice":"Lake_ice_cover",
             "icetick":"Ice_thickness","snowtick":"Snow_thickness", "sensheatf": "Sensible_heat_flux_at_the_lake-atmosphere_interface",
             "latentheatf":"Latent_heat_flux_at_the_lake-atmosphere_interface", "lakeheatf": "Downward_heat_flux_at_the_lake-atmosphere_interface",
             "albedo":"Surface_albedo",
             "turbdiffheat":"Turbulent_diffusivity_of_heat", "sedheatf":"Sediment_upward_heat_flux_at_the_lakesediment_interface"}



def netcdf(output,file_name_txt,file_name_nc,datestart,dateend,lats,lons,unit,variable):
    try:

        # code from tutorial.
        file_name_txt = "%s/%s" % (output, file_name_txt)
        data = pd.read_csv(file_name_txt,sep=" ",header=None)
        print(data.loc[0][0])
        try:
            int(data.loc[0][0])
        except:
            data = data.drop([0], axis=1)
            nbrcol = len(data.columns)
            data.columns = [x for x in range(0,nbrcol)]



        #print(np.array((2,3,1,1)).shape)
        data.round(-3)
        data = data.replace(np.nan,1.e+20)
        data = data.replace('1.e+20f', 1.e+20)
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


        if variable == 'lakeicefrac':
            variable = 'icetick'
        # variables.
        times = rootgrp.createVariable('time','f8',('time',))
        if y!=1:
            levels = rootgrp.createVariable('levlak','f8',('levlak',))
        latitudes = rootgrp.createVariable('lat','f4',('lat',))
        longitudes = rootgrp.createVariable('lon','f4',('lon',))
        if y!=1:
            temp = rootgrp.createVariable('%s'%variable,'f4',('time','levlak','lat','lon',),fill_value=1.e+20)
        else:
            temp = rootgrp.createVariable('%s' % variable, 'f4', ('time',  'lat', 'lon',),fill_value=1.e+20)
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
        temp.missing_value = 1.e+20
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

        # dates = num2date(times[:],units=times.units,calendar=times.calendar)
        # print('dates corresponding to time values:\\n',dates)

        rootgrp.close()

        if y!=1:
            print(file_name_nc[:-1])
            if not os.path.exists(os.path.join(file_name_nc[:-1])):
                os.system("ncks -h -7 -L 9 %s %s"%(file_name_nc,file_name_nc[:-1]))
                os.remove(file_name_nc)

        if os.path.exists(file_name_nc[:-1]) or os.path.exists(file_name_nc):
            os.remove(os.path.join(output,file_name_txt))

        finaloutput = r"D:\final_files"
        try:
            if os.path.exists(os.path.join(file_name_nc[:-1])):
                shutil.copyfile(os.path.join(file_name_nc[:-1]),os.path.join(finaloutput,  file_name_nc[:-1]))
                os.remove(os.path.join(file_name_nc[:-1]))
            elif os.path.exists(os.path.join( file_name_nc)):
                shutil.copyfile(os.path.join( file_name_nc),os.path.join(finaloutput, file_name_nc))
                os.remove(os.path.join(file_name_nc))
        except:
            print("issue with moving file from isimip to final_files")
        return 1

    except:
        print("bug creation file file_name_nc")

        return 0


# mymodel, model, scenario, nooc, co2, variable, lake, daily, date_start, dateend = file_name_nc.strip().split('_')
#
# index = range(0, 60 * 21 * 12)
# columns = ['lake', 'model', 'scenario','variable'] + ["file","levlak",'lat','lon','time','value_vari']
# tableau = pd.DataFrame(index=index, columns=columns)

def validationnetcdf(file_name_nc,vari,lats,lons,maxdepth,datestart,dateend):
    import netCDF4
    import numpy as np

    limit_variables = {"strat": [ 0,  1], "watertemp": [  272, 320],
    "thermodepth": [  0, maxdepth], "ice": [ 0, 1],
    "icetick": [0, 16], "snowtick": [ 0, 10],
    "sensheatf": [-800, 10000], "latentheatf": [-10000, 2500],
    "lakeheatf": [-3000, 10000], "albedo": [0.001, 1],
    "turbdiffheat": [ 0, 50], "sedheatf": [-50, 50]}
    finaloutput =r"D:\final_files"
    readyoutput = r"D:\ready"
    good = True
    stop = False
    index=0
    if os.path.exists(os.path.join(finaloutput,file_name_nc[:-1])):
        netcdf = netCDF4.Dataset(os.path.join(finaloutput,file_name_nc[:-1]))
        #if not os.path.exists(os.path.join(finaloutput,file_name_nc[:-1])):
        #    shutil.copyfile(os.path.join(file_name_nc[:-1]),os.path.join(finaloutput,file_name_nc[:-1]))
    elif os.path.exists(os.path.join(finaloutput,file_name_nc)):
        netcdf = netCDF4.Dataset(os.path.join(finaloutput,file_name_nc))
        #if not os.path.exists(os.path.join(finaloutput, file_name_nc)):
        #    shutil.copyfile(os.path.join( file_name_nc), os.path.join(finaloutput, file_name_nc))
    else:
        stop = True
        good = False
        if os.path.exists(os.path.join(readyoutput, file_name_nc[:-1])) or os.path.exists(os.path.join(readyoutput,file_name_nc)) or os.path.exists(os.path.join(r"D:\move", file_name_nc[:-1])) or os.path.exists(os.path.join(r"D:\move",file_name_nc)):
            return ["file exist", "ok", "ok", "ok", "ok", "ok"]
        else:
            return ["file does not exist","","","","",""]

    if not stop:

        listinfor= {"file":"file exists"}

        if not "levlak" in netcdf.variables.keys():
            listinfor["levlak_min"] = " "
            listinfor["levlak_max"] = " "
        for variable in netcdf.variables.keys():
            var = netcdf.variables[variable]
            min = np.min(var)
            max = np.max(var)
            if variable == "levlak":
                if min == 0:
                    listinfor["levlak_min"] = "ok min levlak"
                else:
                    listinfor["levlak_min"] = "not_ok min levlak"
                    print('prblem min = %s and not 0'%min)
                if max == maxdepth or max == (maxdepth-1) or max == (maxdepth-2) or max == (maxdepth-3):
                    listinfor["levlak_max"] = "ok max levlak"
                else:
                    listinfor["levlak_max"] = "not_ok max levlak"
                    print('prblem max = %s and not %s +- 3' % max)
            elif variable == "lat":
                if len(var) == 1:
                    if min == lats:
                        listinfor["lat"] = "ok lat"
                    else:
                        listinfor["lat"] = "not_ok lat"
                else:
                    listinfor["lat"] = "not_ok long lat"
            elif variable == "lon":
                if len(var) == 1:
                    if min == lons:
                        listinfor["lon"] = "ok lon"
                    else:
                        listinfor["lon"] = "not_ok lon"
                else:
                    listinfor["lon"] = "not_ok long lon"

            elif variable == "time":
                print(len(var),(datetime(dateend,12,31)-datetime(datestart,1,1)).days+1)
                if len(var) == ((datetime(dateend,12,31)-datetime(datestart,1,1)).days+1):
                    first = date2num(datetime(datestart,1,1), units='days since 1661-01-01 00:00:00', calendar='proleptic_gregorian')
                    last = date2num(datetime(dateend, 12, 31), units='days since 1661-01-01 00:00:00',
                                     calendar='proleptic_gregorian')

                    var1 = netcdf.variables[vari]
                    min1 = np.min(var1)
                    max1 = np.max(var1)
                    print(min1,limit_variables.get(vari)[0],max1,limit_variables.get(vari)[1])
                    if min1 >= limit_variables.get(vari)[0] or min1 > 1e+19:
                        if max1 <= limit_variables.get(vari)[1] or max1 > 1e+19:
                            if os.path.exists(os.path.join(finaloutput, file_name_nc[:-1])):
                                if not os.path.exists(os.path.join(r"D:\ready", file_name_nc[:-1])):
                                    shutil.copyfile(os.path.join(finaloutput, file_name_nc[:-1]),
                                                    os.path.join(r"D:\ready", file_name_nc[:-1]))

                            elif os.path.exists(os.path.join(finaloutput, file_name_nc)):
                                if not os.path.exists(os.path.join(r"D:\ready", file_name_nc)):
                                    shutil.copyfile(os.path.join(finaloutput, file_name_nc), os.path.join(r"D:\ready", file_name_nc))


                    if min == first:
                        listinfor["time_start"] = "ok start time"
                    else:
                        listinfor["time_start"] = "not_ok start time VALUE %s"%min
                        good = False
                    if max == last:
                        listinfor["time_end"] = "ok end time"
                    else:
                        listinfor["time_end"] = "not_ok end time VALUE %s"%max
                        good=False
                else:
                    first = date2num(datetime(datestart, 1, 1), units='days since 1661-01-01 00:00:00',
                                     calendar='proleptic_gregorian')
                    last = date2num(datetime(dateend, 12, 31), units='days since 1661-01-01 00:00:00',
                                    calendar='proleptic_gregorian')
                    # if os.path.exists(os.path.join(finaloutput, file_name_nc[:-1])):
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc[:-1]))
                    #     shutil.copyfile(os.path.join(finaloutput, file_name_nc[:-1]), os.path.join(file_name_nc[:-1]))
                    # else:
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc))
                    #     if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #         shutil.copyfile(os.path.join(finaloutput, file_name_nc), os.path.join(file_name_nc))
                    if min == first:
                        listinfor["time_start"] = "ok start time"
                    else:
                        listinfor["time_start"] = "not_ok start time VALUE %s"%min
                        good=False
                    if max == last:
                        listinfor["time_end"] = "ok_errorlen end time"
                        good=False
                    else:
                        listinfor["time_end"] = "not_ok_errorlen end time VALUE %s"%max
                        good=False

            elif variable == vari:
                if min >= limit_variables.get(vari)[0] or min > 1e+19:
                    listinfor["variable_min"] = "ok min variable"
                else:
                    listinfor["variable_min"] = "not_ok min variable VALUE %s"%min
                    # if os.path.exists(os.path.join(finaloutput, file_name_nc[:-1])):
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc[:-1]))
                    #     shutil.copyfile(os.path.join(finaloutput,file_name_nc[:-1]),os.path.join(file_name_nc[:-1]))
                    # else:
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc))
                    #     if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #         shutil.copyfile(os.path.join(finaloutput, file_name_nc), os.path.join( file_name_nc))

                if max <= limit_variables.get(vari)[1] or max > 1e+19:
                    listinfor["variable_max"] = "ok max variable"
                else:
                    listinfor["variable_max"] = "not_ok max variable VALUE %s"%max
                    good=False
                    # if os.path.exists(os.path.join(finaloutput, file_name_nc[:-1])):
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc[:-1]))
                    #     shutil.copyfile(os.path.join(finaloutput,file_name_nc[:-1]),os.path.join(file_name_nc[:-1]))
                    # else:
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc))
                    #     if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #         shutil.copyfile(os.path.join(finaloutput, file_name_nc), os.path.join( file_name_nc))


            else:
                return ["file exist but unknow variable","","","","",""]
        netcdf.close()
        if os.path.exists((os.path.join(r"D:\ready", file_name_nc[:-1]))):
                os.remove(os.path.join(finaloutput, file_name_nc[:-1]))
        elif os.path.exists((os.path.join(r"D:\ready", file_name_nc))):
                os.remove(os.path.join(finaloutput, file_name_nc))
        return [listinfor["file"],listinfor['levlak_min'],listinfor["levlak_max"],listinfor["lat"],listinfor["lon"],listinfor["time_start"],listinfor["time_end"],listinfor["variable_min"],listinfor["variable_max"]]



if __name__ == "__main__":
    file_name_nc="mylake_ewembi_historical_nosoc_co2_turbdiffheat_dickie_lake_daily_1979_2016.nc4"
    if not os.path.exists(os.path.join(file_name_nc[:-1])):
        os.system("ncks -h -7 -L 9 %s %s" % (file_name_nc, file_name_nc[:-1]))
        os.remove(file_name_nc)



    # datestart = 1979
    # dateend = 2016
    # lake = 'Annecy'
    # output = 'D:\output\FR\Annecy'
    # lats = float(run_myLake_ISIMIP.get_latitude(lake, "D:/forcing_data","EWEMBI","historical"))
    # lons = float(run_myLake_ISIMIP.get_longitude(lake, "D:/forcing_data","EWEMBI","historical"))
    # unit='unitless'
    # variable = 'strat'
    # file_name_txt = "MyLake_EWEMBI_historical_nosoc_co2_%s_local_daily_1979_2016.txt"%variable
    # model_name = "MyLake"
    # gcm_observation = "GCM"
    # bias = 'EWEMBI'
    # climate = 'historical'
    # socio = "nosoc"
    # sens = "co2"
    # region = "local"
    # timestep = "daily"
    # increment = "day"
    #
    # file_name_nc = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.nc4" % (model_name, bias, climate, socio, sens, variable, lake, timestep, datestart, dateend)
    # netcdf(output, file_name_txt, file_name_nc, datestart, dateend, lats, lons, unit,variable)