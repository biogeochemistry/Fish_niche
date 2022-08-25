import os
import pandas as pd
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
from numpy import arange, nan, reshape, sqrt
import shutil
import csv
from joblib import Parallel, delayed
import multiprocessing
from Main_fish_niche import summary_characteristics_lake_parallel,summary_characteristics_lake1

num_cores = multiprocessing.cpu_count()  # to modified if you want to choose the number of cores used.

output_final = r"G:\Fish_Niche_archive"
output_initial = r"F:\MCOTE\output"
lake_list = "2017SwedenList.csv"

variables = ['clt', 'hurs', 'tas', 'rsds', 'ps', 'pr', 'sfcWind']

expectedfs = ['Tzt', 'O2zt', 'lambdazt', 'His', 'PARzt','Secchi']
expectedvariable = ["watertemp", "oxygen", "light_attenuation_coeff", "ice_cover", "PAR"]

model_reduced = {1: "KNMI", 2: "DMI", 3: "MPI", 4: "MOH", 5: "IPS", 6: "CNR"}

scenario_reduced = {1: "historical_1971-1980", 2: "historical_2001-2010", 3: "rcp45_2031-2040", 4: "rcp45_2061-2070",
                    5: "rcp45_2091-2100", 6: "rcp85_2031-2040", 7: "rcp85_2061-2070", 8: "rcp85_2091-2100"}

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

output_long = {"strat": "Thermal_stratification", "watertemp": "Water_temperature",
               "thermodepth": "Depth_of_Thermocline", "ice": "Lake_ice_cover",
               "icetick": "Ice_thickness", "snowtick": "Snow_thickness",
               "sensheatf": "Sensible_heat_flux_at_the_lake-atmosphere_interface",
               "latentheatf": "Latent_heat_flux_at_the_lake-atmosphere_interface",
               "lakeheatf": "Downward_heat_flux_at_the_lake-atmosphere_interface",
               "albedo": "Surface_albedo",
               "turbdiffheat": "Turbulent_diffusivity_of_heat",
               "sedheatf": "Sediment_upward_heat_flux_at_the_lakesediment_interface"}


def netcdf(output, file_name_txt, file_name_nc, datestart, dateend, lats, lons, unit, variable):
    """

    :param output:
    :param file_name_txt:
    :param file_name_nc:
    :param datestart:
    :param dateend:
    :param lats:
    :param lons:
    :param unit:
    :param variable:
    :return:
    """
    try:
        for variable in expectedfs:
            file_name_txt = "%s/%s" % (output, file_name_txt)
            data = pd.read_csv(file_name_txt, sep=" ", header=None)
            print(data.loc[0][0])
            try:
                int(data.loc[0][0])
            except:
                data = data.drop([0], axis=1)
                nbrcol = len(data.columns)
                data.columns = [x for x in range(0, nbrcol)]

            data.round(-3)
            data = data.replace(nan, 1.e+20)
            data = data.replace('1.e+20f', 1.e+20)
            data_a = data.to_numpy()
            x = len(data)
            y = len(data.columns)

            file_name = Dataset(file_name_nc, 'w', format='NETCDF4')
            print(file_name.file_format)

            def walktree(top):
                values = top.groups.values()
                yield values
                for value in top.groups.values():
                    for children in walktree(value):
                        yield children

            print(file_name)
            for children in walktree(file_name):
                for child in children:
                    print(child)

            if variable == 'lakeicefrac':
                variable = 'icetick'
            # variables.
            times = file_name.createVariable('time', 'f8', ('time',))
            if y != 1:
                levels = file_name.createVariable('levlak', 'f8', ('levlak',))
            latitudes = file_name.createVariable('lat', 'f4', ('lat',))
            longitudes = file_name.createVariable('lon', 'f4', ('lon',))
            if y != 1:
                temp = file_name.createVariable('%s' % variable, 'f4', ('time', 'levlak', 'lat', 'lon',),
                                                fill_value=1.e+20)
            else:
                temp = file_name.createVariable('%s' % variable, 'f4', ('time', 'lat', 'lon',), fill_value=1.e+20)
            print(temp)

            temp.longname = output_long.get(variable)
            temp.units = unit
            temp.missing_value = 1.e+20
            print('start')
            if y != 1:
                data_b = reshape(data_a, (x, y, 1, 1))
            else:
                data_b = reshape(data_a, (x, 1, 1))
            print(data_b.shape)
            print('temp shape before adding data = ', temp.shape)
            if y != 1:
                temp[0:x, 0:y, :, :] = data_b
            else:
                temp[0:x, :, :] = data_b
            print('temp shape after adding data = ', temp.shape)

        # dimensions.
        if y != 1:
            level = file_name.createDimension('levlak', y)
        time = file_name.createDimension('time', None)
        lat = file_name.createDimension('lat', 1)
        lon = file_name.createDimension('lon', 1)

        print(file_name.dimensions)

        print(time)

        # attributes.
        # import time
        file_name.contact = "Raoul-Marie Couture <Raoul.Couture@chm.ulaval.ca>"
        file_name.institution = "Universite Laval (Ulaval)"
        file_name.comment = "Data prepared for Fish_niche project"

        latitudes.longname = "latitude"
        # time.longname = "time"
        longitudes.longname = "longitude"

        latitudes.units = 'degrees_north'
        longitudes.units = 'degrees_east'
        latitudes.axis = "Y"
        longitudes.axis = "X"

        if y != 1:
            levels.units = 'm'
            levels.axis = "Z"
            levels.longname = "depth_below_water_surface"
            levels.positive = "down"

        times.units = 'days since 1661-01-01 00:00:00'

        times.calendar = 'proleptic_gregorian'

        for name in file_name.ncattrs():
            print('Global attr', name, '=', getattr(file_name, name))

        file_name.close()

        # create some groups.
        file_name = Dataset(file_name_nc, 'a')

        latitudes[:] = lats
        longitudes[:] = lons
        print('latitudes =\n', latitudes[:])
        print('longitudes =\n', longitudes[:])

        # append along two unlimited dimensions by assigning to slice.
        nlats = len(file_name.dimensions['lat'])
        nlons = len(file_name.dimensions['lon'])

        # levels have grown, but no values yet assigned.
        if y != 1:
            print('levels shape after adding pressure data = ', levels.shape)
            levels[:] = [x for x in arange(0, y, 1)]

        r = range((datetime(dateend, 12, 31) - datetime(datestart, 1, 1)).days + 1)
        print(r)
        dates = [datetime(datestart, 1, 1) + timedelta(days=x) for x in r]
        times[:] = date2num(dates, units=times.units, calendar=times.calendar)
        print('time values (in units %s): ' % times.units + '\\n', times[:])

        # dates = num2date(times[:],units=times.units,calendar=times.calendar)
        # print('dates corresponding to time values:\\n',dates)

        file_name.close()

        if y != 1:
            print(file_name_nc[:-1])
            # if not os.path.exists(os.path.join(file_name_nc[:-1])):
            #     os.system("ncks -h -7 -L 9 %s %s"%(file_name_nc,file_name_nc[:-1]))
            #     os.remove(file_name_nc)

        if os.path.exists(file_name_nc[:-1]) or os.path.exists(file_name_nc):
            os.remove(os.path.join(output, file_name_txt))

        finaloutput = r"D:\final_files"
        try:
            if os.path.exists(os.path.join(file_name_nc[:-1])):
                shutil.copyfile(os.path.join(file_name_nc[:-1]), os.path.join(finaloutput, file_name_nc[:-1]))
                os.remove(os.path.join(file_name_nc[:-1]))
            elif os.path.exists(os.path.join(file_name_nc)):
                shutil.copyfile(os.path.join(file_name_nc), os.path.join(finaloutput, file_name_nc))
                os.remove(os.path.join(file_name_nc))
        except:
            print("issue with moving file from isimip to final_files")
        return 1

    except:
        print("bug creation file file_name_nc")

        return 0


def validationnetcdf(file_name_nc, vari, lats, lons, maxdepth, datestart, dateend):
    """

    :param file_name_nc:
    :param vari:
    :param lats:
    :param lons:
    :param maxdepth:
    :param datestart:
    :param dateend:
    :return:
    """

    limit_variables = {"strat": [0, 1], "watertemp": [272, 320],
                       "thermodepth": [0, maxdepth], "ice": [0, 1],
                       "icetick": [0, 16], "snowtick": [0, 10],
                       "sensheatf": [-800, 10000], "latentheatf": [-10000, 2500],
                       "lakeheatf": [-3000, 10000], "albedo": [0.001, 1],
                       "turbdiffheat": [0, 50], "sedheatf": [-50, 50]}
    finaloutput = r"D:\final_files"
    readyoutput = r"D:\ready"

    stop = False

    if os.path.exists(os.path.join(finaloutput, file_name_nc[:-1])):
        netcdf_dataset = Dataset(os.path.join(finaloutput, file_name_nc[:-1]))
        # if not os.path.exists(os.path.join(finaloutput,file_name_nc[:-1])):
        #    shutil.copyfile(os.path.join(file_name_nc[:-1]),os.path.join(finaloutput,file_name_nc[:-1]))
    elif os.path.exists(os.path.join(finaloutput, file_name_nc)):
        netcdf_dataset = Dataset(os.path.join(finaloutput, file_name_nc))
        # if not os.path.exists(os.path.join(finaloutput, file_name_nc)):
        #    shutil.copyfile(os.path.join( file_name_nc), os.path.join(finaloutput, file_name_nc))
    else:
        stop = True

        if os.path.exists(os.path.join(readyoutput, file_name_nc[:-1])) or os.path.exists(
                os.path.join(readyoutput, file_name_nc)) or os.path.exists(
            os.path.join(r"D:\move", file_name_nc[:-1])) or os.path.exists(os.path.join(r"D:\move", file_name_nc)):
            return ["file exist", "ok", "ok", "ok", "ok", "ok"]
        else:
            return ["file does not exist", "", "", "", "", ""]

    if not stop:

        listinfor = {"file": "file exists"}

        if not "levlak" in netcdf_dataset.variables.keys():
            listinfor["levlak_min"] = " "
            listinfor["levlak_max"] = " "
        for variable in netcdf_dataset.variables.keys():
            var = netcdf_dataset.variables[variable]
            min_var = min(var)
            max_var = max(var)
            if variable == "levlak":
                if min_var == 0:
                    listinfor["levlak_min"] = "ok min levlak"
                else:
                    listinfor["levlak_min"] = "not_ok min levlak"
                    print('prblem min = %s and not 0' % min_var)
                if max_var == maxdepth or max_var == (maxdepth - 1) or max_var == (maxdepth - 2) or max_var == (
                        maxdepth - 3):
                    listinfor["levlak_max"] = "ok max levlak"
                else:
                    listinfor["levlak_max"] = "not_ok max levlak"
                    print('prblem max = %s and not %s +- 3' % max_var)
            elif variable == "lat":
                if len(var) == 1:
                    if min_var == lats:
                        listinfor["lat"] = "ok lat"
                    else:
                        listinfor["lat"] = "not_ok lat"
                else:
                    listinfor["lat"] = "not_ok long lat"
            elif variable == "lon":
                if len(var) == 1:
                    if min_var == lons:
                        listinfor["lon"] = "ok lon"
                    else:
                        listinfor["lon"] = "not_ok lon"
                else:
                    listinfor["lon"] = "not_ok long lon"

            elif variable == "time":
                print(len(var), (datetime(dateend, 12, 31) - datetime(datestart, 1, 1)).days + 1)
                if len(var) == ((datetime(dateend, 12, 31) - datetime(datestart, 1, 1)).days + 1):
                    first = date2num(datetime(datestart, 1, 1), units='days since 1661-01-01 00:00:00',
                                     calendar='proleptic_gregorian')
                    last = date2num(datetime(dateend, 12, 31), units='days since 1661-01-01 00:00:00',
                                    calendar='proleptic_gregorian')

                    var1 = netcdf_dataset.variables[vari]
                    min1 = min(var1)
                    max1 = max(var1)
                    print(min1, limit_variables.get(vari)[0], max1, limit_variables.get(vari)[1])
                    if min1 >= limit_variables.get(vari)[0] or min1 > 1e+19:
                        if max1 <= limit_variables.get(vari)[1] or max1 > 1e+19:
                            if os.path.exists(os.path.join(finaloutput, file_name_nc[:-1])):
                                if not os.path.exists(os.path.join(r"D:\ready", file_name_nc[:-1])):
                                    shutil.copyfile(os.path.join(finaloutput, file_name_nc[:-1]),
                                                    os.path.join(r"D:\ready", file_name_nc[:-1]))

                            elif os.path.exists(os.path.join(finaloutput, file_name_nc)):
                                if not os.path.exists(os.path.join(r"D:\ready", file_name_nc)):
                                    shutil.copyfile(os.path.join(finaloutput, file_name_nc),
                                                    os.path.join(r"D:\ready", file_name_nc))

                    if min_var == first:
                        listinfor["time_start"] = "ok start time"
                    else:
                        listinfor["time_start"] = "not_ok start time VALUE %s" % min_var

                    if max_var == last:
                        listinfor["time_end"] = "ok end time"
                    else:
                        listinfor["time_end"] = "not_ok end time VALUE %s" % max_var

                else:
                    first = date2num(datetime(datestart, 1, 1), units='days since 1661-01-01 00:00:00',
                                     calendar='proleptic_gregorian')
                    last = date2num(datetime(dateend, 12, 31), units='days since 1661-01-01 00:00:00',
                                    calendar='proleptic_gregorian')
                    # if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc))
                    #     if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #         shutil.copyfile(os.path.join(finaloutput, file_name_nc), os.path.join(file_name_nc))
                    if min_var == first:
                        listinfor["time_start"] = "ok start time"
                    else:
                        listinfor["time_start"] = "not_ok start time VALUE %s" % min_var

                    if max_var == last:
                        listinfor["time_end"] = "ok_errorlen end time"

                    else:
                        listinfor["time_end"] = "not_ok_errorlen end time VALUE %s" % max_var


            elif variable == vari:
                if min_var >= limit_variables.get(vari)[0] or min_var > 1e+19:
                    listinfor["variable_min"] = "ok min variable"
                else:
                    listinfor["variable_min"] = "not_ok min variable VALUE %s" % min_var
                    # if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc))
                    #     if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #         shutil.copyfile(os.path.join(finaloutput, file_name_nc), os.path.join( file_name_nc))

                if max_var <= limit_variables.get(vari)[1] or max_var > 1e+19:
                    listinfor["variable_max"] = "ok max variable"
                else:
                    listinfor["variable_max"] = "not_ok max variable VALUE %s" % max_var

                    # if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #     netcdf = netCDF4.Dataset(os.path.join(finaloutput, file_name_nc))
                    #     if os.path.exists(os.path.join(finaloutput, file_name_nc)):
                    #         shutil.copyfile(os.path.join(finaloutput, file_name_nc), os.path.join( file_name_nc))


            else:
                return ["file exist but unknow variable", "", "", "", "", ""]
        netcdf_dataset.close()
        if os.path.exists((os.path.join(r"D:\ready", file_name_nc[:-1]))):
            os.remove(os.path.join(finaloutput, file_name_nc[:-1]))
        elif os.path.exists((os.path.join(r"D:\ready", file_name_nc))):
            os.remove(os.path.join(finaloutput, file_name_nc))
        return [listinfor["file"], listinfor['levlak_min'], listinfor["levlak_max"], listinfor["lat"], listinfor["lon"],
                listinfor["time_start"], listinfor["time_end"], listinfor["variable_min"], listinfor["variable_max"]]


def parallel(row):
    lake_folder = os.path.join(output_final, "IO","model_inputs", "%s" % lakes_data['lake_id'][row])
    if not os.path.exists(lake_folder):
        os.makedirs(lake_folder)
    eh = lakes_data['ebhex'][row]
    eh = eh[2:] if eh[:2] == '0x' else eh
    while len(eh) < 6:
        eh = '0' + eh
    d1, d2, d3 = eh[:2], eh[:4], eh[:6]

    for model in models:
        print(row, model)
        for scenario in scenarios:

            print(row, model, scenario)
            try:
                original_input_path = os.path.join(output_initial, d1, d2, d3, "EUR-11_%s_%s-%s_%s_%s0101-%s1231" % (
                    models[model][0], scenarios[scenario][0],
                    scenarios[scenario][2], models[model][1],
                    scenarios[scenario][1], scenarios[scenario][3] + 4))
                # copy input
                input_path = os.path.join(original_input_path, "2020input")
                if not os.path.exists(
                        os.path.join(lake_folder, "%s_%s_input" % (model_reduced[model], scenario_reduced[scenario]))):
                    if os.path.exists(input_path):
                        shutil.copy2(input_path, os.path.join(lake_folder, "%s_%s_input" % (
                        model_reduced[model], scenario_reduced[scenario])))




                # copy init and par
                if model == 2 and scenario == 2:
                    par_path = os.path.join(original_input_path, "2020par")
                    if os.path.exists(par_path):
                        shutil.copy2(par_path, os.path.join(lake_folder, "par"))
                    else:
                        print("%s par error"%lakes_data['lake_id'][row])

                    init_path = os.path.join(original_input_path, "2020init")
                    if os.path.exists(init_path):
                        shutil.copy2(init_path, os.path.join(lake_folder, "init"))
                    else:
                        print("%s init error"%lakes_data['lake_id'][row])

                    laketest = lakes_data['lake_id'][row]
                    # print(lakes_data['lake_id'][row])
                    if lakes_data['lake_id'][row] in [30704,32276,33494,16765,14939,310,99045,6950,31895,33590,67035,698]:

                        par_path = os.path.join(original_input_path, "calibration_result", "2020par")
                        if os.path.exists(par_path):
                            shutil.copy2(par_path, os.path.join(lake_folder, "calibration_par"))
                        else:
                            print("%s calibration error" % lakes_data['lake_id'][row])



                # copy init and par
                lake_folder1 = os.path.join(output_final, "Postproc","stepwise_regression_result_all_model_and_scenario", "%s" % lakes_data['lake_id'][row],
                                           "%s_%s" % (
                                           model_reduced[model], scenario_reduced[scenario]))
                if not os.path.exists(lake_folder1):
                    os.makedirs(lake_folder1)
                for variable in expectedfs:
                    post_path = os.path.join(original_input_path, "%s.csv"%variable)
                    if variable == 'Tzt':
                        variablei = 'Temperature'
                    elif variable == 'O2zt':
                        variablei = 'Oxygen'
                    elif variable == 'lambdazt':
                        variablei = 'lambda'
                    elif variable == 'His':
                        variablei = 'Ice'
                    elif variable == 'PARzt':
                        variablei = "PAR"
                    elif variable == "Secchi":
                        variablei = "Secchi"

                    if not os.path.exists(os.path.join(lake_folder1, "%s.csv"%variable)):
                        if os.path.exists(post_path):
                            shutil.copy2(post_path, os.path.join(lake_folder1, "%s.csv"%variablei))

                    # if model == 2 and scenario == 2:
                    #     # if lakes_data['lake_id'][row] == 698:
                    #     #     print('here')
                    #     # if os.path.exists(os.path.join(original_input_path, "calibration_result", "Tzt.csv")):
                    #     #     print(lakes_data['lake_id'][row])
                    #
                    #     if lakes_data['lake_id'][row] in [310,698, 14939,6950, 16765,30704, 31895,32276,33494,33590,67035, 99045]:
                    #


                    if model == 2 and scenario == 2:
                        if lakes_data['lake_id'][row] in [30704,32276,33494,16765,14939,310,99045,6950,31895,33590,67035,698]:
                            lake_foldercali = os.path.join(output_final, "Postproc", "calibration_result",
                                                           "%s" % lakes_data['lake_id'][row], "%s_%s" % (
                                                               model_reduced[model], scenario_reduced[scenario]))
                            if not os.path.exists(lake_foldercali):
                                os.makedirs(lake_foldercali)
                            if not os.path.exists(os.path.join(lake_foldercali, "%s.csv"%variable)):
                                par_path = os.path.join(original_input_path, "calibration_result","%s.csv"%variable)
                                if not os.path.exists(os.path.join(output_final, "Postproc", "calibration_result")):
                                    os.makedirs(os.path.join(output_final, "Postproc", "calibration_result"))

                                if os.path.exists(par_path):
                                    shutil.copy2(par_path, os.path.join(lake_foldercali,  "%s.csv"%variablei))

                            if variablei in ["Temperature","Oxygen","Secchi"]:

                                obs_folder = os.path.join(output_final, "Postproc","calibration_result",
                                                                "%s" % lakes_data['lake_id'][row],"%s_%s" % (model_reduced[model], scenario_reduced[scenario]))
                                if not os.path.exists(obs_folder):
                                    os.makedirs(obs_folder)
                                if not os.path.exists(os.path.join(obs_folder, "%scompare.csv" % variablei)):
                                    original_path = os.path.join(original_input_path, "calibration_result",
                                                                 "%scompare.csv" % variable)
                                    if os.path.exists(original_path):
                                        shutil.copy2(original_path, os.path.join(obs_folder, "%s_compare.csv" % variablei))
                                        with open(os.path.join(obs_folder, "%s_compare.csv" % variablei), 'r') as csvfile:
                                            csv_dict = [row for row in csv.DictReader(csvfile)]
                                            if len(csv_dict) == 0:
                                                os.remove(os.path.join(obs_folder, "%s_compare.csv" % variablei))

                                obs_folder = os.path.join(output_final, "IO", "obs", "%s" % lakes_data['lake_id'][row])

                                if not os.path.exists(obs_folder):
                                    os.makedirs(obs_folder)
                                if not os.path.exists(os.path.join(obs_folder, "Observed_%s.csv" % variablei)):

                                    original_path = os.path.join(original_input_path, "Observed_%s.csv" % variablei)
                                    if os.path.exists(original_path):
                                        shutil.copy2(original_path, os.path.join(obs_folder, "Observed_%s.csv" % variablei))
                                        with open(os.path.join(obs_folder, "Observed_%s.csv" % variablei),'r') as csvfile:
                                            csv_dict = [row for row in csv.DictReader(csvfile)]
                                            if len(csv_dict) == 0:
                                                os.remove(os.path.join(obs_folder, "Observed_%s.csv" % variablei))


            except:
                print("Error",row, model, scenario)

    print("End lake %s" % lakes_data['lake_id'][row])

def parallel2(row):
    lake_folder = os.path.join(output_final, "IO","model_inputs", "%s" % lakes_data['lake_id'][row])
    if not os.path.exists(lake_folder):
        os.makedirs(lake_folder)
    eh = lakes_data['ebhex'][row]
    eh = eh[2:] if eh[:2] == '0x' else eh
    while len(eh) < 6:
        eh = '0' + eh
    d1, d2, d3 = eh[:2], eh[:4], eh[:6]

    for model in [2]:#models:
        print(row, model)
        for scenario in [2]:#scenarios:

            print(row, model, scenario)
            try:
                original_input_path = os.path.join(output_initial, d1, d2, d3, "EUR-11_%s_%s-%s_%s_%s0101-%s1231" % (
                    models[model][0], scenarios[scenario][0],
                    scenarios[scenario][2], models[model][1],
                    scenarios[scenario][1], scenarios[scenario][3] + 4))
                # copy input
                input_path = os.path.join(original_input_path, "2020input")
                if not os.path.exists(
                        os.path.join(lake_folder, "%s_%s_input" % (model_reduced[model], scenario_reduced[scenario]))):
                    if os.path.exists(input_path):
                        shutil.copy2(input_path, os.path.join(lake_folder, "%s_%s_input" % (
                        model_reduced[model], scenario_reduced[scenario])))




                # copy init and par
                if model == 2 and scenario == 2:
                    par_path = os.path.join(original_input_path, "2020par")
                    if os.path.exists(par_path):
                        shutil.copy2(par_path, os.path.join(lake_folder, "par"))
                    else:
                        print("%s par error"%lakes_data['lake_id'][row])

                    init_path = os.path.join(original_input_path, "2020init")
                    if os.path.exists(init_path):
                        shutil.copy2(init_path, os.path.join(lake_folder, "init"))
                    else:
                        print("%s init error"%lakes_data['lake_id'][row])

                    laketest = lakes_data['lake_id'][row]
                    # print(lakes_data['lake_id'][row])
                    if lakes_data['lake_id'][row] in [30704,32276,33494,16765,14939,310,99045,6950,31895,33590,67035,698]:

                        par_path = os.path.join(original_input_path, "calibration_result", "2020par")
                        if os.path.exists(par_path):
                            shutil.copy2(par_path, os.path.join(lake_folder, "calibration_par"))
                        else:
                            print("%s calibration error" % lakes_data['lake_id'][row])



                # copy init and par
                lake_folder1 = os.path.join(output_final, "Postproc","stepwise_regression_result_all_model_and_scenario", "%s" % lakes_data['lake_id'][row],
                                           "%s_%s" % (
                                           model_reduced[model], scenario_reduced[scenario]))
                if not os.path.exists(lake_folder1):
                    os.makedirs(lake_folder1)
                for variable in expectedfs:
                    post_path = os.path.join(original_input_path, "%s.csv"%variable)
                    if variable == 'Tzt':
                        variablei = 'Temperature'
                    elif variable == 'O2zt':
                        variablei = 'Oxygen'
                    elif variable == 'lambdazt':
                        variablei = 'lambda'
                    elif variable == 'His':
                        variablei = 'Ice'
                    elif variable == 'PARzt':
                        variablei = "PAR"
                    elif variable == "Secchi":
                        variablei = "Secchi"

                    if not os.path.exists(os.path.join(lake_folder1, "%s.csv"%variable)):
                        if os.path.exists(post_path):
                            shutil.copy2(post_path, os.path.join(lake_folder1, "%s.csv"%variablei))

                    # if model == 2 and scenario == 2:
                    #     # if lakes_data['lake_id'][row] == 698:
                    #     #     print('here')
                    #     # if os.path.exists(os.path.join(original_input_path, "calibration_result", "Tzt.csv")):
                    #     #     print(lakes_data['lake_id'][row])
                    #
                    #     if lakes_data['lake_id'][row] in [310,698, 14939,6950, 16765,30704, 31895,32276,33494,33590,67035, 99045]:
                    #


                    if model == 2 and scenario == 2:
                        if lakes_data['lake_id'][row] in [30704,32276,33494,16765,14939,310,99045,6950,31895,33590,67035,698]:
                            lake_foldercali = os.path.join(output_final, "Postproc", "calibration_result",
                                                           "%s" % lakes_data['lake_id'][row], "%s_%s" % (
                                                               model_reduced[model], scenario_reduced[scenario]))
                            if not os.path.exists(lake_foldercali):
                                os.makedirs(lake_foldercali)
                            if not os.path.exists(os.path.join(lake_foldercali, "%s.csv"%variable)):
                                par_path = os.path.join(original_input_path, "calibration_result","%s.csv"%variable)
                                if not os.path.exists(os.path.join(output_final, "Postproc", "calibration_result")):
                                    os.makedirs(os.path.join(output_final, "Postproc", "calibration_result"))

                                if os.path.exists(par_path):
                                    shutil.copy2(par_path, os.path.join(lake_foldercali,  "%s.csv"%variablei))

                            if variablei in ["Temperature","Oxygen","Secchi"]:

                                obs_folder = os.path.join(output_final, "Postproc","calibration_result",
                                                                "%s" % lakes_data['lake_id'][row],"%s_%s" % (model_reduced[model], scenario_reduced[scenario]))
                                if not os.path.exists(obs_folder):
                                    os.makedirs(obs_folder)
                                if not os.path.exists(os.path.join(obs_folder, "%scompare.csv" % variablei)):
                                    original_path = os.path.join(original_input_path, "calibration_result",
                                                                 "%scompare.csv" % variable)
                                    if os.path.exists(original_path):
                                        shutil.copy2(original_path, os.path.join(obs_folder, "%s_compare.csv" % variablei))
                                        with open(os.path.join(obs_folder, "%s_compare.csv" % variablei), 'r') as csvfile:
                                            csv_dict = [row for row in csv.DictReader(csvfile)]
                                            if len(csv_dict) == 0:
                                                os.remove(os.path.join(obs_folder, "%s_compare.csv" % variablei))

                                obs_folder = os.path.join(output_final, "IO", "obs", "%s" % lakes_data['lake_id'][row])

                                if not os.path.exists(obs_folder):
                                    os.makedirs(obs_folder)
                                if not os.path.exists(os.path.join(obs_folder, "Observed_%s.csv" % variablei)):

                                    original_path = os.path.join(original_input_path, "Observed_%s.csv" % variablei)
                                    if os.path.exists(original_path):
                                        shutil.copy2(original_path, os.path.join(obs_folder, "Observed_%s.csv" % variablei))
                                        with open(os.path.join(obs_folder, "Observed_%s.csv" % variablei),'r') as csvfile:
                                            csv_dict = [row for row in csv.DictReader(csvfile)]
                                            if len(csv_dict) == 0:
                                                os.remove(os.path.join(obs_folder, "Observed_%s.csv" % variablei))


            except:
                print("Error",row, model, scenario)

    print("End lake %s" % lakes_data['lake_id'][row])

def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)


if __name__ == "__main__":
    verification_all = []
    lakes_data = pd.read_csv(lake_list, encoding='latin')
    # Inputs copy
    inputs = os.path.join(output_final, "IO")
    if not os.path.exists(inputs):
        os.makedirs(inputs)

    # for lake in range(0, len(lakes_data)):
    #     parallel(lake)
    # summary_characteristics_lake1(0, lakes_listcsv=r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList.csv")
    # Parallel ( n_jobs=num_cores ) ( delayed ( parallel2 ) (lake) for lake in range(0,len(lakes_data)))#series[i-1]
    #
    #
    # print("/n/n/n/n/n LINE________________________________________")
    #
    #
    # Parallel(n_jobs=num_cores)(delayed(parallel)(lake) for lake in range(0, len(lakes_data)))  # series[i-1]
    remove_empty_folders(output_final)


    # for model in [2]:
    #     for scenario in [2]:
    #         print("model :",model, "scenario :",scenario)
    #         summary_characteristics_lake_parallel(modelid=model, scenarioid=scenario, lakes_listcsv=r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList.csv")
    for model in [5]:
        for scenario in [1,2,3,4,5,6,7,8]:
            print("model :",model, "scenario :",scenario)
            summary_characteristics_lake_parallel(modelid=model, scenarioid=scenario,lakes_listcsv=r"C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017SwedenList.csv")
    remove_empty_folders(output_final)
    #
    # for row in range(0,len(lakes_data)):
    #
