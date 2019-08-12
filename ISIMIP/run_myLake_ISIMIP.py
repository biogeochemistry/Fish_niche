import csv
import os
import numpy
import datetime
import netCDF4 as ncdf

# Pour un lac
# Doit appeler les autres scripts pour créer les fichiers
# Le outpath est déterminé par mylake_init et doit ensuite être passé aux scripts suivants
#

""" Main script for MyLake - ISIMIP
Calls the init, input and par scripts to create the appropriate files for MyLake model
Then launches MyLake for the specified lake
"""

#Comment variables, models and scenarios here

variables = ["hurs",
             "pr",
             "ps",
             "rsds",
             "sfcWind",
             "tas"
             ]

def myLake_input(lake_name, model, scenario, forcing_data_directory, output_directory):
    """
    Creates input files for myLake model from forcing data. Forcing data is assumed to be in netCDF format. Variables
    can be changed as needed. The naming scheme of forcing data files is assumed to be the standardfor ISIMIP.
    No return value.

    :param lake_name: Type string. The name of the lake for which the input files are being prepared.
    :param forcing_data_directory: Type string. The folder containing the netCDF files for forcing data for a single lake.
                                    Assumes that all files are in the same directory without any sub-folders.
    :param output_directory: Type string. In a typical run, this is the return value of mylakeinit function.
    :return: No value
    """

    print("Outputing {}_{}_{}_input".format(lake_name[:3], model, scenario))

    list_dict = {"Year": [], "Month": [], "Day": [], "hurs": [], "pr": [], "ps": [], "rsds": [], "sfcWind": [], "tas": []}
    start_time = "2006-01-01"
    if scenario == "historical": start_time = "1861-01-01"
    elif scenario == "piControl": start_time = "1661-01-01"

    with open(os.path.join(output_directory, "{}_{}_{}_input".format(lake_name[:3], model, scenario)), "w") as input_file:

        input_file.writelines(["-999\tMyLake Input\n", "Year\tMonth\tDay\tGlobal radiation (MJ/m2)\tCloud cover(-)\t"
                                "Air temperature (deg C)\tRelative humidity (%)\tAir pressure (hPa)\tWind speed (m/s)\t"
                                "Precipitation (mm/day)\tInflow (m3/day)\tInflow_T (deg C)\tInflow_C\tInflow_S (kg/m3)\t"
                                "Inflow_TP (mg/m3)\tInflow_DOP (mg/m3)\tInflow_Chla (mg/m3)\tInflow_DOC (mg/m3)\t"
                                "DIC\tDO\tNO3\tNH4\tSO4\tFe2\tCa\tpH\tCH4\tFe3\tAl3\tSiO4\tSiO2\tdiatom\n"])

        for variable in variables:
            ncdf_file = ncdf.Dataset(forcing_data_directory + "/{}_{}_{}_{}.allTS.nc".format(variable, model, scenario, lake_name), "r", format = "NETCDF4")


            for x in ncdf_file.variables[variable][:]:
                if variable == "tas":   #converting from Kelvins to Celsius
                    temp = float(x) - 273.15
                    list_dict[variable].append(temp)

                elif variable == "ps":  #converting from Pa to hPa
                    press = float(x)/100
                    list_dict[variable].append(press)

                elif variable == "pr":  #converting from kg/m**2/s to mm/day
                    prec = float(x) * 86400
                    list_dict[variable].append(prec)

                elif variable == "rsds":    #converting from W/m**2 to MJ/m**2
                    rsds = float(x) * 24 * 60 *60 / 1000000
                    list_dict[variable].append(rsds)

                else : list_dict[variable].append(float(x))

            if variable is variables[0]:
                for y in ncdf_file.variables["time"][:]:
                    list_dict["Year"].append(str(ncdf.num2date(y, "days since {}".format(start_time)))[0:4])
                    list_dict["Month"].append(str(ncdf.num2date(y, "days since {}".format(start_time)))[5:7])
                    list_dict["Day"].append(str(ncdf.num2date(y, "days since {}".format(start_time)))[8:10])

            ncdf_file.close()

        input_file.write("\n".join(["\t".join(["%s" % year, "%s" % month, "%s" % day, "%f" % rsds,
                                    "0", "%f" % tas, "%f" % hurs, "%f" % ps, "%f" % sfcwind, "%f" % pr,
                                    "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
                                    "0", "0", "0", "0", "0", "0", "0"])
                                    for year, month, day, hurs, pr, ps, rsds, sfcwind, tas in zip(
                                    list_dict["Year"],
                                    list_dict["Month"],
                                    list_dict["Day"],
                                    list_dict["hurs"],
                                    list_dict["pr"],
                                    list_dict["ps"],
                                    list_dict["rsds"],
                                    list_dict["sfcWind"],
                                    list_dict["tas"])]))

    print("{}_{}_{}_input Done".format(lake_name[:3], model, scenario))



def mylakeinit(init_info_dict, I_scDOC = 1):
    """
    For ISI-MIP
    Creates the init file. Uses a dictionary to find the values for each parameters.
    :param init_info_dict: Type dict. The dictionary obtained from init_info
    :param I_scDOC: A REMPLIR
    """
    lines = [
        '\t'.join(
            [('%.2f' % d), ('%.0f' % a), ('%.f' % w_t)] + ['0'] * 5 + ['%s' % (2000 * I_scDOC)] + ['0'] * 5 + ['12000']
            + ['0'] * 15)  # MC 06-01-2018 add I_scDOC and initial 8000 become 2000#MC 06-29-2018 12000
        # Z, Az and T, ...., DOC, .... DO, ...
        for d, a, w_t in zip(init_info_dict["depth_levels"], init_info_dict["areas"], init_info_dict["w_temp"])]

    # lines[0] = lines[0] + '\t0\t0'  # snow and ice, plus 16 dummies
    firstlines = '''-999	"MyLake init"
    Z (m)	Az (m2)	Tz (deg C)	Cz	Sz (kg/m3)	TPz (mg/m3)	DOPz (mg/m3)	Chlaz (mg/m3)	DOCz (mg/m3)	TPz_sed (mg/m3)	
    Chlaz_sed (mg/m3)	"Fvol_IM (m3/m3	 dry w.)"	Hice (m)	Hsnow (m)	DO	dummy	dummy	dummy	dummy	dummy	
    dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy	dummy'''
    lines = [firstlines] + lines
    with open(init_info_dict["outpath"], 'w') as f:
        f.write('\n'.join(lines))

    print("{} Done".format(init_info_dict["outpath"]))

    return init_info_dict["outdir"]                 # To pass the output folder to the other modules

def init_info(lakeName, observation_path, date_init = 101):
    """
    J. Bellavance 2018/11/19
    For ISI-MIP
    Opens hypsomtery and temperature csv files for a lake in the observations directory. Obtains the depth levels,
    observed bathymetric area for each levels and the first observed mean temperature for each level. Also prepares the
    outpath for the init file.

    :param: hypsometry_path : Type string. Path to the hypsometry csv file
    :param: temperature_path : Type string. Path to the temperature csv file

    :return: Type dict. depth_levels, areas, w_temp (mean temperatures) and outhpath as keys, and lists of values as values.
             outhpath has the output directory path as a value instead, as a string.
    """

    with open("{}/{}_hypsometry.csv".format(observation_path, lakeName), "r") as obs:
        reader = list(csv.reader(obs))[1:]
        out_dir, out_folder = reader[0][0][:2], reader[0][0][3:]

        depth_levels = []
        areas = []
        for row in reader:
            depth_levels.append(float(row[2]))
            areas.append(float(row[3]))

    if os.path.exists("{}/{}_temp_daily.csv".format(observation_path, lakeName)):

        with open("{}/{}_temp_daily.csv".format(observation_path, lakeName), "r") as obs:
            reader = list(csv.reader(obs))[1:]

            w_temp = find_init_temp(reader, depth_levels, date_init)

    else:
        reader = []

        for file in os.listdir(observation_path):
            with open(file, "r") as obs:
                reader.append(list(csv.reader(obs))[1:])

        w_temp = find_init_temp(reader, depth_levels, date_init)



    outdir = os.path.join("input", "{}".format(out_dir), "{}".format(out_folder))

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, "{}_init".format(out_folder))

    return {"depth_levels": depth_levels, "areas": areas, "w_temp": w_temp, "outdir": outdir, "outpath": outpath}


def find_init_temp(observations, depth_levels, date_init):
    """
    J. Bellavance 2018/12/18
    For ISI-MIP
    With temperature .csv file opened, searches for the specified date in the time stamp column. Then checks if the data
    set for that date is complete (is there a temperature value for every known depth level for this lake). If not,
    interpolate the missing data (with missing_temp).

    :param observations: Type list. A list made from an opened .csv file.
    :param depth_levels: Type list. The depth levels obtained from the hypsometry file. Depth levels values are floats.
    :param date_init: Type int. Date used to initialise data. Must be in the form of 'MMDD'. Year must not be specified.
    :return: Type list. A complete set of mean temperatures for init files, ordered by depth levels
    """
    if len(observations) == 0:
        print("Date not found, using dummy temperatures")
        return list("4"*len(depth_levels))

    obs_list = []

    for observation in observations:
        if int(observation[2][4:]) < date_init or int(observation[2][4:]) > date_init + 20:
            continue
        elif obs_list == []: obs_list.append(observation)
        elif observation[2][4:] == obs_list[0][2][4:]:
            obs_list.append(observation)

    w_temp = []
    m = 0

    for depth in depth_levels:
        if float(obs_list[m][3]) == depth:
            w_temp.append(float(obs_list[m][4]))
            m += 1

        else:
            w_temp.append("")

    if "" in w_temp: return missing_temp(w_temp, depth_levels)
    else: return w_temp

def missing_temp(temp_list, depth_levels):
    """
    2019/10/01
    Interpolates missing temperatures for find_init_temp
    :param temp_list: Type list. The list of initial temperatures from find_init_temp, with empty strings where
    temperatures are missing.
    :param depth_levels: Type list. The list of depth levels used in find_init_temp.
    :return: Type list. The list of initial temperatures with the interpolated values.
    """
    observed_depths = []

    for depth in depth_levels:
        if temp_list[depth_levels.index(depth)] != "":
            observed_depths.append(depth)

    while "" in temp_list:
        temp_list.remove("")

    for depth in depth_levels:
        if depth in observed_depths: continue

        else:
            if depth < observed_depths[0]:
                temp_list.insert(0, temp_list[0])
                observed_depths.insert(0, depth)
            elif depth > observed_depths[-1]:
                temp_list.append(temp_list[-1])
                observed_depths.append(depth)

            else:
                temp_list.insert(depth_levels.index(depth), numpy.interp(depth, observed_depths, temp_list))
                observed_depths.insert(depth_levels.index(depth), depth)

    return temp_list

def mylakepar(longitude, latitude, lake_name, outdir, kz_N0 = 0.00007, c_shelter = "NaN", alb_melt_ice = 0.3, alb_melt_snow = 0.55, i_scv = 1, i_sct = 0, swa_b0 = 2.5, swa_b1=1,k_BOD=0.01,k_SOD=100,I_scDOC=1):
    """
    Creates MyLake parameter file. If the file LAE_para_all1.txt is present, it will be used to prepare the parameters.
    Otherwise, the string in this function while be used.

    :param longitude: Type int. Longitude coordinate of Mylake in degrees.
    :param latitude: Type int. Latitude coordinate of Mylake in degrees
    :param lake_name: Type str. Name of the Lake.
    :param outdir: Type str. Output folder.
    :param c_shelter: Type str. Wind correction, a fraction between 0 and 1.
    :param alb_melt_ice: Type float. Albedo ice, a fraction between 0 and 1.
    :param alb_melt_snow: Type float. Albedo snow, a fraction between 0 and 1.
    :param i_scv: Type float. Scaler volume inflow, multiplicative.
    :param i_sct: Type float. Scaler temperature inflow, additive.
    :param swa_b0: Type float. Water colour, 200-800 nm.
    :param  swa_b1: Type float. Water colour.

    :return: None
    """

    if (os.path.isfile ( "LAE_para_all1.txt" )): #this file allows change of the four coefficients, if nothing is given, will uses initial values
        print('using file')
        with open ( "LAE_para_all1.txt", "r" ) as infile:
            out = infile.read () % (latitude, longitude, kz_N0, c_shelter, alb_melt_ice, alb_melt_snow, i_scv, i_sct, I_scDOC, swa_b0, swa_b1, k_BOD, k_SOD)

    else:
        out = '''-999	"Mylake parameters"			
    Parameter	Value	Min	Max	Unit
    dz	1.0	0.5	2	m
    Kz_ak	NaN	NaN	NaN	(-)
    Kz_ak_ice	0.0009	NaN	NaN	(-)
    Kz_N0	%f	NaN	NaN	s-2                 #7.00E-05
    C_shelter	%s	NaN	NaN	(-)
    latitude	%.5f	NaN	NaN	dec.deg
    longitude	%.5f	NaN	NaN	dec.deg
    alb_melt_ice	%f	NaN	NaN	(-)
    alb_melt_snow	%f	NaN	NaN	(-)
    PAR_sat	3.00E-05	1.00E-05	1.00E-04	mol m-2 s-1
    f_par	0.89	NaN	NaN	(-)
    beta_chl	0.015	0.005	0.045	m2 mg-1
    lamgbda_I	5	NaN	NaN	m-1
    lambda_s	15	NaN	NaN	m-1
    sed_sld	0.36	NaN	NaN	(m3/m3)
    I_scV 	%f	NaN	NaN	(-)
    I_scT	%f	NaN	NaN	deg C
    I_scC	1	NaN	NaN	(-)
    I_scS	1	1.1	1.9	(-)
    I_scTP	1	0.4	0.8	(-)
    I_scDOP	1	NaN	NaN	(-)
    I_scChl	1	NaN	NaN	(-)
    I_scDOC	%s	NaN	NaN	(-)
    swa_b0	%f	NaN	NaN	m-1
    swa_b1	%f	0.8	1.3	m-1
    S_res_epi	3.30E-07	7.30E-08	1.82E-06	m d-1 (dry mass)
    S_res_hypo	3.30E-08	NaN	NaN	m d-1 (dry mass)
    H_sed	0.03	NaN	NaN	m
    Psat_Lang	2500	NaN	NaN	mg m-3
    Fmax_Lang	8000	5000	10000	mg kg-1
    Uz_Sz	0.3	0.1	1	m d-1
    Uz_Chl	0.16	0.05	0.5	m d-1
    Y_cp	1	NaN	NaN	(-)
    m_twty	0.2	0.1	0.3	d-1
    g_twty	1.5	1	1.5	d-1
    k_sed_twty	2.00E-04	NaN	NaN	d-1
    k_dop_twty	0	NaN	NaN	d-1
    P_half	0.2	0.2	2	mg m-3
    PAR_sat2	3.00E-05	NaN	NaN	mol m-2 s-1
    beta_chl2	0.015	NaN	NaN	m2 mg-1
    Uz_Chl2	0.16	NaN	NaN	m d-1
    m_twty2	0.2	NaN	NaN	d-1
    g_twty2	1.5	NaN	NaN	d-1
    P_half2	0.2	NaN	NaN	mg m-3
    oc_DOC	0.01	NaN	NaN	m2 mg-1
    qy_DOC	0.1	NaN	NaN	mg mol-1
    k_BOD	%s	NaN	NaN	d-1
    k_SOD	%s	NaN	NaN	mg m-2
    theta_BOD	1.047	NaN	NaN	(-)
    theta_BOD_ice	1.13	NaN	NaN	(-)
    theta_SOD	1	NaN	NaN	(-)
    theta_SOD_ice	1	NaN	NaN	(-)
    theta_T	4	NaN	NaN	deg.celcius
    pH	5.2	NaN	NaN	(-)
    I_scDIC	1	NaN	NaN	(-)
    Mass_Ratio_C_Chl	100	NaN	NaN	(-)
    SS_C	0.25	NaN NaN 57
    density_org_H_nc	1.95	NaN NaN 58
    density_inorg_H_nc	2.65	NaN NaN 59
    I_scO	1	NaN NaN (-)
    ''' % (kz_N0, c_shelter, latitude, longitude, alb_melt_ice, alb_melt_snow, i_scv, i_sct, I_scDOC, swa_b0, swa_b1, k_BOD, k_SOD)

    outpath = outdir + "\{}_par".format(lake_name[:3])

    with open(outpath, 'w') as f:
        f.write(out)

    print("{} Done".format(outpath))

def get_longitude(lake_name, forcing_data_directory):
    """
    Obtains longitude from a given ncdf file.
    :param lake_name: string.
    :param forcing_data_directory: string. The directory with the ncdf files.
    :return: float. the longitude of the lake.
    """
    ncdf_file = ncdf.Dataset(
        forcing_data_directory + "/hurs_GFDL-ESM2M_historical_{}.allTS.nc".format(lake_name), "r", format="NETCDF4")

    return ncdf_file.variables["lon"][0]

def get_latitude(lake_name, forcing_data_directory):
    """
    Obtains latitude from a given ncdf file.
    :param lake_name: string.
    :param forcing_data_directory: string. The directory with the ncdf files.
    :return: float. the latitude of the lake.
    """
    ncdf_file = ncdf.Dataset(
        forcing_data_directory + "/hurs_GFDL-ESM2M_historical_{}.allTS.nc".format(lake_name), "r", format="NETCDF4")

    return ncdf_file.variables["lat"][0]



def generate_input_files(hypsometry_path, temperature_path, lake_name, f_lake_name, forcing_data_directory, longitude, latitude, model, scenario):
    """
    Creates all files needed for a run of mylake model with a single lake. The input function will generate ALL needed
    input files(one for each combination of scenario, model and variable)
    :param hypsometry_path: Type string. The path to the hypsometry/bathymetry csv/xls file
    :param temperature_path: Type string. Path to the temperature csv/xls file
    :param lake_name: Type string.
    :param forcing_data_directory: Type string. Path to the forcing data folder.
    :param longitude: Type int.
    :param latitude: Type int.
    :return: None
    """
    outdir = mylakeinit(init_info(hypsometry_path, temperature_path))
    mylakepar(longitude, latitude, lake_name, outdir)
    myLake_input(f_lake_name, model, scenario, forcing_data_directory, outdir)

def simulation_years(scenarioid):
    if scenarioid == 'piControl':
        y1, y2 = 1661, 1860
    elif scenarioid == 'Historical':
        y1, y2 = 1860, 2005
    else:
        y1, y2 = 2006, 2300

    return y1, y2

def run_myLake(observations_path, input_directory, region, lakeName, modelid, scenarioid, flag = None):
    """
    Runs the MyLake simulation using the input, init and parameters files. Makes a single run for a combination of lake,
    model and scenario.

    :param input_directory: string. folder containing all input files.
    :param lakeName: string. Name of the lake to simulate
    :param modelid: string. model used
    :param scenarioid: string. scenario used
    :param flag: None or string, determines if the run is for calibration or simulation. Should be set to None or "calibration".

    :return: None
    """

    init_file = os.path.join(input_directory, "{}_init".format(lakeName[:3]))
    parameter_file = os.path.join(input_directory, "{}_par".format(lakeName[:3]))
    input_file = os.path.join(input_directory, "{}_{}_{}_input".format(lakeName[:3], modelid, scenarioid))
    outfolder = os.path.join("output", region, lakeName, modelid, scenarioid)

    if flag == "calibration":   #Ideally, calibrations are done using 2013 and 2014 as the years. If not possible, the last years of observations are used
        if os.path.exists("{}/{}_temp_daily.csv".format(observations_path, lakeName)):
            with open("{}/{}_temp_daily.csv".format(observations_path, lakeName), "r") as obs:
                reader = list(csv.reader(obs))[1:]

                first_year = int(reader[0][2][:4])
                last_year = int(reader[-1][2][:4]) + 1

                if first_year <= 2013 and last_year >= 2014:
                    y1, y2 = 2013, 2014

                else:
                    y2 = int(reader[-1][2][:4])
                    y1 = y2 - 1

        elif os.path.exists("{}/{}_temp_subdaily_2013.csv".format(observations_path, lakeName)) and os.path.exists(
                "{}/{}_temp_subdaily_2014.csv".format(observations_path, lakeName)): y1, y2 = 2013, 2014

        else:
            file_list = os.listdir(observations_path)
            y2 = 0
            for file in file_list:
                if int(file[len(lakeName) + 15:-5]) > y2: y2 = int(file[len(lakeName) + 14:-4])

            y1 = y2 -1

    else:
        y1, y2 = simulation_years(scenarioid)


    if not os.path.exists ( outfolder ):
        os.makedirs ( outfolder )


    cmd = 'matlab -wait -r -nosplash -nodesktop mylakeGoran(\'%s\',\'%s\',\'%s\',%d,%d,\'%s\');quit' % (init_file, parameter_file, input_file, y1, y2, outfolder)
    print ( cmd )
    os.system ( cmd )

    expectedfs = ['Tzt.csv', 'O2zt.csv', 'Attn_zt.csv', 'Qst.csv', 'DOCzt.csv', 'lambdazt.csv']
    flags = [os.path.exists(os.path.join(outfolder, f)) for f in expectedfs]

    if all(flags):
        with open(os.path.join(outfolder, 'RunComplete'), 'w') as f:
            f.write(datetime.datetime.now().isoformat())
        ret = 0
    ret = 0 if all(flags) else 100
    return ret


if __name__ == "__main__":
    #myLake_input("Langtjern", "GFDL-ESM2M", "historical", "forcing_data/Langtjern", "input\\NO\Lan")
    generate_input_files("Annie", "observations/Annie", "Annie", 'Annie', "forcing_data/Annie", get_longitude('Annie', "forcing_data/Annie"), get_latitude('Annie', "forcing_data/Annie"), "GFDL-ESM2M", "rcp26")
    #mylakepar(9.75000, 60.25000, "Langtjern", "input\\NO\Lan", kz_N0= 1.61132863e-04, c_shelter= "1.79267238e-02", alb_melt_ice= 4.56082677e-01, alb_melt_snow= 4.73366534e-01, swa_b0= 2.00915072, swa_b1= 8.62103358e-01)
    run_myLake("observations\Annie", "input\\US\Ann", "US", "Annie", "GFDL-ESM2M", "rcp26", flag="calibration")