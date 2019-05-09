import csv
import os
import numpy
import netCDF4 as ncdf

# Pour un lac
# Doit appeler les autres scripts pour créer les fichiers
# Le outpath est déterminé par mylake_init et doit ensuite être passé aux scripts suivants
#

""" Main script for MyLake - ISIMIP
Calls the init, input and par scripts to create the appropriate files for MyLake model
Then launches MyLake for the specified lake
"""

variables = ["hurs", "pr", "ps", "rsds", "sfcWind", "tas"]
models = ["GFDL-ESM2M", "HadGEM2-ES", "IPSL-CM5A-LR", "MIROC5"]
scenarios = ["historical", "piControl", "rcp26", "rcp60"]


def myLake_input(lake_name, forcing_data_directory, output_directory):
    """
    Creates input files for myLake model from forcing data. Forcing data is assumed to be in netCDF format. Variables,
    models and scenarios can be changed as needed. The naming scheme of forcing data files is assumed to be the standard
    for ISIMIP. No return value.

    :param lake_name: Type string. The name of the lake for which the input files are being prepared.
    :param forcing_data_directory: Type string. The folder containing the netCDF files for forcing data for a single lake.
                                    Assumes that all files are in the same directory without any sub-folders.
    :param output_directory: Type string. In a typical run, this is the return value of mylakeinit function.
    :return: No value
    """
    for model in models:
        for scenario in scenarios:
            print("Outputing {}_{}_{}_input".format(lake_name[:3], model, scenario))

            list_dict = {"Year": [], "Month": [], "Day": [], "hurs": [], "pr": [], "ps": [], "rsds": [], "sfcWind": [], "tas": []}

            with open(os.path.join(output_directory, "{}_{}_{}_input".format(lake_name[:3], model, scenario)), "w") as input_file:

                input_file.writelines(["-999\tISIMIP input tests\n", "Year\tMonth\tDay\tGlobal radiation\tCloud cover\t"
                                        "Air temperature\tRelative humidity\tAir pressure\tWind speed\tPrecipitation\t"
                                        "Inflow_V\tInflow_T\tInflow_PT\tInflow_PST\tInflow_DP\tInflow_C\tInflow_PP\n"])

                for variable in variables:
                    ncdf_file = ncdf.Dataset(forcing_data_directory + "/{}_{}_{}_{}.allTS.nc".format(variable, model, scenario, lake_name), "r", format = "NETCDF4")


                    for x in ncdf_file.variables[variable][:]:
                        list_dict[variable].append(float(x))

                    if variable is variables[0]:
                        for y in ncdf_file.variables["time"][:]:
                            list_dict["Year"].append(str(ncdf.num2date(y, "days since 1900-01-01"))[0:4])
                            list_dict["Month"].append(str(ncdf.num2date(y, "days since 1900-01-01"))[5:7])
                            list_dict["Day"].append(str(ncdf.num2date(y, "days since 1900-01-01"))[8:10])

                    ncdf_file.close()

                input_file.write("\n".join(["\t".join(["%s" % year, "%s" % month, "%s" % day, "%f" % rsds,
                                            "0", "%f" % tas, "%f" % hurs, "%f" % ps, "%f" % sfcwind, "%f" % pr,
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

def init_info(hypsometry_path, temperature_path):
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

    with open("{}".format(hypsometry_path), "r") as obs:
        reader = list(csv.reader(obs))[1:]
        out_dir, out_folder = reader[0][0][:2], reader[0][0][3:]

        depth_levels = []
        areas = []
        for row in reader:
            depth_levels.append(float(row[2]))
            areas.append(float(row[3]))

    with open("{}".format(temperature_path), "r") as obs:
        reader = list(csv.reader(obs))[1:]

        w_temp = find_init_temp(reader, depth_levels)



    outdir = os.path.join("output", "{}".format(out_dir), "{}".format(out_folder))

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, "{}_init".format(out_folder))

    return {"depth_levels": depth_levels, "areas": areas, "w_temp": w_temp, "outdir": outdir, "outpath": outpath}


def find_init_temp(observations, depth_levels, date_init = 601):
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
    try:
        if len(observations) == 0:
            print("Date not found, using dummy temperatures")
            return list("4"*len(depth_levels))

        if float(observations[0][2][4:]) < date_init:
            return find_init_temp(observations[1:], depth_levels, date_init)

        w_temp = []
        m = 0

        for depth in depth_levels:
            if float(observations[m][3]) == depth:
                w_temp.append(float(observations[m][4]))
                m += 1

            else:
                w_temp.append("")

        if "" in w_temp: return missing_temp(w_temp, depth_levels)
        else: return w_temp

    except RecursionError:
        print("Date not found, using dummy temperatures")
        return list("4" * len(depth_levels))


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

def mylakepar(longitude, latitude, lake_name, outdir,swa_b1=0.1,k_BOD=0.01,k_SOD=100,I_scDOC=1):
    """
    Creates MyLake parameter file. If the file LAE_para_all1.txt is present, it will be used to prepare the parameters.
    Otherwise, the string in this function while be used.

    :param longitude: Type int. Longitude coordinate of Mylake in degrees.
    :param latitude: Type int. Latitude coordinate of Mylake in degrees
    :param outpath: Type str. Filename where a file of Mylake parameters will be written. In a typical run, this is the
    return value of mylakeinit function
    :return: None
    """

    if (os.path.isfile ( "LAE_para_all1.txt" )): #this file allows change of the four coefficients, if nothing is given, will uses initial values
        print('using file')
        with open ( "LAE_para_all1.txt", "r" ) as infile:
            out = infile.read () % (latitude, longitude, I_scDOC, swa_b1, k_BOD, k_SOD)

    else:
        out = '''-999	"Mylake parameters"			
    Parameter	Value	Min	Max	Unit
    dz	1.0	0.5	2	m
    Kz_ak	0.007	NaN	NaN	(-)
    Kz_ak_ice	0.003	NaN	NaN	(-)
    Kz_N0	7.00E-05	NaN	NaN	s-2
    C_shelter	NaN	NaN	NaN	(-)
    latitude	%.5f	NaN	NaN	dec.deg
    longitude	%.5f	NaN	NaN	dec.deg
    alb_melt_ice	0.6	NaN	NaN	(-)
    alb_melt_snow	0.9	NaN	NaN	(-)
    PAR_sat	3.00E-05	1.00E-05	1.00E-04	mol m-2 s-1
    f_par	0.89	NaN	NaN	(-)
    beta_chl	0.015	0.005	0.045	m2 mg-1
    lamgbda_I	5	NaN	NaN	m-1
    lambda_s	15	NaN	NaN	m-1
    sed_sld	0.36	NaN	NaN	(m3/m3)
    I_scV 	1.339	NaN	NaN	(-)
    I_scT	1.781	NaN	NaN	deg C
    I_scC	1	NaN	NaN	(-)
    I_scS	1	1.1	1.9	(-)
    I_scTP	1	0.4	0.8	(-)
    I_scDOP	1	NaN	NaN	(-)
    I_scChl	1	NaN	NaN	(-)
    I_scDOC	%s	NaN	NaN	(-)
    swa_b0	0.727	NaN	NaN	m-1
    swa_b1	%s	0.8	1.3	m-1
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
    ''' % (latitude, longitude, I_scDOC, swa_b1, k_BOD, k_SOD)

    outpath = outdir + "/{}_par".format(lake_name[:3])

    with open(outpath, 'w') as f:
        f.write(out)

def get_longitude(lake_name, forcing_data_directory):
    ncdf_file = ncdf.Dataset(
        forcing_data_directory + "/hurs_GFDL-ESM2M_historical_{}.allTS.nc".format(lake_name), "r", format="NETCDF4")

    return ncdf_file.variables["lon"][0]

def get_latitude(lake_name, forcing_data_directory):
    ncdf_file = ncdf.Dataset(
        forcing_data_directory + "/hurs_GFDL-ESM2M_historical_{}.allTS.nc".format(lake_name), "r", format="NETCDF4")

    return ncdf_file.variables["lat"][0]



def generate_data_files(hypsometry_path, temperature_path, lake_name, forcing_data_directory, longitude, latitude):
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
    myLake_input(lake_name, forcing_data_directory, outdir)



def run_myLake(modelid, scenarioid, eh, subid, depth, area, longitude, latitude,k_BOD=0.01,swa_b1=1,k_SOD=100,I_scDOC=1):
    pass

if __name__ == "__main__":

    generate_data_files("observations/NO_Lan/Langtjern_hypsometry.csv", "observations/NO_Lan/Langtjern_temperature.csv", "Langtjern", "forcing_data/Langtjern", get_longitude("Langtjern", "forcing_data/Langtjern"), get_latitude("Langtjern", "forcing_data/Langtjern"))
