import csv
import os
import numpy

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





if __name__ == "__main__":
    print(init_info("observations/US_Alq/Allequash_hypsometry.csv", "observations/US_Alq/Allequash_temperature.csv"))
    mylakeinit(init_info("observations/US_Alq/Allequash_hypsometry.csv", "observations/US_Alq/Allequash_temperature.csv"))

