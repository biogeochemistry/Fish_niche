import os
import netCDF4 as ncdf

# Ouvrir les fichiers
# Extraire la liste de chaque variable
# Copier dans fichier input
# Faire un fichier pour chaque model/scenario

variables = ["hurs", "pr", "ps", "rsds", "sfcWind", "tas"]
models = ["GFDL-ESM2M", "HadGEM2-ES", "IPSL-CM5A-LR", "MIROC5"]
scenarios = ["historical", "piControl", "rcp26", "rcp60"]


def myLake_input(lake_name, forcing_data_directory, output_directory):
    """"""
    for model in models:
        for scenario in scenarios:
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


            print("fichier fini")



if __name__ == "__main__":
    myLake_input("Langtjern","C:/Users/User/Documents/GitHub/Fish_niche/ISIMIP/forcing_data/Langtjern", "C:/Users/User/Documents/GitHub/Fish_niche/ISIMIP/output/NO/Lan")