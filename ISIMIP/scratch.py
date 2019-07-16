from netCDF4 import Dataset
file = Dataset("forcing_data/Langtjern/rsds_GFDL-ESM2M_rcp26_Langtjern.allTS.nc", "r", format = "NETCDF4")
#print(file)

#print(file.variables)
joules = 300*86400
Mjoules = joules / 10**6
print("300 * 86400 = {}/10**6 = {}".format(joules,Mjoules))

