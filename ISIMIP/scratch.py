import netCDF4 as ncdf

test = ncdf.Dataset("forcing_data\Langtjern\hurs_GFDL-ESM2M_historical_Langtjern.allTS.nc", "r")
variable = test.variables["time"]

print(variable[:])