import netCDF4 as ncdf
import cftime


test = ncdf.Dataset("forcing_data\Langtjern\hurs_GFDL-ESM2M_historical_Langtjern.allTS.nc", "r")
variable = test.variables["time"]
dimension = test.dimensions["time"]
hurs = test.variables["hurs"]

print(variable)
print(hurs)