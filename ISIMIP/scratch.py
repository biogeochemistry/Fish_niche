import pysftp
import os

from netCDF4 import Dataset

"""
with pysftp.Connection('mistralpp.dkrz.de', username='b380750', password='TwopFaP5') as sftp:
    print("Connected")
    sftp.cwd("/mnt/lustre01/work/bb0820/ISIMIP/ISIMIP2b/InputData/GCM_atmosphere/biascorrected/local_lakes")
    print(sftp.getcwd())
    print(sftp.listdir())
    lake = 'Langtjern'
    file = 'hurs_GFDL-ESM2M_historical_Langtjern.allTS.nc'

    sftp.get("{}/{}".format(lake, file), localpath="forcing_data\\{}".format(file))

nc = Dataset("forcing_data\\hurs_GFDL-ESM2M_historical_Langtjern.allTS.nc", "r", format="NETCDF4")
print(nc.variables)

nc.close()

os.remove("forcing_data\\hurs_GFDL-ESM2M_historical_Langtjern.allTS.nc")

"""

print(os.listdir("observations/Annie"))