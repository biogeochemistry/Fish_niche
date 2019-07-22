import pysftp
from netCDF4 import Dataset

with pysftp.Connection('mistralpp.dkrz.de', username='b380750', password='TwopFaP5') as sftp:
    print("Connected")
    sftp.cwd("/mnt/lustre01/work/bb0820/ISIMIP/ISIMIP2b/InputData/GCM_atmosphere/biascorrected/local_lakes")
    print(sftp.getcwd())
    print(sftp.listdir())
    #data = Dataset('Langtjern/hurs_GFDL-ESM2M_EWEMBI-ISIMIP3BASD_rcp60_Langtjern.allTS.nc', "r", format = "NETCDF4")

