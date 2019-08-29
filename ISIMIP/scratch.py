import pysftp
import os
from scipy.optimize import differential_evolution

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

bounds = [(0.1, 10), (0.1, 10), (0.1, 10)]

def func(params):
    x, y, z = params
    return x*2/y + z**3/x + y



lam = lambda params: func(params)
res = differential_evolution(lam, bounds, tol= 10, disp= True)
print(res)

print(res.get('x'[:]))
print(list(res.get('x'[0]))[0])
print(tuple(res.get('x'[0])))
print(res.get('fun'))