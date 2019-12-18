import pysftp
import os
input_variables = ["hurs",
                   "pr",
                   "ps",
                   "rsds",
                   "sfcWind",
                   "tas"
                    ]
models = [ "GFDL-ESM2M",
          "HadGEM2-ES",
          "IPSL-CM5A-LR",
          "MIROC5"]#,"EWEMBI"
          #]
scenarios = ["historical",
             "piControl",
             "rcp26",
             "rcp60",
             "rcp85"
             ]
lakes=["Allequash", "Annie", "BigMuskellunge", "BlackOak", "Crystal",
                 "CrystalBog", "Delavan","FallingCreek", "Fish", "Great_Pond",
                 "Green_lake", "Laramie", "Mendota", "Monona","Okauchee",
                 "Sammamish", "Sparkling", "Sunapee", "Tahoe", "Toolik",
                 "Trout", "TroutBog", "Two_Sisters","Washington", "Wingra",
                 "Biel", "LowerZurich", "Neuchatel","Alqueva","Annecy",
                 "Bourget", "Geneva","Argyle", "BurleyGriffin", "MtBold",
                 "Dickie", "Eagle_lake", "Harp","Ekoln", "Erken",
                 "EsthwaiteWater", "Windermere","Feeagh","Kilpisjarvi", "Kuivajarvi",
                 "Paajarvi","Kinneret","Kivu","Klicava", "Rimov",
                 "Zlutice","Langtjern","Mozhaysk", "Vendyurskoe","Muggelsee",
                 "Rappbode", "Stechlin","Ngoring","NohipaloMustjarv", "NohipaloValgejarv",
                 "Vortsjarv","Sau","Rotorua", "Tarawera", "Taupo",
                 "Waahi"]
with pysftp.Connection('mistralpp.dkrz.de', username='b380750', password='TwopFaP5') as sftp:
    sftp.cwd("/mnt/lustre01/work/bb0820/ISIMIP/ISIMIP2b/InputData/GCM_atmosphere/biascorrected/local_lakes")
    print(sftp.listdir())
    for lake in lakes:
        for model in models:
            for scenario in scenarios:
                for var in input_variables:
                    print("{}/{}_{}_{}_{}.allTS.nc".format(lake, var, model, scenario, lake))
                    try:
                        sftp.get("{}/{}_{}_{}_{}.allTS.nc".format(lake, var, model, scenario, lake), localpath=r"D:\forcing_data\{}_{}_{}_{}.allTS.nc".format(var, model, scenario, lake))
                        print("end")
                    except:
                        print("{}/{}_{}_{}_{}.allTS.nc dont exist".format(lake, var, model, scenario, lake))

