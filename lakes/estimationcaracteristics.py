#!/usr/bin/env python3
"""
    File name: estimationcharacteristics.py
    Author: Mariane Cote
    Date created: 20/03/2021
    Python Version: 3.6

"""
from math import sqrt, pi,log10
import numpy as np
import pandas as pd
import os
from lake_information import final_equation_parameters, Area_base_i

skype_data_path = r"C:\Users\macot620\Documents\GitHub\Fish_niche\lake_subset_SHMI_data\SHYPEdata"



def estimated_missing_characteristics(lakes_list):
    """
    The function calculates the characteristics used for estimated the parameters for the 210 lakes.
    :param lakes_list: list of the 211 lakes we want to model.
    :return: None
    """
    data = pd.read_csv(lakes_list, encoding="ISO-8859-1")
    pd.to_numeric(data["area"], downcast='float')
    pd.to_numeric(data["depth"], downcast='float')
    data['Mean'] = np.nan
    data["volume"] = np.nan
    data["Turnover"] = np.nan
    data["sedimentArea"] = np.nan

    data["Subcatchment_Area"] = np.nan
    data["Catchment_Area"] = np.nan
    data["C.L"] = np.nan
    data["SC.L"] = np.nan

    # data['swa_b1'] = np.nan
    # data["swa_b0"] = np.nan
    # data["c_shelter"] = np.nan
    # data["I_scV"] = np.nan
    # data["I_scO"] = np.nan
    # data["K_SOD"] = np.nan
    # data["bod"] = np.nan
    # data["scdoc"] = np.nan


    for lake_pos, max_depth in enumerate(data["depth"]):
        volumeestimated = False
        subid = data["subid"][lake_pos]
        skypefile = pd.ExcelFile(os.path.join(skype_data_path,"%s.xls"%subid))

        catchment_info = pd.read_excel(skypefile, "Områdesinformation")

        allcatchment = list(catchment_info.loc[catchment_info["Modellinformation"] == "Area [km²]:"]["Unnamed: 1"])
        # Two catchment areas are available; the sub-catchment and the entire catchment area. Need to convert the areas from km2 to m2.
        subcatchment = float(allcatchment[0]) * 1000000
        catchment = float(allcatchment[1]) * 1000000

        data["Subcatchment_Area"][lake_pos] = subcatchment
        data["Catchment_Area"][lake_pos] = catchment
        data["C.L"][lake_pos] = catchment / data['area'][lake_pos]
        data["SC.L"][lake_pos] = subcatchment / data['area'][lake_pos]

        lake_info = pd.read_excel(skypefile, "Sjöuppgifter")
        try:
            mean_depth = float(lake_info.loc[lake_info['Sjöuppgifter'] == "Medeldjup [m]:"]["Unnamed: 1"])
        except:
            mean_depth = float(np.nan)
        try:
            max_depth = float(lake_info.loc[lake_info['Sjöuppgifter'] == "Maxdjup [m]:"]["Unnamed: 1"])
        except:
            max_depth = float(np.nan)
        try:
            turnover = float(lake_info.loc[lake_info['Sjöuppgifter'] == "Omsättningstid [år]:"]["Unnamed: 1"])
        except:
            turnover = float(23.35*data["C.L"][lake_pos]**(-1.072))

        try:
            int(mean_depth)
        except:
            volumeestimated = True
            volume = 10 **(1.204 *log10(data["area"][lake_pos]) - 0.629)
            mean_depth = volume/data["area"][lake_pos]

        try:
            int(max_depth)
            if data['depth'][lake_pos] < mean_depth:
                data["depth"][lake_pos] = max_depth
                data["Mean"][lake_pos] = mean_depth
            else:
                max_depth = data["depth"][lake_pos]
                data["Mean"][lake_pos] = mean_depth
        except:
            max_depth = data["depth"][lake_pos]
            if data['depth'][lake_pos] < mean_depth:
                volumeestimated = True
                volume = 10 ** (1.204 * log10(data["area"][lake_pos]) - 0.629)
                mean_depth = volume / data["area"][lake_pos]

                if data['depth'][lake_pos] < mean_depth:
                    data["Mean"][lake_pos] = np.nan
                    volume = np.nan
                else:
                    volume = 10 ** (1.204 * log10(data["area"][lake_pos]) - 0.629)
                    mean_depth = volume / data["area"][lake_pos]
                    data["Mean"][lake_pos] = mean_depth
            else:
                data["Mean"][lake_pos] = mean_depth

        data["Turnover"][lake_pos] = turnover


        try:
            sediment = 0
            if not volumeestimated:
                volume = 0
            sed = []
            lat = []
            basei = []
            basei1 = []
            for i in range(0, int(max_depth)):

                # lateral area of cylinder = 2π * r * h where h = 1 m  and r = (Area_basei / π)**0.5
                area_laterali = 2*pi*(Area_base_i(i, data["area"][lake_pos], data["depth"][lake_pos], data["Mean"][lake_pos])/pi)**0.5
                lat.append(area_laterali)
                # sediment area of cylinder = lateral area + base ring area. ring area = (base area - base area of the layer under this one)
                area_sediment = area_laterali + (Area_base_i(i, data["area"][lake_pos], data["depth"][lake_pos], data["Mean"][lake_pos])-Area_base_i(i+1, data["area"][lake_pos], data["depth"][lake_pos], data["Mean"][lake_pos]))
                sed.append(area_sediment)
                basei.append(Area_base_i(i, data["area"][lake_pos], data["depth"][lake_pos], data["Mean"][lake_pos]))
                basei1.append(Area_base_i(i+1, data["area"][lake_pos], data["depth"][lake_pos], data["Mean"][lake_pos]))
                #the sediment area of the lake is equal to the sum of the sediment area of each layer.
                sediment += area_sediment
                # Since each cylindric layers have a height of 1 meter,
                # the lake's volume is equal to the sum of each cylinder's area
                # (cylinder's volume = base's area * cylinder's height).
                if not volumeestimated:
                    volume += Area_base_i(i, data["area"][lake_pos], data["depth"][lake_pos], data["Mean"][lake_pos])
            data["sedimentArea"][lake_pos] = sediment

            data["volume"][lake_pos] = volume
            print("done")

            # swa_b1,swa_b0,c_shelter,i_sct,i_scv, i_sco,i_scdoc,k_sod,k_bod,kzn0,albice,albsnow = \
            #     final_equation_parameters(data["longitude"][lake_pos], data["latitude"][lake_pos], max_depth,
            #                               mean_depth, data["C.L"][lake_pos], data["SC.L"][lake_pos], turnover,
            #                               data["area"][lake_pos], volume, sediment)
            #
            # data['swab1'][lake_pos] = swa_b1
            # data["swab0"][lake_pos] = swa_b0
            # data["shelter"][lake_pos] = c_shelter
            #
            # data["scv"][lake_pos] = i_scv
            # data["sco"][lake_pos] = i_sco
            # data["sod"][lake_pos] = k_sod
            # data["bod"][lake_pos] = k_bod
            # data["scdoc"][lake_pos] = i_scdoc
        except:
            print("missing variable(s)")



    data.to_csv(lakes_list, index=False, encoding="ISO-8859-1")

if __name__ == "__main__":
    lakes_list = r"2017SwedenList.csv"
    estimated_missing_characteristics(lakes_list)