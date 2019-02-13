#!/usr/bin/env python3
"""
    File name: sedimentArea.py
    Author: Mariane Cote
    Date created: 02/13/2019
    Python Version: 3.6

Calculates the sediment area for each lake in the lake's list and
    adds a column with those values into the CSV file.

"""
from math import sqrt, pi
import pandas as pd


if __name__ == "__main__":
    csvf = r"C:\Users\Marianne\Documents\Fish_niche\lakes\2017SwedenList.csv"
    data = pd.read_csv(csvf, encoding="ISO-8859-1")
    pd.to_numeric(data["area"], downcast='float')
    pd.to_numeric(data["depth"], downcast='float')
    data["sedimentArea"] = (data["area"])/2 * (((data["area"])/(2*pi))**2+data["depth"]**2)**(1/2)
    # equation for lateral surface of cone
    data.to_csv(csvf, index=False, encoding="ISO-8859-1")
