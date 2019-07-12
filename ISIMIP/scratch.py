
with open("input/NO/Lan/Lan_par", "r") as param_file:
    for line in param_file.readlines():

        if "C_shelter" in line:
            print(line.split('\t')[1])

        """if "C_shelter" in line:
            if float(list(line)[1]) <= 0:
        elif "Swa_b0" in line:
            if float(list(line)[1]) <= 0: 
        elif "Swa_b01" in line:
            if float(list(line)[1]) <= 0: 
        elif "I_ScV" in line:
            if float(list(line)[1]) <= 0: 
        elif "I_ScT" in line:
            if float(list(line)[1]) <= 0: 
        elif "Alb_melt_ice" in line:
            if float(list(line)[1]) <= 0: 
        elif "Alb_melt_snow" in line:
            if float(list(line)[1]) <= 0: """