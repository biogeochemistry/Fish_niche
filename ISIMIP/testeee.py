import numpy
y1=2006
y2=2299
if y2 - y1 > 150:
    years = list(range(y1, y2, 150))
    years.append(y2)
    all_files = []
    yrange = range(0, len(years)-1)
    for i in range(0, len(years)-1):
        test = (years[i])
        if i + 1 != len(years) - 1:
            yinit = years[i]
            yend = years[i + 1] - 1

            print(yinit, yend)
        else:
            yinit = years[i]
            yend = years[i + 1]
            print(yinit, yend)