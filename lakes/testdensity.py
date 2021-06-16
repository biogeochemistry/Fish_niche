import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("2017SwedenList.csv",encoding='ISO-8859-1')
x=data['area']*1e-6 #KM2

sns.distplot(x,hist=True,kde=True,norm_hist=True, kde_kws={"shade": True})

weights = np.ones_like(np.array(x))/float(len(np.array(x)))
plt.hist(x,weights=weights, bins = 100)
#sns.kdeplot(data['area'], shade=True)

plt.savefig("densityrmseA.png")
plt.close()
