

import h5py
from geobipy import plotting as cP
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from geobipy import TempestData, TdemData

#%%
# Reading in the Data
# +++++++++++++++++++

#%%
dataFolder = "..//data//data//"

skytem = TdemData.read_csv(dataFolder + 'skytem_saline_clay.csv', system=[dataFolder + 'SkytemHM.stm', dataFolder + 'SkytemLM.stm'])

skytem.predicted_primary_field = skytem.primary_field
skytem.predicted_secondary_field = skytem.secondary_field

plt.figure()
skytem.plot_data(x='x', system=1)

tempest = TempestData.read_csv(dataFolder + 'tempest_saline_clay.csv', system=dataFolder + 'Tempest.stm')

tempest.predicted_primary_field = tempest.primary_field
tempest.predicted_secondary_field = tempest.secondary_field

plt.figure()
tempest.plot_data(x='x')

sp = skytem.datapoint(index=0)
tp = tempest.datapoint(index=0)

plt.figure()
sp.plot()
plt.figure()
tp.plot()

plt.show()

