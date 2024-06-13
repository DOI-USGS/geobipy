"""
Tempest dataset
--------------------
"""
#%%
import h5py
from geobipy import plotting as cP
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from geobipy import TempestData

#%%
# Reading in the Data
# +++++++++++++++++++

#%%
dataFolder = "..//..//supplementary//data//"

# # The data file name
# dataFiles = dataFolder + 'Tempest.nc'
# # The EM system file name
# systemFiles = dataFolder + 'Tempest.stm'

# #%%
# # Read in the data from file
# TD = TempestData.read_netcdf(dataFiles, systemFiles)

# TD.write_csv(dataFolder + 'Tempest.csv')
TD = TempestData.read_csv(dataFolder + 'tempest_saline_clay.csv', system_filename=dataFolder + 'Tempest.stm')


#%%
# Plot the locations of the data points
plt.figure(figsize=(8,6))
_ = TD.scatter2D()
plt.title("Scatter plot")

#%%
# Plot all the data along the specified line
plt.figure(figsize=(8,6))
_ = TD.plotLine(0.0)
plt.title('Line {}'.format(225401.0))

#%%
# Or, plot specific channels in the data
plt.figure(figsize=(8,6))
_ = TD.plot_data(system=0, channels=[0, 6, 18])
plt.title("3 channels of data")

#%%
plt.figure()
_ = TD.pcolor(system=0)
plt.title('Data as an array')

#%%
plt.figure()
ax = TD.scatter2D(c=TD.data[:, TD.channel_index(system=0, channel=10)], equalize=True)
plt.axis('equal')
plt.title(f"scatter plot of channel {TD.channel_index(system=0, channel=10)}")

with h5py.File('tdem.h5', 'w') as f:
    TD.createHdf(f, 'tdem')
    TD.writeHdf(f, 'tdem')

with h5py.File('tdem.h5', 'r') as f:
    TD3 = TempestData.fromHdf(f['tdem'])

with h5py.File('tdem.h5', 'r') as f:
    tdp = TempestData.fromHdf(f['tdem'], index=0)


# #%%
# # Obtain a line from the data set
# # +++++++++++++++++++++++++++++++
# line = TD.line(0.0)

# #%%
# plt.figure()
# _ = line.scatter2D()
# plt.title('Channel')

# #%%
# plt.figure()
# _ = line.plot_data(xAxis='index', log=10)
# plt.title("All data along line")

plt.show()
