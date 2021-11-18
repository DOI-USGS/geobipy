"""
Tempest dataset
--------------------
"""
#%%
from geobipy import plotting as cP
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from geobipy.src.classes.core.StatArray import StatArray
from geobipy.src.classes.data.dataset.TempestData import TempestData

#%%
# Reading in the Data
# +++++++++++++++++++

################################################################################
dataFolder = "..//supplementary//Data//"

# # The data file name
# dataFiles = dataFolder + 'Tempest.nc'
# # The EM system file name
# systemFiles = dataFolder + 'Tempest.stm'

# ################################################################################
# # Read in the data from file
# TD = TempestData.read_netcdf(dataFiles, systemFiles)

# TD.write_csv(dataFolder + 'Tempest.csv')
TD = TempestData.read_csv(dataFolder + 'Tempest.csv', system_filename=dataFolder + 'Tempest.stm')

################################################################################
# Plot the locations of the data points
plt.figure(figsize=(8,6))
_ = TD.scatter2D(equalize=True)
plt.title("Scatter plot")

################################################################################
# Plot all the data along the specified line
plt.figure(figsize=(8,6))
_ = TD.plotLine(0.0)
plt.title('Line {}'.format(225401.0))

################################################################################
# Or, plot specific channels in the data
plt.figure(figsize=(8,6))
_ = TD.plot(system=0, channels=[0, 6, 18])
plt.title("3 channels of data")

################################################################################
plt.figure()
_ = TD.pcolor(system=0)
plt.title('Data as an array')

################################################################################
plt.figure()
ax = TD.scatter2D(s=1.0, c=TD.data[:, TD.channel_index(system=0, channel=10)], equalize=True)
plt.axis('equal')
plt.title("scatter plot of specific channel")

################################################################################
# TD.toVTK('TD1', format='binary')


#%%
# Obtain a line from the data set
# +++++++++++++++++++++++++++++++
line = TD.line(0.0)

################################################################################
plt.figure()
_ = line.scatter2D()
plt.title('Channel')

################################################################################
plt.figure()
_ = line.plot(xAxis='index', log=10)
plt.title("All data along line")

plt.show()
