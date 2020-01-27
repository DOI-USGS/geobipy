"""
Time Domain Data Set
--------------------
"""

#%%
from geobipy import customPlots as cP
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from geobipy.src.classes.core.StatArray import StatArray
from geobipy.src.classes.data.dataset.TdemData import TdemData

#%%
# Reading in the Data
# +++++++++++++++++++

################################################################################
dataFolder = "..//supplementary//Data//"
# The data file name
dataFiles=[dataFolder + 'Skytem_High.txt', dataFolder + 'Skytem_Low.txt']
# The EM system file name
systemFiles=[dataFolder + 'SkytemHM-SLV.stm', dataFolder + 'SkytemLM-SLV.stm']

################################################################################
# Read in the data from file
TD = TdemData()
TD.read(dataFiles, systemFiles)


################################################################################
# Plot the locations of the data points
plt.figure(figsize=(8,6))
_ = TD.scatter2D()


################################################################################
# Plot all the data along the specified line
plt.figure(figsize=(8,6))
_ = TD.plotLine(100101.0, log=10)

################################################################################
# Or, plot specific channels in the data
plt.figure(figsize=(8,6))
_ = TD.plot(system=0, channels=TD.active[:3], log=10)

################################################################################
plt.figure()
plt.subplot(211)
_ = TD.pcolor(system=0, log=10, xscale='log')
plt.subplot(212)
_ = TD.pcolor(system=1, log=10, xscale='log')

################################################################################
plt.figure()
ax = TD.scatter2D(s=1.0, c=TD.dataChannel(system=0, channel=23), equalize=True)
plt.axis('equal')

################################################################################
# TD.toVTK('TD1', format='binary')


#%%
# Obtain a line from the data set
# +++++++++++++++++++++++++++++++
line = TD.line(100601.0)

################################################################################
plt.figure()
_ = line.scatter2D(c = line.dataChannel(10, system=1))

################################################################################
plt.figure()
_ = line.plot(xAxis='x', log=10)