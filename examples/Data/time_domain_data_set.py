"""
Time Domain Data Set
--------------------
"""
################################################################################

from geobipy import customPlots as cP
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from geobipy.src.classes.core.StatArray import StatArray
from geobipy.src.classes.data.dataset.TdemData import TdemData


################################################################################

dataFolder = "..//supplementary//Data//"

# The data file name
dataFiles=[dataFolder + 'Skytem_High.txt', dataFolder + 'Skytem_Low.txt']
# The EM system file name
systemFiles=[dataFolder + 'SkytemHM-SLV.stm', dataFolder + 'SkytemLM-SLV.stm']

################################################################################


TD = TdemData()
TD.read(dataFiles, systemFiles)


################################################################################


TD.iActive


################################################################################


plt.figure()
TD.scatter2D()
plt.show()


################################################################################


TD.times(1)


################################################################################


np.unique(TD.line)


################################################################################


t0=TD.times(0)
plt.figure()
ax1=plt.subplot(221)
TD.getDataPoint(0).plot()
plt.xlabel('')
plt.subplot(222, sharex=ax1)
TD.getDataPoint(50).plot()
plt.xlabel('')
plt.ylabel('')
plt.subplot(223, sharex=ax1)
TD.getDataPoint(100).plot()
plt.title('')
plt.subplot(224, sharex=ax1)
TD.getDataPoint(200).plot()
plt.ylabel('')
plt.title('')
plt.show()


################################################################################


plt.figure()
TD.plotWaveform()
plt.show()

################################################################################


plt.figure()
ax = TD.scatter2D(s=1.0, c=TD.getDataChannel(system=0, channel=23), equalize=True)
plt.axis('equal')
plt.show()


################################################################################


np.nanmax(TD._data[:,16])


################################################################################


TD.iActive


################################################################################


plt.figure()
TD.plot(system=0, channels=TD.iActive[:3], log=10)
plt.show()

################################################################################


plt.figure()
plt.subplot(211)
TD.pcolor(system=0, log=10, xscale='log')
plt.subplot(212)
TD.pcolor(system=1, log=10, xscale='log')
plt.show()

################################################################################


plt.figure()
TD.plotLine(100601.0, log=10)
plt.show()

################################################################################


TD._data


################################################################################


TD.toVTK('TD1', format='binary')


################################################################################


line = TD.getLine(100601.0)


################################################################################


plt.figure()
line.scatter2D(c = line.getDataChannel(10, system=1))
plt.show()

################################################################################


plt.figure()
line.plot(xAxis='x', log=10)
plt.show()