"""
Frequency Domain Line Inversion
-------------------------------

"""
from geobipy import LineResults
from geobipy import FdemData
import matplotlib.pyplot as plt
from os.path import join
import numpy as np

################################################################################

hdfFile = '' #TODO
dataDirectory = #TODO
dataFileName = dataDirectory + "//"
systemFileName = dataDirectory + "//"
LR = LineResults(hdfFile, dataDirectory)
axis = 'y'

################################################################################
LR.getIDs()
print(LR.iDs)

################################################################################

LR.getX()
LR.x


################################################################################


LR.getY()
LR.y


################################################################################


plt.figure()
LR.plotXsection(bestModel=False, log=10, invertPar=True, vmin=1.0, vmax=2.5, useVariance=True, xAxis=axis)
LR.plotDataElevation(linewidth=1, xAxis=axis)
LR.plotElevation(linewidth=1.0, xAxis=axis)


################################################################################


plt.figure()
LR.plotXsection(bestModel=True, log=10, invertPar=True, vmin=1.0, vmax=np.log10(500.0), useVariance=True, xAxis=axis)
LR.plotDataElevation(linewidth=1.0, xAxis=axis)
LR.plotElevation(linewidth=1.0, xAxis=axis)


################################################################################


plt.figure()
LR.plotOpacity(cmap='gray', xAxis=axis)
LR.plotDataElevation(linewidth=1.0, xAxis=axis)
LR.plotElevation(linewidth=1.0, xAxis=axis)


################################################################################


plt.figure()
LR.plotKlayers(axis='y')


################################################################################


plt.figure()
ax1 = plt.subplot(121)
LR.plotRelativeError(marker=None, xAxis=axis)
ax2 = plt.subplot(122, sharey=ax1)
LR.relErr.hist(100, rotate=True)


################################################################################


plt.figure()
ax1 = plt.subplot(121)
LR.plotAdditiveError(linestyle='none')
plt.subplot(122, sharey = ax1)
LR.addErr.hist(100, rotate=True)


################################################################################


plt.figure()
plt.subplot(311)
LR.plotTotalError(channel=6, marker=None, xAxis=axis)
plt.subplot(312)
LR.plotTotalError(channel=7, marker=None, xAxis=axis)
plt.subplot(313)
LR.plotTotalError(channel=8, marker=None, xAxis=axis)


################################################################################


plt.figure()
plt.subplot(131)
LR.pcolorObservedData(log=10)
plt.subplot(132)
LR.pcolorPredictedData(log=10)
plt.subplot(133)
LR.pcolorDataResidual(abs=True, log=10)


################################################################################


plt.figure()
ax = plt.subplot(121)
LR.plotPredictedData(channel=10, xAxis=axis, linewidth=1)
LR.plotObservedData(channel=10, xAxis=axis, linewidth=1)
ax2 = plt.subplot(122, sharex = ax)
LR.plotDataResidual(channel=10, abs=True, xAxis=axis, linewidth=1)


################################################################################


plt.figure()
LR.histogram(nBins=100, log=10, invertPar=False)


################################################################################


plt.figure()
LR.histogram(nBins=100, depth1=25.0, depth2=50.0, log=10, bestModel=False)


################################################################################


plt.figure()
LR.plotInterfaces(useVariance=True, xAxis=axis, cmap='gray_r')
LR.plotElevation(linewidth=0.5, xAxis=axis)
LR.plotDataElevation(linewidth=0.5, xAxis=axis)


################################################################################


plt.figure()
plt.subplot(211)
LR.plotAdditiveErrorDistributions(system=0, cmap='gray_r', trim=True, xAxis=axis)
plt.subplot(212)
LR.plotRelativeErrorDistributions(system=0, cmap='gray_r', trim=True, xAxis=axis)


################################################################################


plt.figure()
LR.plotElevationDistributions(cmap='gray_r', xAxis=axis)


################################################################################


plt.figure()
LR.plotKlayersDistributions(cmap='gray_r', xAxis=axis, trim=True)


################################################################################


plt.figure()
LR.crossplotErrors()




################################################################################
# Exporting to VTK

LR.toVtk('test')

################################################################################
# Summary plot of the line results with location map.

plt.figure()
FD = FdemData()
FD.read(dataFileName, systemFileName)
LR.plotSummary(data=FD, fiducial=33407.0, bestModel=False, log=10, invertPar=True, vmin=1.0, vmax=2.5, useVariance=True, xAxis='y')


################################################################################
# Extracting the results for a single data point.

plt.figure()
LR.plotDataPointResults(33407.0)


################################################################################


R = LR.getResults(fid=33407.0)


################################################################################


plt.figure()
R.kHist.plot()


################################################################################


plt.figure()
R.initFigure(forcePlot=True)
R.plot(forcePlot=True)
