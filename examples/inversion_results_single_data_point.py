"""
Single Data Point Inversion
---------------------------
"""

from geobipy import LineResults
import matplotlib.pyplot as plt
from os.path import join
import numpy as np

################################################################################
# Results for a single data point

sysPath = join('supplementary', 'Data')
fName = join('supplementary','run_TimeDomain','30010.0.h5')
LR = LineResults(fName, sysPath=sysPath)


################################################################################


LR.iDs


################################################################################


R = LR.getResults(index=0)


################################################################################


R.plot(forcePlot=True)


################################################################################


plt.figure()
R._plotAcceptanceVsIteration()


################################################################################


plt.figure()
R._plotMisfitVsIteration()


################################################################################


plt.figure()
R._plotElevationPosterior()


################################################################################


plt.figure()
R._plotNumberOfLayersPosterior()


################################################################################


plt.figure()
R._plotObservedPredictedData()


################################################################################


plt.figure()
R._plotHitmapPosterior()


################################################################################


plt.figure()
plt.subplot(211)
R._plotRelativeErrorPosterior()
plt.subplot(212)
R._plotAdditiveErrorPosterior()


################################################################################


plt.figure()
R._plotLayerDepthPosterior()


################################################################################


plt.figure()
R.DzHist.plot()
