"""
Histogram 2D
------------

"""
from geobipy import StatArray
from geobipy import Histogram2D
import matplotlib.pyplot as plt
import numpy as np


################################################################################


x = StatArray(np.linspace(-4.0, 4.0, 101), 'Variable 1')
y = StatArray(np.linspace(-4.0, 4.0, 101), 'Variable 2')


################################################################################


H = Histogram2D(xBins=x, yBins=y)


################################################################################


a = np.random.randn(1000000)
b = np.random.randn(1000000)
addThese = np.asarray([a, b])


################################################################################


H.update(addThese)


################################################################################


plt.figure()
H.pcolor(cmap='gray_r')


################################################################################


h1 = H.axisHistogram(0)
h2 = H.axisHistogram(1)


################################################################################


plt.figure()
h1.plot()


################################################################################


plt.figure()
H.comboPlot()


################################################################################


plt.figure()
H.pcolor(cmap='gray_r')
H.plotConfidenceIntervals(axis=0)
H.plotConfidenceIntervals(axis=1)


################################################################################


mean = H.axisMean()


################################################################################


plt.figure()
H.pcolor(cmap='gray_r')
H.plotMean()


################################################################################


plt.figure()
H.pcolor(cmap='gray_r')
H.plotMedian()


################################################################################


H.axisPercentage(50.0)


################################################################################


plt.figure(figsize=(9.5, 5))
ax = plt.subplot(121)
H.pcolor(cmap='gray_r', noColorbar=True)
H.plotConfidenceIntervals(axis=0)
H.plotMedian()
H.plotMean(color='y')

plt.subplot(122, sharex=ax, sharey=ax)
H.pcolor(cmap='gray_r', noColorbar=True)
H.plotConfidenceIntervals(axis=1)
H.plotMedian(axis=1)
H.plotMean(axis=1, color='y')


################################################################################


plt.figure(figsize=(9.5, 5))
ax = plt.subplot(121)
H1 = H.intervalMean([-4.0, -2.0, 2.0, 4.0])
H1.pcolor(cmap='gray_r', equalize=True, noColorbar=True)
H1.plotConfidenceIntervals()
plt.subplot(122, sharex=ax, sharey=ax)
H1 = H.intervalMean([-4.0, -2.0, 2.0, 4.0], axis=1)
H1.pcolor(cmap='gray_r', equalize=True, noColorbar=True)
H1.plotConfidenceIntervals(axis=1)


################################################################################


H.confidenceRange()


################################################################################


H.axisTransparency()
