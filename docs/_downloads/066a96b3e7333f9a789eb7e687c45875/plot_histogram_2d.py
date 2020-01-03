"""
Histogram 2D
------------

This 2D histogram class allows efficient updating of histograms, plotting and
saving as HDF5.

"""

#%%
import geobipy
from geobipy import StatArray
from geobipy import Histogram2D
import matplotlib.pyplot as plt
import numpy as np


#%%
# Create some histogram bins in x and y
x = StatArray(np.linspace(-4.0, 4.0, 101), 'Variable 1')
y = StatArray(np.linspace(-4.0, 4.0, 101), 'Variable 2')

################################################################################
# Instantiate
H = Histogram2D(xBins=x, yBins=y)


################################################################################
# Generate some random numbers
a = np.random.randn(1000000)
b = np.random.randn(1000000)
x = np.asarray([a, b])


################################################################################
# Update the histogram counts
H.update(x)


################################################################################
plt.figure()
_ = H.pcolor(cmap='gray_r')


################################################################################
# Generate marginal histograms along an axis
h1 = H.marginalHistogram(axis=0)
h2 = H.marginalHistogram(axis=1)


################################################################################
# Note that the names of the variables are automatically displayed
plt.figure()
plt.subplot(121)
h1.plot()
plt.subplot(122)
_ = h2.plot()


################################################################################
# Create a combination plot with marginal histograms.
# sphinx_gallery_thumbnail_number = 3
plt.figure()
_ = H.comboPlot(cmap='gray_r')


################################################################################
# We can overlay the histogram with its credible intervals
plt.figure()
H.pcolor(cmap='gray_r')
H.plotCredibleIntervals(axis=0, percent=95.0)
_ = H.plotCredibleIntervals(axis=1, percent=95.0)


################################################################################
# Take the mean or median estimates from the histogram
mean = H.mean()
median = H.median()


################################################################################
# Or plot the mean and median
plt.figure()
H.pcolor(cmap='gray_r')
H.plotMean()
H.plotMedian()
plt.legend()

################################################################################
plt.figure(figsize=(9.5, 5))
ax = plt.subplot(121)
H.pcolor(cmap='gray_r', noColorbar=True)
H.plotCredibleIntervals(axis=0)
H.plotMedian()
H.plotMean(color='y')

plt.subplot(122, sharex=ax, sharey=ax)
H.pcolor(cmap='gray_r', noColorbar=True)
H.plotCredibleIntervals(axis=1)
H.plotMedian(axis=1)
H.plotMean(axis=1, color='y')


################################################################################
plt.figure(figsize=(9.5, 5))
ax = plt.subplot(121)
H1 = H.intervalStatistic([-4.0, -2.0, 2.0, 4.0], statistic='mean', axis=0)
H1.pcolor(cmap='gray_r', equalize=True, noColorbar=True)
H1.plotCredibleIntervals(axis=0)
plt.subplot(122, sharex=ax, sharey=ax)
H1 = H.intervalStatistic([-4.0, -2.0, 2.0, 4.0], statistic='mean', axis=1)
H1.pcolor(cmap='gray_r', equalize=True, noColorbar=True)
H1.plotCredibleIntervals(axis=1)


################################################################################
# Get the range between credible intervals
H.credibleRange(percent=95.0)


################################################################################
# We can map the credible range to an opacity or transparency
H.opacity()
H.transparency()
