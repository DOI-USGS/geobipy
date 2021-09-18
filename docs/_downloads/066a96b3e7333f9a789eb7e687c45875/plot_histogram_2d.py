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
x = StatArray(np.linspace(-4.0, 4.0, 100), 'Variable 1')
y = StatArray(np.linspace(-4.0, 4.0, 105), 'Variable 2')

################################################################################
# Instantiate
H = Histogram2D(xEdges=x, yEdges=y)


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
plt.title("2D Histogram")
_ = H.pcolor(cmap='gray_r')


################################################################################
# Generate marginal histograms along an axis
h1 = H.marginalize(axis=0)
h2 = H.marginalize(axis=1)


################################################################################
# Note that the names of the variables are automatically displayed
plt.figure()
plt.suptitle("Marginals along each axis")
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
plt.title("90% credible intervals overlain")
H.pcolor(cmap='gray_r')
H.plotCredibleIntervals(axis=0, percent=95.0)
_ = H.plotCredibleIntervals(axis=1, percent=95.0)


################################################################################
# Take the mean or median estimates from the histogram
mean = H.mean()
median = H.median()

################################################################################
plt.figure(figsize=(9.5, 5))
plt.suptitle("Mean, median, and credible interval overlain")
ax = plt.subplot(121)
H.pcolor(cmap='gray_r', noColorbar=True)
H.plotCredibleIntervals(axis=0)
H.plotMedian()
H.plotMean(color='y')
plt.legend()

plt.subplot(122, sharex=ax, sharey=ax)
H.pcolor(cmap='gray_r', noColorbar=True)
H.plotCredibleIntervals(axis=1)
H.plotMedian(axis=1)
H.plotMean(axis=1, color='y')
plt.legend()


################################################################################
# Get the range between credible intervals
H.credibleRange(percent=95.0)


################################################################################
# We can map the credible range to an opacity or transparency
H.opacity()
H.transparency()


import h5py
with h5py.File('h2d.h5', 'w') as f:
    H.toHdf(f, 'h2d')

with h5py.File('h2d.h5', 'r') as f:
    H1 = Histogram2D.fromHdf(f['h2d'])


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

plt.show()
