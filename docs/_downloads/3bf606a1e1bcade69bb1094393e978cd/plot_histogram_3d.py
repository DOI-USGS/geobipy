"""
Histogram 3D
------------

This 3D histogram class allows efficient updating of histograms, plotting and
saving as HDF5.

"""

#%%
import geobipy
from geobipy import StatArray
from geobipy import Histogram3D
import matplotlib.pyplot as plt
import numpy as np


#%%
# Create some histogram bins in x and y
x = StatArray(np.linspace(-4.0, 4.0, 11), 'Variable 1')
y = StatArray(np.linspace(-4.0, 4.0, 21), 'Variable 2')
z = StatArray(np.linspace(-4.0, 4.0, 31), 'Variable 3')

################################################################################
# Instantiate
H = Histogram3D(xEdges=x, yEdges=y, zEdges=z)


################################################################################
# Generate some random numbers
a = np.random.randn(100000)
b = np.random.randn(100000)
c = np.random.randn(100000)
x = np.asarray([a, b, c])


################################################################################
# Update the histogram counts
H.update(x)


# ################################################################################
# plt.figure()
# _ = H.pcolor(cmap='gray_r')


################################################################################
# Generate marginal histograms along an axis
plt.figure()
plt.suptitle("Marginals along each axis")
for axis in range(3):
    plt.subplot(1, 3, axis+1)
    _ = H.marginalize(axis=axis).plot()


# ################################################################################
# # Take the mean estimate from the histogram
# plt.figure()
# plt.suptitle("Mean along each axis")
# for axis in range(3):
#     plt.subplot(1, 3, axis+1)
#     _ = H.mean(axis=axis).pcolor()

# ################################################################################
# # Take the median estimate from the histogram
# plt.figure()
# plt.suptitle("Median along each axis")
# for axis in range(3):
#     plt.subplot(1, 3, axis+1)
#     _ = H.median(axis=axis).pcolor()

# ################################################################################
# # We can overlay the histogram with its credible intervals
# plt.figure()
# H.pcolor(cmap='gray_r')
# H.plotCredibleIntervals(axis=0, percent=95.0)
# _ = H.plotCredibleIntervals(axis=1, percent=95.0)


# ################################################################################
# # Take the mean or median estimates from the histogram
# mean = H.mean()
# median = H.median()


# ################################################################################
# # Or plot the mean and median
# plt.figure()
# H.pcolor(cmap='gray_r')
# H.plotMean()
# H.plotMedian()
# plt.legend()

# ################################################################################
# plt.figure(figsize=(9.5, 5))
# ax = plt.subplot(121)
# H.pcolor(cmap='gray_r', noColorbar=True)
# H.plotCredibleIntervals(axis=0)
# H.plotMedian()
# H.plotMean(color='y')

# plt.subplot(122, sharex=ax, sharey=ax)
# H.pcolor(cmap='gray_r', noColorbar=True)
# H.plotCredibleIntervals(axis=1)
# H.plotMedian(axis=1)
# H.plotMean(axis=1, color='y')


# ################################################################################
# plt.figure(figsize=(9.5, 5))
# ax = plt.subplot(121)
# H1 = H.intervalStatistic([-4.0, -2.0, 2.0, 4.0], statistic='mean', axis=0)
# H1.pcolor(cmap='gray_r', equalize=True, noColorbar=True)
# H1.plotCredibleIntervals(axis=0)
# plt.subplot(122, sharex=ax, sharey=ax)
# H1 = H.intervalStatistic([-4.0, -2.0, 2.0, 4.0], statistic='mean', axis=1)
# H1.pcolor(cmap='gray_r', equalize=True, noColorbar=True)
# H1.plotCredibleIntervals(axis=1)


# ################################################################################
# # Get the range between credible intervals
# H.credibleRange(percent=95.0)


# ################################################################################
# # We can map the credible range to an opacity or transparency
# H.opacity()
# H.transparency()

plt.show()

# ################################################################################
# # We can plot the mesh in 3D!
# pv_mesh  = H.plot_pyvista(linewidth=0.5)
# pv_mesh.plot(show_edges=True, show_grid=True)
# %%
