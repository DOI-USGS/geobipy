"""
Histogram 1D
------------

This histogram class allows efficient updating of histograms, plotting and
saving as HDF5
"""

################################################################################

import h5py
from geobipy import hdfRead
from geobipy import StatArray
from geobipy import Histogram1D
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# Histogram with regular bins

x = np.random.randn(1000)
bins = StatArray(np.linspace(-4,4,101), 'Regular bins')



################################################################################
# Set the histogram using the bins, and update

H = Histogram1D(bins = bins)
H.update(x)


################################################################################
# Plot the histogram

plt.figure()
H.plot()


################################################################################
# We can clip additions to the histogram using clip=True. In this case outliers
# will land in the outermost bins.

x = np.full(100, 1000.0)
H.update(x, trim=True)


################################################################################


plt.figure()
H.plot()


################################################################################
# We can write/read the histogram to/from a HDF file

with h5py.File('Histogram.h5','w') as hf:
    H.toHdf(hf,'Histogram')


################################################################################


H1 = hdfRead.readKeyFromFiles('Histogram.h5','/','Histogram')


################################################################################


plt.figure()
H1.plot()


################################################################################
# Histogram with irregular bins

x = np.cumsum(np.arange(10))
irregularBins = np.hstack([-x[::-1], x[1:]])


################################################################################


edges = StatArray(irregularBins, 'irregular bins')


################################################################################


H = Histogram1D(bins=edges)


################################################################################


H.binCentres


################################################################################


H.bins


################################################################################


addThese = (np.random.randn(10000)*20.0) - 10.0


################################################################################


H.update(addThese, trim=False)


################################################################################


plt.figure()
H.plot()


################################################################################


plt.figure()
H.pcolor(grid=True)


################################################################################
# Histogram with linear space entries that are logged internally

positiveBins = StatArray(np.logspace(-5, 3), 'positive bins')


################################################################################


positiveBins


################################################################################


H = Histogram1D(bins=positiveBins, log='e')


################################################################################
# Generate random 10**x

addThese = 10.0**(np.random.randn(1000)*2.0)


################################################################################


H.update(addThese)


################################################################################


plt.figure()
H.plot()
