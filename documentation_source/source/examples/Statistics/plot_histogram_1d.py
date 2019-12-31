"""
Histogram 1D
------------

This histogram class allows efficient updating of histograms, plotting and
saving as HDF5
"""

#%%
import h5py
from geobipy import hdfRead
from geobipy import StatArray
from geobipy import Histogram1D
import numpy as np
import matplotlib.pyplot as plt

#%%
# Histogram with regular bins
# +++++++++++++++++++++++++++

# Create regularly spaced bins
bins = StatArray(np.linspace(-3, 3, 101), 'Regular bins')

################################################################################

# Set the histogram using the bins, and update
H = Histogram1D(bins = bins)

################################################################################

# We can update the histogram with some new values
x = np.random.randn(1000)
H.update(x, clip=True, trim=True)

# Plot the histogram
plt.figure()
_ = H.plot()

################################################################################
# Get the median, and 95% confidence values
print(H.credibleIntervals(percent=95.0))

################################################################################

# We can write the histogram to a HDF file
with h5py.File('Histogram.h5','w') as hf:
    H.toHdf(hf,'Histogram')

################################################################################

# And read it back in from Hdf5
H1 = hdfRead.readKeyFromFiles('Histogram.h5','/','Histogram')

plt.figure()
_ = H1.plot()


#%%
# Histogram with irregular bins
# +++++++++++++++++++++++++++++

# Create irregularly spaced bins
x = np.cumsum(np.arange(10))
irregularBins = np.hstack([-x[::-1], x[1:]]) 


################################################################################
# Create a named StatArray
edges = StatArray(irregularBins, 'irregular bins')


################################################################################
# Instantiate the histogram with bin edges
H = Histogram1D(bins=edges)


################################################################################
# Generate random numbers
x = (np.random.randn(10000)*20.0) - 10.0


###############################################################################
# Update the histogram
H.update(x)


################################################################################
plt.figure()
_ = H.plot()


################################################################################
# We can plot the histogram as a pcolor plot
# .
plt.figure()
_ = H.pcolor(grid=True, transpose=True)


#%%
# Histogram with linear space entries that are logged internally
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create some bins spaced logarithmically
positiveBins = StatArray(np.logspace(-5, 3), 'positive bins')

################################################################################
print(positiveBins)

################################################################################
# Instantiate the Histogram with log=10
H = Histogram1D(bins=positiveBins, log=10)

################################################################################
# Generate random 10**x numbers
x = 10.0**(np.random.randn(1000)*2.0)

################################################################################
# The update takes in the numbers in linear space and takes their log=10
H.update(x, trim=True)

################################################################################
plt.figure()
_ = H.plot()
