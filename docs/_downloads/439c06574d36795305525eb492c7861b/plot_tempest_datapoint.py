"""
Tempest Datapoint Class
-----------------------
"""

#%%
# There are three ways in which to create a time domain datapoint
#
# 1) :ref:`Instantiating a time domain datapoint`
#
# 2) :ref:`Reading a datapoint from a file`
#
# 3) :ref:`Obtaining a datapoint from a dataset`
#
# Once instantiated, see :ref:`Using a time domain datapoint`
#
# Credits:
# We would like to thank Ross Brodie at Geoscience Australia for his airborne time domain forward modeller
# https://github.com/GeoscienceAustralia/ga-aem
#
# For ground-based time domain data, we are using Dieter Werthmuller's python package Empymod
# https://empymod.github.io/
#
# Thanks to Dieter for his help getting Empymod ready for incorporation into GeoBIPy

#%%
from os.path import join
import numpy as np
import h5py
import matplotlib.pyplot as plt
from geobipy import hdfRead
from geobipy import Waveform
from geobipy import SquareLoop, CircularLoop
from geobipy import butterworth
from geobipy import TdemSystem
from geobipy import TempestData
# from geobipy import TemDataPoint
from geobipy import Model1D
from geobipy import StatArray
from geobipy import Distribution

dataFolder = "..//supplementary//Data//"
# dataFolder = "source//examples//supplementary//Data"

###############################################################################
# Obtaining a datapoint from a dataset
# ++++++++++++++++++++++++++++++++++++
# More often than not, our observed data is stored in a file on disk.
# We can read in a dataset and pull datapoints from it.
#
# For more information about the time domain data set, see :ref:`Time domain dataset`

# The data file name
dataFile = dataFolder + 'Tempest.nc'
# The EM system file name
systemFile = dataFolder + 'Tempest.stm'

################################################################################
# Initialize and read an EM data set
D = TempestData.read_netcdf(dataFile, systemFile)

################################################################################
# Get a datapoint from the dataset
tdp = D.datapoint(0)

# plt.figure()
# tdp.plot()


################################################################################
# Using a time domain datapoint
# +++++++++++++++++++++++++++++

################################################################################
# We can define a 1D layered earth model, and use it to predict some data
par = StatArray(np.r_[500.0, 20.0], "Conductivity", "$\frac{S}{m}$")
mod = Model1D(edges=np.r_[0, 75.0, np.inf], parameters=par)

################################################################################
# Forward model the data
tdp.forward(mod)

################################################################################
plt.figure()
plt.subplot(121)
_ = mod.pcolor()
plt.subplot(122)
_ = tdp.plot()
_ = tdp.plotPredicted()
plt.tight_layout()
plt.suptitle('Model and response')

################################################################################
plt.figure()
tdp.plotDataResidual(xscale='log')
plt.title('data residual')

################################################################################
# Compute the sensitivity matrix for a given model
J = tdp.sensitivity(mod)
plt.figure()
_ = np.abs(J).pcolor(equalize=True, log=10, flipY=True)

################################################################################
# Attaching statistical descriptors to the datapoint
# ++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Define a multivariate log normal distribution as the prior on the predicted data.
tdp.predictedData.set_prior('MvLogNormal', tdp.data[tdp.active], tdp.std[tdp.active]**2.0)

################################################################################
# This allows us to evaluate the likelihood of the predicted data
print(tdp.likelihood(log=True))
# Or the misfit
print(tdp.dataMisfit())

################################################################################
# We can perform a quick search for the best fitting half space
halfspace = tdp.find_best_halfspace()
print('Best half space conductivity is {} $S/m$'.format(halfspace.par))
plt.figure()
_ = tdp.plot()
_ = tdp.plotPredicted()









plt.show()