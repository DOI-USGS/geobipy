"""
Time Domain Datapoint Class
---------------------------
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
from geobipy import TdemData
from geobipy import TdemDataPoint
from geobipy import Model1D
from geobipy import StatArray
from geobipy import Distribution

dataFolder = "..//supplementary//Data//"
# dataFolder = "source//examples//supplementary//Data"

###############################################################################
# Aerotem example
# +++++++++++++++

# The data file name
dataFile = dataFolder + 'aerotem.txt'
# The EM system file name
systemFile = dataFolder + 'aerotem.stm'

################################################################################
# Initialize and read an EM data set
# Prepare the dataset so that we can read a point at a time.
Dataset = TdemData._initialize_sequential_reading(dataFile, systemFile)
# Get a datapoint from the file.
tdp = Dataset._read_record()

# ################################################################################
# # Initialize and read an EM data set
# D = TdemData.read_csv(dataFile, systemFile)

################################################################################
# Get a datapoint from the dataset
plt.figure()
tdp.plot()

################################################################################
# Using a time domain datapoint
# +++++++++++++++++++++++++++++

################################################################################
# We can define a 1D layered earth model, and use it to predict some data
par = StatArray(np.r_[500.0, 20.0], "Conductivity", "$\frac{S}{m}$")
mod = Model1D(edges=np.r_[0.0, 75.0, np.inf], parameters=par)

################################################################################
# Forward model the data
tdp.forward(mod)

################################################################################
plt.figure()
plt.subplot(121)
_ = mod.pcolor()
plt.subplot(122)
_ = tdp.plot()
_ = tdp.plot_predicted()
plt.tight_layout()

################################################################################
# Compute the sensitivity matrix for a given model
J = tdp.sensitivity(mod)
plt.figure()
_ = np.abs(J).pcolor(equalize=True, log=10, flipY=True)

################################################################################
# Attaching statistical descriptors to the datapoint
# ++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Set values of relative and additive error for both systems.
tdp.relErr = 0.05
tdp.addErr = 1e-8
# Define a multivariate log normal distribution as the prior on the predicted data.
tdp.predictedData.prior = Distribution('MvLogNormal', tdp.data[tdp.active], tdp.std[tdp.active]**2.0)

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
_ = tdp.plot_predicted()

################################################################################
# Compute the misfit between observed and predicted data
print(tdp.dataMisfit())

################################################################################
# Plot the misfits for a range of half space conductivities
plt.figure()
_ = tdp.plotHalfSpaceResponses(-6.0, 4.0, 200)
plt.title("Halfspace responses")

################################################################################
# We can attach priors to the height of the datapoint,
# the relative error multiplier, and the additive error noise floor

# Define the distributions used as priors.
heightPrior = Distribution('Uniform', min=np.float64(tdp.z) - 2.0, max=np.float64(tdp.z) + 2.0)
relativePrior = Distribution('Uniform', min=0.01, max=0.5)
additivePrior = Distribution('Uniform', min=1e-8, max=1e-5, log=True)
tdp.set_priors(height_prior=heightPrior, relative_error_prior=relativePrior, additive_error_prior=additivePrior)

################################################################################
# In order to perturb our solvable parameters, we need to attach proposal distributions
heightProposal = Distribution('Normal', mean=tdp.z, variance = 0.01)
relativeProposal = Distribution('MvNormal', mean=tdp.relErr, variance=2.5e-4)
additiveProposal = Distribution('MvLogNormal', mean=tdp.addErr, variance=2.5e-3, linearSpace=True)
tdp.set_proposals(heightProposal, relativeProposal, additiveProposal)

################################################################################
# With priorss set we can auto generate the posteriors
tdp.set_posteriors()

################################################################################
# Perturb the datapoint and record the perturbations
# Note we are not using the priors to accept or reject perturbations.
for i in range(10):
    tdp.perturb()
    tdp.updatePosteriors()

################################################################################
# Plot the posterior distributions
fig = plt.figure()
ax = tdp.init_posterior_plots(fig)
fig.tight_layout()

tdp.plot_posteriors(axes=ax, best=tdp)


plt.show()

#%%
# File Format for a time domain datapoint
# +++++++++++++++++++++++++++++++++++++++
# Here we describe the file format for a time domain datapoint.
#
# For individual datapoints we are using the AarhusInv data format.
#
# Here we take the description for the AarhusInv TEM data file, modified to reflect what we can
# currently handle in GeoBIPy.
#
# Line 1 :: string
#   User-defined label describing the TEM datapoint.
#   This line must contain the following, separated by semicolons.
#   XUTM=
#   YUTM=
#   Elevation=
#   StationNumber=
#   LineNumber=
#   Current=
#
# Line 2 :: first integer, sourceType
#   7 = Rectangular loop source parallel to the x - y plane
# Line 2 :: second integer, polarization
#   3 = Vertical magnetic field
#
# Line 3 :: 6 floats, transmitter and receiver offsets relative to X/Y UTM location.
#   If sourceType = 7, Position of the center loop sounding.
#
# Line 4 :: Transmitter loop dimensions
#   If sourceType = 7, 2 floats.  Loop side length in the x and y directions
#
# Line 5 :: Fixed
#   3 3 3
#
# Line 6 :: first integer, transmitter waveform type. Fixed
#   3 = User defined waveform.
#
# Line 6 :: second integer, number of transmitter waveforms. Fixed
#   1
#
# Line 7 :: transmitter waveform definition
#   A user-defined waveform with piecewise linear segments.
#   A full transmitter waveform definition consists of a number of linear segments
#   This line contains an integer as the first entry, which specifies the number of
#   segments, followed by each segment with 4 floats each. The 4 floats per segment
#   are the start and end times, and start and end amplitudes of the waveform. e.g.
#   3  -8.333e-03 -8.033e-03 0.0 1.0 -8.033e-03 0.0 1.0 1.0 0.0 5.4e-06 1.0 0.0
#
# Line 8 :: On time information. Not used but needs specifying.
#   1 1 1
#
# Line 9 :: On time low-pass filters.  Not used but need specifying.
#   0
#
# Line 10 :: On time high-pass filters. Not used but need specifying.
#   0
#
# Line 11 :: Front-gate time. Not used but need specifying.
#   0.0
#
# Line 12 :: first integer, Number of off time filters
#   Number of filters
#
# Line 12 :: second integer, Order of the butterworth filter
#   1 or 2
#
# Line 12 :: cutoff frequencies Hz, one per the number of filters
#   e.g. 4.5e5
#
# Line 13 :: Off time high pass filters.
#   See Line 12
#
# Lines after 13 contain 3 columns that pertain to
# Measurement Time, Data Value, Estimated Standard Deviation
#
# Example data files are contained in
# `the supplementary folder`_ in this repository
#
# .. _the supplementary folder: https://github.com/usgs/geobipy/tree/master/documentation_source/source/examples/supplementary/Data