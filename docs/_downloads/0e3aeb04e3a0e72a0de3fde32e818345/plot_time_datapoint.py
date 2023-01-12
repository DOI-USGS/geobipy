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

################################################################################
# Instantiating a time domain datapoint
# +++++++++++++++++++++++++++++++++++++
#
# In this first example, we define a ground based WalkTEM data point.
#
# Ground time domain data are forward modelled using the `empymod package <https://empymod.readthedocs.io/en/stable/index.html>`_


################################################################################
# Define some time gates

# # Low moment
# lm_off_time = np.array([
#     1.149E-05, 1.350E-05, 1.549E-05, 1.750E-05, 2.000E-05, 2.299E-05,
#     2.649E-05, 3.099E-05, 3.700E-05, 4.450E-05, 5.350E-05, 6.499E-05,
#     7.949E-05, 9.799E-05, 1.215E-04, 1.505E-04, 1.875E-04, 2.340E-04,
#     2.920E-04, 3.655E-04, 4.580E-04, 5.745E-04, 7.210E-04
# ])

# # High moment
# hm_off_time = np.array([
#     9.810e-05, 1.216e-04, 1.506e-04, 1.876e-04, 2.341e-04, 2.921e-04,
#     3.656e-04, 4.581e-04, 5.746e-04, 7.211e-04, 9.056e-04, 1.138e-03,
#     1.431e-03, 1.799e-03, 2.262e-03, 2.846e-03, 3.580e-03, 4.505e-03,
#     5.670e-03, 7.135e-03
# ])

# ################################################################################
# # Define some observed data values for each time gate.
# lm_data = np.array([
#     7.980836E-06, 4.459270E-06, 2.909954E-06, 2.116353E-06, 1.571503E-06,
#     1.205928E-06, 9.537814E-07, 7.538660E-07, 5.879494E-07, 4.572059E-07,
#     3.561824E-07, 2.727531E-07, 2.058368E-07, 1.524225E-07, 1.107586E-07,
#     7.963634E-08, 5.598970E-08, 3.867087E-08, 2.628711E-08, 1.746382E-08,
#     1.136561E-08, 7.234771E-09, 4.503902E-09
# ])

# # High moment
# hm_data = np.array([
#     1.563517e-07, 1.139461e-07, 8.231679e-08, 5.829438e-08, 4.068236e-08,
#     2.804896e-08, 1.899818e-08, 1.268473e-08, 8.347439e-09, 5.420791e-09,
#     3.473876e-09, 2.196246e-09, 1.372012e-09, 8.465165e-10, 5.155328e-10,
#     3.099162e-10, 1.836829e-10, 1.072522e-10, 6.161256e-11, 3.478720e-11
# ])

# ################################################################################
# # Create a Waveform
# #
# # The Waveform class defines a half waveform
# lm_waveform = Waveform(time=np.r_[-1.041E-03, -9.850E-04, 0.000E+00, 4.000E-06],
#                        amplitude=np.r_[0.0, 1.0, 1.0, 0.0],
#                        current=1.0)
# hm_waveform = Waveform(time=np.r_[-8.333E-03, -8.033E-03, 0.000E+00, 5.600E-06],
#                        amplitude=np.r_[0.0, 1.0, 1.0, 0.0],
#                        current=1.0)

# plt.figure()
# lm_waveform.plot(label='Low Moment')
# hm_waveform.plot(label='High Moment', linestyle='-.')
# plt.legend()

# ################################################################################
# # Define the transmitter and reciever loops
# transmitter = SquareLoop(sideLength=40.0)
# receiver = CircularLoop()

# ################################################################################
# # Define two butterworth filters to be applied to the off-time data.
# filters = [butterworth(1, 4.5e5, btype='low'), butterworth(1, 3.e5, btype='low')]

# ################################################################################
# # Create the time domain systems for both moments
# lm_system = TdemSystem(offTimes=lm_off_time,
#                        transmitterLoop=transmitter,
#                        receiverLoop=receiver,
#                        loopOffset=np.r_[0.0, 0.0, 0.0], # Centre loop sounding
#                        waveform=lm_waveform,
#                        offTimeFilters=filters)

# hm_system = TdemSystem(offTimes=hm_off_time,
#                        transmitterLoop=transmitter,
#                        receiverLoop=receiver,
#                        loopOffset=np.r_[0.0, 0.0, 0.0], # Centre loop sounding
#                        waveform=hm_waveform,
#                        offTimeFilters=filters)

# systems = [lm_system, hm_system]

# ################################################################################
# # Instantiate the time domain datapoint
# d = np.hstack([lm_data, hm_data])

# tdp = TdemDataPoint(x=0.0, y=0.0, z=0.0, elevation=0.0,
#                     data=d, std=None, predictedData=None,
#                     system=systems, lineNumber=0.0, fiducial=0.0)

# ###############################################################################
# plt.figure()
# tdp.plot(with_error_bars=False)

# sphinx_gallery_thumbnail_number = 2

# ################################################################################
# # Reading a datapoint from a file
# # +++++++++++++++++++++++++++++++
# # We can read in time domain datapoints from individual datapoint files using the
# # AarhusInv data format.
# tdp = TdemDataPoint()
# tdp.read([dataFolder+"//WalkTEM_LM.txt", dataFolder+"//WalkTEM_HM.txt"])

# ###############################################################################
# plt.figure()
# tdp.plot()

###############################################################################
# Obtaining a datapoint from a dataset
# ++++++++++++++++++++++++++++++++++++
# More often than not, our observed data is stored in a file on disk.
# We can read in a dataset and pull datapoints from it.
#
# For more information about the time domain data set, see :ref:`Time domain dataset`

# The data file name
dataFile=[dataFolder + 'Skytem_High.txt', dataFolder + 'Skytem_Low.txt']
# The EM system file name
systemFile=[dataFolder + 'SkytemHM-SLV.stm', dataFolder + 'SkytemLM-SLV.stm']

################################################################################
# Initialize and read an EM data set
D = TdemData.read_csv(dataFile, systemFile)

################################################################################
# Get a datapoint from the dataset
tdp = D.datapoint(0)

plt.figure()
tdp.plot()

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
_ = tdp.plot_predicted()
plt.tight_layout()
plt.suptitle('new')

################################################################################
plt.figure()
tdp.plotDataResidual(xscale='log')
plt.title('new')

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

# Set values of relative and additive error for both systems.
tdp.relErr = [0.05, 0.05]
tdp.addErr = [1e-11, 1e-12]

# Define the distributions used as priors.
heightPrior = Distribution('Uniform', min=np.float64(tdp.z) - 2.0, max=np.float64(tdp.z) + 2.0)
relativePrior = Distribution('Uniform', min=np.r_[0.01, 0.01], max=np.r_[0.5, 0.5])
additivePrior = Distribution('Uniform', min=np.r_[1e-12, 1e-13], max=np.r_[1e-10, 1e-11], log=True)
tdp.set_priors(height_prior=heightPrior, relative_error_prior=relativePrior, additive_error_prior=additivePrior)

################################################################################
# In order to perturb our solvable parameters, we need to attach proposal distributions
heightProposal = Distribution('Normal', mean=tdp.z, variance = 0.01)
relativeProposal = Distribution('MvNormal', mean=tdp.relErr, variance=2.5e-4)
additiveProposal = Distribution('MvLogNormal', mean=tdp.addErr, variance=2.5e-3, linearSpace=True)
tdp.setProposals(heightProposal, relativeProposal, additiveProposal)

################################################################################
# With priorss set we can auto generate the posteriors
tdp.setPosteriors()

################################################################################
# Perturb the datapoint and record the perturbations
# Note we are not using the priors to accept or reject perturbations.
for i in range(1000):
    tdp.perturb(True, True, True, False)
    tdp.updatePosteriors()

################################################################################
# Plot the posterior distributions
plt.figure()
_ = tdp.z.plotPosteriors()

################################################################################
# plt.figure()
# _ = tdp.errorPosterior[0].comboPlot(cmap='gray_r')


###############################################################################
# Aerotem example
# +++++++++++++++

# The data file name
dataFile=dataFolder + 'aerotem.txt'
# The EM system file name
systemFile=dataFolder + 'aerotem.stm'

################################################################################
# Initialize and read an EM data set
D = TdemData.read_csv(dataFile, systemFile)

################################################################################
# Get a datapoint from the dataset
tdp = D.datapoint(0)
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