"""
Frequency domain datapoint
--------------------------
"""

#%%
# There are two ways in which to create a frequency domain datapoint,
#
# 1) :ref:`Instantiating a frequency domain data point`
#
# 2) :ref:`Obtaining a datapoint from a dataset`
#
# Once instantiated, see :ref:`Using a frequency domain datapoint`

#%%
from os.path import join
import numpy as np
import h5py
import matplotlib.pyplot as plt
from geobipy import hdfRead
from geobipy import CircularLoop
from geobipy import FdemSystem
from geobipy import FdemData
from geobipy import FdemDataPoint
from geobipy import Model1D
from geobipy import StatArray
from geobipy import Distribution

################################################################################
# Instantiating a frequency domain data point
# +++++++++++++++++++++++++++++++++++++++++++
#
# To instantiate a frequency domain datapoint we need to define some
# characteristics of the acquisition system.
#
# We need to define the frequencies in Hz of the transmitter,
# and the geometery of the loops used for each frequency.

frequencies = np.asarray([380.0, 1776.0, 3345.0, 8171.0, 41020.0, 129550.0])

transmitterLoops = [CircularLoop(orient='z'),     CircularLoop(orient='z'),
                    CircularLoop('x', moment=-1), CircularLoop(orient='z'),
                    CircularLoop(orient='z'),     CircularLoop(orient='z')]

receiverLoops    = [CircularLoop(orient='z', x=7.93),    CircularLoop(orient='z', x=7.91),
                    CircularLoop('x', moment=1, x=9.03), CircularLoop(orient='z', x=7.91),
                    CircularLoop(orient='z', x=7.91),    CircularLoop(orient='z', x=7.89)]

################################################################################
# Now we can instantiate the system.
fds = FdemSystem(frequencies, transmitterLoops, receiverLoops)

################################################################################
# And use the system to instantiate a datapoint
#
# Note the extra arguments that can be used to create the data point.
# data is for any observed data one might have, while std are the estimated standard
# deviations of those observed data.
#
# Define some in-phase then quadrature data for each frequency.
data = np.r_[145.3, 435.8, 260.6, 875.1, 1502.7, 1516.9,
             217.9, 412.5, 178.7, 516.5, 405.7, 255.7]

fdp = FdemDataPoint(x=0.0, y=0.0, z=30.0, elevation=0.0,
                    data=data, std=None, predictedData=None,
                    system=fds, lineNumber=0.0, fiducial=0.0)

# ###############################################################################
# plt.figure()
# _ = fdp.plot()

################################################################################
# Obtaining a datapoint from a dataset
# ++++++++++++++++++++++++++++++++++++
#
# More often than not, our observed data is stored in a file on disk.
# We can read in a dataset and pull datapoints from it.
#
# For more information about the frequency domain data set see :ref:`Frequency domain dataset`

################################################################################
# Set some paths and file names
dataFolder = "..//supplementary//Data//"
# The data file name
dataFile = dataFolder + 'Resolve2.txt'
# The EM system file name
systemFile = dataFolder + 'FdemSystem2.stm'

################################################################################
# Initialize and read an EM data set
# Prepare the dataset so that we can read a point at a time.
Dataset = FdemData._initialize_sequential_reading(dataFile, systemFile)
# Get a datapoint from the file.
fdp = Dataset._read_record()


# ################################################################################
# # Initialize and read an EM data set
# D = FdemData.read_csv(dataFile,systemFile)

# ################################################################################
# # Get a data point from the dataset
# fdp = D.datapoint(0)
# plt.figure()
# _ = fdp.plot()

###############################################################################
# Using a datapoint
# +++++++++++++++++

################################################################################
# We can define a 1D layered earth model, and use it to predict some data
nCells = 19
par = StatArray(np.linspace(0.01, 0.1, nCells), "Conductivity", "$\frac{S}{m}$")
thk = StatArray(np.ones(nCells) * 10.0)
thk[-1] = np.inf
mod = Model1D(nCells = nCells, parameters=par, widths=thk)

################################################################################
# Forward model the data
fdp.forward(mod)

###############################################################################
plt.figure()
plt.subplot(121)
_ = mod.pcolor()
plt.subplot(122)
_ = fdp.plotPredicted()
plt.tight_layout()

################################################################################
# Compute the sensitivity matrix for a given model
J = fdp.sensitivity(mod)
plt.figure()
_ = np.abs(J).pcolor(equalize=True, log=10, flipY=True)

################################################################################
# Attaching statistical descriptors to the datapoint
# ++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Set values of relative and additive error for both systems.
fdp.relErr = 0.05
fdp.addErr = 10.0
# Define a multivariate log normal distribution as the prior on the predicted data.
fdp.predictedData.prior = Distribution('MvLogNormal', fdp.data[fdp.active], fdp.std[fdp.active]**2.0)

################################################################################
# This allows us to evaluate the likelihood of the predicted data
print(fdp.likelihood(log=True))
# Or the misfit
print(fdp.dataMisfit())

################################################################################
# We can perform a quick search for the best fitting half space
# halfspace = fdp.FindBestHalfSpace()
# print('Best half space conductivity is {} $S/m$'.format(halfspace.par))
# plt.figure()
# _ = fdp.plot()
# _ = fdp.plotPredicted()

################################################################################
# Compute the misfit between observed and predicted data
print(fdp.dataMisfit())

################################################################################
# Plot the misfits for a range of half space conductivities
plt.figure()
# _ = fdp.plotHalfSpaceResponses(-6.0, 4.0, 200)

plt.title("Halfspace responses");

################################################################################
# We can attach priors to the height of the datapoint,
# the relative error multiplier, and the additive error noise floor


# Define the distributions used as priors.
heightPrior = Distribution('Uniform', min=np.float64(fdp.z) - 2.0, max=np.float64(fdp.z) + 2.0)
relativePrior = Distribution('Uniform', min=0.01, max=0.5)
additivePrior = Distribution('Uniform', min=5, max=15)
fdp.set_priors(height_prior=heightPrior, relative_error_prior=relativePrior, additive_error_prior=additivePrior)

################################################################################
# In order to perturb our solvable parameters, we need to attach proposal distributions
heightProposal = Distribution('Normal', mean=fdp.z, variance = 0.01)
relativeProposal = Distribution('MvNormal', mean=fdp.relErr, variance=2.5e-7)
additiveProposal = Distribution('MvLogNormal', mean=fdp.addErr, variance=1e-4)
fdp.set_proposals(heightProposal, relativeProposal, additiveProposal)

###############################################################################
# With priors set we can auto generate the posteriors
fdp.set_posteriors()

# Perturb the datapoint and record the perturbations
for i in range(10000):
    fdp.perturb()
    fdp.updatePosteriors()

################################################################################
# Plot the posterior distributions
fig = plt.figure()
gs = fig.add_gridspec(nrows=1, ncols=1)
ax = fdp.init_posterior_plots(gs[0, 0])
fig.tight_layout()

fdp.plot_posteriors(axes=ax, best=fdp)

# import h5py
# with h5py.File('fdp.h5', 'w') as f:
#     fdp.toHdf(f, 'fdp', withPosterior=True)

# with h5py.File('fdp.h5', 'r') as f:
#     fdp1 = FdemDataPoint.fromHdf(f['fdp'])




plt.show()