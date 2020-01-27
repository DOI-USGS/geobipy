"""
Frequency domain datapoint
--------------------------
"""

#%%
# There are two ways in which to create a datapoint,
#
# 1) :ref:`Instantiating a frequency domain data point`
#
# 2) :ref:`Obtaining a datapoint from a dataset`

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
# Note the extra arguments that can be used to create the data point.
# data is for any observed data one might have, while std are the estimated standard 
# deviations of those observed data.
fdp = FdemDataPoint(x=0.0, y=0.0, z=30, elevation=0.0, 
                    data=None, std=None, predictedData=None, 
                    system=fds, lineNumber=0.0, fiducial=0.0)

################################################################################
# We can define a 1D layered earth model, and use it to predict some data
nCells = 19
par = StatArray(np.linspace(0.01, 0.1, nCells), "Conductivity", "$\frac{S}{m}$")
thk = StatArray(np.ones(nCells-1) * 10.0)
mod = Model1D(nCells = nCells, parameters=par, thickness=thk)

################################################################################
# Forward model the data
fdp.forward(mod)

################################################################################
plt.figure()
plt.subplot(121)
_ = mod.pcolor()
plt.subplot(122)
_ = fdp.plotPredicted()
plt.tight_layout()

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
D = FdemData()
D.read(dataFile,systemFile)

################################################################################
# Get a data point from the dataset
P = D.datapoint(0)
plt.figure()
_ = P.plot()

################################################################################
# Predict data using the same model as before
P.forward(mod)
plt.figure()
_ = P.plot()
_ = P.plotPredicted()
plt.tight_layout();

################################################################################
# Attaching statistical descriptors to the datapoint
# ++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Define a multivariate log normal distribution as the prior on the predicted data.
P.predictedData.setPrior('MvLogNormal', P.data[P.active], P.std[P.active]**2.0)

################################################################################
# This allows us to evaluate the likelihood of the predicted data
print(P.likelihood(log=True))
# Or the misfit
print(P.dataMisfit())

################################################################################
# We can perform a quick search for the best fitting half space
halfspace = P.FindBestHalfSpace()
print('Best half space conductivity is {} $S/m$')
plt.figure()
P.plot()
P.plotPredicted()

################################################################################
# Compute the misfit between observed and predicted data
print(P.dataMisfit())

################################################################################
# Plot the misfits for a range of half space conductivities
plt.figure()
_ = P.plotHalfSpaceResponses(-6.0, 4.0, 200)
plt.title("Halfspace responses")

################################################################################
# Compute the sensitivity matrix for a given model

J = P.sensitivity(mod)
plt.figure()
np.abs(J).pcolor(equalize=True, log=10);

# ################################################################################
# # We can save the FdemDataPoint to a HDF file

# with h5py.File('FdemDataPoint.h5','w') as hf:
#     P.createHdf(hf, 'fdp')
#     P.writeHdf(hf, 'fdp')

# ################################################################################
# # And then read it in

# P1=hdfRead.readKeyFromFiles('FdemDataPoint.h5','/','fdp')
