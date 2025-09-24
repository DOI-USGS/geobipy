"""
Tempest Datapoint Class
-----------------------
"""

#%%
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
from geobipy import TempestData
# from geobipy import TemDataPoint
from geobipy import RectilinearMesh1D
from geobipy import Model
from geobipy import StatArray
from geobipy import Distribution
from geobipy import get_prng

dataFolder = "..//..//supplementary//data//"
# dataFolder = "source//examples//supplementary//Data"

# Obtaining a tempest datapoint from a dataset
# ++++++++++++++++++++++++++++++++++++++++++++
# More often than not, our observed data is stored in a file on disk.
# We can read in a dataset and pull datapoints from it.
#
# For more information about the time domain data set, see :ref:`Time domain dataset`

# The data file name
dataFile = dataFolder + 'tempest_saline_clay.csv'
# The EM system file name
systemFile = dataFolder + 'Tempest.stm'

# Prepare the dataset so that we can read a point at a time.
Dataset = TempestData._initialize_sequential_reading(dataFile, systemFile)
# Get a datapoint from the file.
tdp = Dataset._read_record(0)

plt.figure()
tdp.plot()

prng = get_prng(seed=146100583096709124601953385843316024947)

#%%
# Using a tempest domain datapoint
# ++++++++++++++++++++++++++++++++

#%%
# We can define a 1D layered earth model, and use it to predict some data
par = StatArray(np.r_[0.01, 0.1, 1.], "Conductivity", "$\frac{S}{m}$")
mod = Model(mesh=RectilinearMesh1D(edges=np.r_[0.0, 50.0, 75.0, np.inf]), values=par)

# par = StatArray(np.logspace(-3, 3, 30), "Conductivity", "$\frac{S}{m}$")
# e = np.linspace(0, 350, 31); e[-1] = np.inf
# mod = Model(mesh=RectilinearMesh1D(edges=e), values=par)

#%%
# Forward model the data
tdp.forward(mod)

#%%
plt.figure()
plt.subplot(121)
_ = mod.pcolor(transpose=True)
plt.subplot(122)
_ = tdp.plot()
_ = tdp.plot_predicted()
plt.tight_layout()
plt.suptitle('Model and response')

#%%
plt.figure()
tdp.plotDataResidual(xscale='log')
plt.title('data residual')

#%%
from geobipy import DataArray
# Compute the sensitivity matrix for a given model
J = tdp.sensitivity(mod)
J = np.dot(J.T, J)
plt.figure()
_ = J.pcolor(equalize=True, flipY=True)

#%%
# Attaching statistical descriptors to the tempest datapoint
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from numpy.random import Generator
from numpy.random import PCG64DXSM
generator = PCG64DXSM(seed=0)
prng = Generator(generator)

# Set relative errors for the primary fields, and secondary fields.
tdp.relative_error = 0.001

# Set the additive errors for
tmp = np.asarray([[0.011474, 0.012810, 0.008507, 0.005154, 0.004742, 0.004477, 0.004168, 0.003539, 0.003352, 0.003213, 0.003161, 0.003122, 0.002587, 0.002038, 0.002201],
                  [0.007383, 0.005693, 0.005178, 0.003659, 0.003426, 0.003046, 0.003095, 0.003247, 0.002775, 0.002627, 0.002460, 0.002178, 0.001754, 0.001405, 0.001283]])
tdp.additive_error = np.sqrt((tmp**2.0).sum(axis=0))
# Define a multivariate log normal distribution as the prior on the predicted data.
tdp.predicted_data.prior = Distribution('MvLogNormal', tdp.data[tdp.active], tdp.std[tdp.active]**2.0, prng=prng)

#%%
# This allows us to evaluate the likelihood of the predicted data
print(tdp.likelihood(log=True))
# Or the misfit
print(tdp.data_misfit())

#%%
# Plot the misfits for a range of half space conductivities
plt.figure()
plt.subplot(1, 2, 1)
_ = tdp.plot_halfspace_responses(-6.0, 4.0, 200)
plt.title("Halfspace responses")

#%%
# We can perform a quick search for the best fitting half space
halfspace = tdp.find_best_halfspace()
print('Best half space conductivity is {} $S/m$'.format(halfspace.values))
plt.subplot(1, 2, 2)
_ = tdp.plot()
_ = tdp.plot_predicted()

plt.figure()
tdp.plot_secondary_field()
tdp.plot_predicted_secondary_field()

#%%
# We can attach priors to the height of the datapoint,
# the relative error multiplier, and the additive error noise floor

# Define the distributions used as priors.
relative_prior = Distribution('Uniform', min=0.01, max=0.5, prng=prng)
receiver_x_prior = Distribution('Uniform', min=tdp.receiver.x - 1.0, max=tdp.receiver.x + 1.0, prng=prng)
receiver_z_prior = Distribution('Uniform', min=tdp.receiver.z - 1.0, max=tdp.receiver.z + 1.0, prng=prng)
transmitter_pitch_prior = Distribution('Uniform', min=tdp.receiver.pitch - 5.0, max=tdp.receiver.pitch + 5.0, prng=prng)
tdp.set_priors(prng=prng,
               relative_error_prior=relative_prior,
               receiver_x_prior=receiver_x_prior,
               receiver_z_prior=receiver_z_prior,
               transmitter_pitch_prior=transmitter_pitch_prior)

#%%
# In order to perturb our solvable parameters, we need to attach proposal distributions
relative_proposal = Distribution('MvNormal', mean=tdp.relative_error, variance=1e-5, prng=prng)
receiver_x_proposal = Distribution('Normal', mean=tdp.receiver.x, variance = 0.01, prng=prng)
receiver_z_proposal = Distribution('Normal', mean=tdp.receiver.z, variance = 0.01, prng=prng)
transmitter_pitch_proposal = Distribution('Normal', mean=tdp.receiver.pitch, variance = 0.01, prng=prng)
tdp.set_proposals(prng=prng,
                  relative_error_proposal=relative_proposal,
                  receiver_x_proposal=receiver_x_proposal,
                  receiver_z_proposal=receiver_z_proposal,
                  transmitter_pitch_proposal=transmitter_pitch_proposal)
                #   solve_additive_error=True, additive_error_proposal_variance=1e-4, prng=prng)

#%%
# With priors set we can auto generate the posteriors
tdp.set_posteriors()

#%%
# Perturb the datapoint and record the perturbations
# Note we are not using the priors to accept or reject perturbations.
for i in range(1000):
    tdp.perturb()
    tdp.update_posteriors()

plt.figure()
ax = tdp._init_posterior_plots()
print(ax)

tdp.plot_posteriors(axes=ax)

plt.show()
# %%
