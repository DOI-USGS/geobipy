"""
1D Model without an infinite halfspace
--------------------------------------

"""

#%%
from geobipy import StatArray
from geobipy import Model1D
from geobipy import Distribution
import matplotlib.pyplot as plt
import numpy as np
import h5py
from geobipy import hdfRead

################################################################################
# Make a test model with 10 layers, and increasing parameter values

par = StatArray(np.linspace(0.001, 0.02, 10), "Conductivity", "$\\frac{S}{m}$")
thk = StatArray(np.arange(1, 11))
mod = Model1D(parameters=par, thickness=thk, hasHalfspace=False)

################################################################################
# Randomness and Model Perturbations
# ++++++++++++++++++++++++++++++++++
# We can set the priors on the 1D model by assigning minimum and maximum layer
# depths and a maximum number of layers.  These are used to create priors on
# the number of cells in the model, a new depth interface, new parameter values
# and the vertical gradient of those parameters.
# The halfSpaceValue is used as a reference value for the parameter prior.
prng = np.random.RandomState()
# Set the priors
mod.setPriors(halfSpaceValue = 0.01,
              minDepth = 1.0, 
              maxDepth = 150.0, 
              maxLayers = 30, 
              parameterPrior = True, 
              gradientPrior = True, 
              prng = prng)

################################################################################
# To propose new models, we specify the probabilities of creating, removing, perturbing, and not changing
# a layer interface
pProposal = Distribution('LogNormal', 0.01, np.log(2.0)**2.0, linearSpace=True, prng=prng)
mod.setProposals(probabilities = [0.25, 0.25, 0.25, 0.25], parameterProposal=pProposal, prng=prng)

################################################################################
# We can then perturb the layers of the model
# perturbed = mod.perturbStructure()
remapped, perturbed = mod.perturb()

################################################################################
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(121)
mod.pcolor(grid=True)
ax = plt.subplot(122)
perturbed.pcolor(grid=True)

################################################################################
# We can evaluate the prior of the model using depths only
print('Log probability of the Model given its priors: ', mod.priorProbability(False, False, log=True))
# Or with priors on its parameters, and parameter gradient with depth.
print('Log probability of the Model given its priors: ', mod.priorProbability(True, True, log=True))

#%%
# Perturbing a model multiple times
# +++++++++++++++++++++++++++++++++
# In the stochasitic inference process, we perturb the model structure, 
# and parameter values, multiple times. 
# Each time the model is perturbed, we can record its state
# in a posterior distribution.
#
# For a 1D model, the parameter posterior is a 2D hitmap with depth in one dimension
# and the parameter value in the other.
# We also attach a 1D histogram for the number of layers,
# and a 1D histogram for the locations of interfaces.
#
# Since we have already set the priors on the Model, we can set the posteriors
# based on bins from from the priors.
mod.setPosteriors()

mod0 = mod.deepcopy()

################################################################################
# Now we randomly perturb the model, and update its posteriors.
mod.updatePosteriors()
for i in range(1000):
    remapped, perturbed = mod.perturb()

    # And update the model posteriors
    perturbed.updatePosteriors()

    mod = perturbed

################################################################################
# We can now plot the posteriors of the model.
#
# Remember in this case, we are simply perturbing the model structure and parameter values
# The proposal for the parameter values is fixed and centred around a single value.
fig = plt.figure(figsize=(8, 6))
plt.subplot(131)
mod.nCells.posterior.plot()
ax = plt.subplot(132)
mod.par.posterior.pcolor(cmap='gray_r', xscale='log', noColorbar=True, flipY=True)
plt.subplot(133, sharey=ax)
mod.depth.posterior.plot(rotate=True, flipY=True);

