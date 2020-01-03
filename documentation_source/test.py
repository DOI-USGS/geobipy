from geobipy import StatArray
from geobipy import Histogram1D
import numpy as np
import matplotlib.pyplot as plt
import h5py
from geobipy import hdfRead


#%%
# Instantiating a new StatArray class
# +++++++++++++++++++++++++++++++++++
#
# The StatArray can take any numpy function that returns an array as an input.
# The name and units of the variable can be assigned to the StatArray.

Density = StatArray(np.random.randn(1), name="Density", units="$\frac{g}{cc}$")
Density.summary()


#%%
# Attaching Prior and Proposal Distributions to a StatArray
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# The StatArray class has been built so that we may easily attach not only names and units, but statistical distributions too.  We won't go into too much detail about the different distribution classes here so check out the :ref:`Distribution Class` for a better description.
#
# Two types of distributions can be attached to the StatArray.
#
# * Prior Distribution
#     The prior represents how the user believes the variable should behave from a statistical standpoint.  The values of the variable can be evaluated against the attached prior, to determine how likely they are to have occured https://en.wikipedia.org/wiki/Prior_probability
#
# * Proposal Distribution
#     The proposal describes a probability distribution from which to sample when we wish to perturb the variable https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

# Obtain an instantiation of a random number generator
prng = np.random.RandomState()
Density.setPrior('Uniform', -2.0, 2.0, prng=prng)

#%%
# We can also attach a proposal distribution

Density.setProposal('Normal', 0.0, 1.0, prng=prng)
Density.summary()
print("Class type of the prior: ",type(Density.prior))
print("Class type of the proposal: ",type(Density.proposal))


#%%
# The values in the variable can be evaluated against the prior.
# In this case, we have 3 elements in the variable, and a univariate Normal for the prior. 
# Therefore each element is evaluated to get 3 probabilities, one for each element.
print(Density.probability(log=False))

################################################################################
# The univariate proposal distribution can generate random samples from itself.
proposed = Density.propose()
print(proposed)


################################################################################
# We can perturb the variable by drawing from the attached proposal distribution.
Density.perturb()
