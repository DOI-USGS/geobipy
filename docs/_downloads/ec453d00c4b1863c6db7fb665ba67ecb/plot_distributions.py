"""
Distribution Class
++++++++++++++++++

Handles the initialization of different statistical distribution
"""

#%%
from geobipy import Distribution
from geobipy import plotting as cP
import matplotlib.pyplot as plt
import numpy as np

from numpy.random import Generator
from numpy.random import PCG64DXSM
generator = PCG64DXSM(seed=0)
prng = Generator(generator)

#%%
# Univariate Normal Distribution
# ++++++++++++++++++++++++++++++
D = Distribution('Normal', 0.0, 1.0, prng=prng)

# Get the bins of the Distribution from +- 4 standard deviations of the mean
bins = D.bins()

# Grab random samples from the distribution
D.rng(10)

# We can then get the Probability Density Function for those bins
pdf = D.probability(bins, log=False)

# And we can plot that PDF
# sphinx_gallery_thumbnail_number = 1
plt.figure()
plt.plot(bins, pdf)

#%%
# Multivariate Normal Distribution
# ++++++++++++++++++++++++++++++++
D = Distribution('MvNormal',[0.0,1.0,2.0],[1.0,1.0,1.0], prng=prng)
D.rng()


#%%
# Uniform Distribution
D = Distribution('Uniform', 0.0, 1.0, prng=prng)
D.bins()
