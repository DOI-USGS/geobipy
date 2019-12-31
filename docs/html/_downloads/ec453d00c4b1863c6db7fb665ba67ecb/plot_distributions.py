"""
Distribution Class
++++++++++++++++++

Handles the initialization of different statistical distribution
"""

#%%
from geobipy import Distribution
from geobipy import customPlots as cP
import matplotlib.pyplot as plt
import numpy as np

#%%
# Univariate Normal Distribution
# ++++++++++++++++++++++++++++++
D = Distribution('Normal', 0.0, 1.0)

# Get the bins of the Distribution from +- 4 standard deviations of the mean
bins = D.bins()

# Grab random samples from the distribution
D.rng(10)

# We can then get the Probability Density Function for those bins
pdf = D.probability(bins, log=False)

# And we can plot that PDF
# sphinx_gallery_thumbnail_number = 1
plt.figure()
_ = cP.plot(bins,pdf)

#%%
# Multivariate Normal Distribution
# ++++++++++++++++++++++++++++++++
D = Distribution('MvNormal',[0.0,1.0,2.0],[1.0,1.0,1.0])
D.rng()


################################################################################
# Uniform Distribution
D = Distribution('Uniform', 0.0, 1.0)
D.bins()
