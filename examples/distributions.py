"""
Distribution Class
------------------

Handles the initialization of different statistical distribution
"""
from geobipy import Distribution
from geobipy import customPlots as cP
import numpy as np

################################################################################
# Normal Distribution
D = Distribution('Normal',0.0,1.0)

################################################################################
# Grab random samples from the distribution
D.rng(10)

################################################################################
# Get the bins of the Distribution from +- 4 standard deviations of the mean
bins = D.getBins()
bins

################################################################################
# We can then get the Probability Density Function for those bins
pdf = D.probability(bins)

################################################################################
# And we can plot that PDF
cP.plot(bins,pdf)

################################################################################
# Multivariate Normal Distribution
D = Distribution('MvNormal',[0.0,1.0,2.0],[1.0,1.0,1.0])
D.rng()

################################################################################
bins = D.getBins()

s = np.random.random.__self__
s.gamma

bins

################################################################################
# Uniform Distribution
D = Distribution('Uniform', 0.0, 1.0)
D.getBins()
