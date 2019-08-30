"""
Stat Array Class
----------------

Extends the numpy ndarray class to add extra attributes such as names, and
units, and allows us to attach statistical descriptors of the array.
The direct extension to numpy maintains speed and functionality of numpy arrays.

"""

from geobipy import StatArray
import numpy as np
import matplotlib.pyplot as plt
import h5py
from geobipy import hdfRead


################################################################################
# Instantiating a new StatArray class
# ===================================
#
# The StatArray can take any numpy function that returns an array as an input.
# The name and units of the variable can be assigned to the StatArray.

Density = StatArray(np.random.randn(3), name="Density", units="$\frac{g}{cc}$")
Density.summary()


################################################################################
# Attaching Prior and Proposal Distributions to a StatArray
# =========================================================
#
# The StatArray class has been built so that we may easily attach not only names and units, but statistical distributions too.  We won't go into too much detail about the different distribution classes here so check out [This Notebook](Distributions.ipynb) for a better description.
#
# Two types of distributions can be attached to the StatArray.
#
# * Prior Distribution
#     The prior represents how the user believes the variable should behave from a statistical standpoint.  The values of the variable can be evaluated against the attached prior, to determine how likely they are to have occured https://en.wikipedia.org/wiki/Prior_probability
#
# * Proposal Distribution
#     The proposal describes a probability distribution from which to sample when we wish to perturb the variable https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
#
# Attach a univariate normal distribution as the prior
# ++++++++++++++++++++++++++++++++++++++++++++++++++++


# Obtain an instantiation of a random number generator
prng = np.random.RandomState()
mean = 0.0
variance = 1.0
Density.setPrior('Normal', mean, variance, prng=prng)

################################################################################
# We can also attach a proposal distribution

Density.setProposal('Normal', mean, variance, prng=prng)
Density.summary()
print("Class type of the prior: ",type(Density.prior))
print("Class type of the proposal: ",type(Density.proposal))


################################################################################
# The values in the variable can be evaluated against the prior
# In this case, we have 3 elements in the variable, and a univariate Normal for the prior. Therefore each element is evaluated to get 3 probabilities, one for each element.

print(Density.probability())

################################################################################
# The univarite proposal distribution can generate random samples from itself.

print(Density.proposal.rng())


################################################################################
# We can perturb the variable by drawing from the attached proposal distribution.

Density.perturb()
Density.summary()

################################################################################
# Attach a multivariate normal distribution as the prior and proposal
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Attach the multivariate prior

mean = np.random.randn(Density.size)
variance = np.ones(Density.size)
Density.setPrior('MvNormal', mean, variance, prng=prng)


################################################################################
# Since the prior is multivariate, the appropriate equations are used to
# evaluate the probability for all elements in the StatArray.
# This produces a single probability.

print(Density.probability())

################################################################################
# Attach the multivariate proposal

mean = np.random.randn(Density.size)
variance = np.ones(Density.size)
Density.setProposal('MvNormal', mean, variance, prng=prng)


################################################################################
# Perturb the variables using the multivariate proposal.

Density.perturb()
Density.summary()


################################################################################
# Basic manipulation
# ==================
#
# The StatArray contains other functions to perform basic array manipulations
#
# These routines essentially wrap around numpy functions, but the result will have the same name and units, and if any prior or proposal are set, those will be carried through too.
#
# 1D example
# ++++++++++

x = StatArray(-np.cumsum(np.arange(10.0)))
print(x)

################################################################################


print(x.insert(i=[0, 9], values=[999.0, 999.0]))


################################################################################


print(x.prepend(999.0))


################################################################################


print(x.prepend([998.0, 999.0]))


################################################################################


print(x.append([998.0, 999.0]))


################################################################################


print(x.resize(14))


################################################################################


print(x.delete([5,8]))


################################################################################


print(x.edges())


################################################################################


print(x.internalEdges())


################################################################################


print(x.firstNonZero())


################################################################################


print(x.lastNonZero())


################################################################################


print(x.abs())


################################################################################
# 2D example
# ++++++++++

x = StatArray(np.asarray([[0, -2, 3],[3, 0, -1],[1, 2, 0]]))
print(x)


################################################################################


print(x.insert(i=0, values=4))


################################################################################


print(x.insert(i=[2, 3], values=5, axis=1))


################################################################################


print(x.insert(i=2, values=[10, 11, 12], axis=1))


################################################################################


print(x.prepend(999))


################################################################################


print(x.prepend([999, 998, 997], axis=1))


################################################################################


print(x.append([[999, 998, 997]]))


################################################################################


print(x.resize([5,5]))


################################################################################


print(x.delete(5))


################################################################################


print(x.delete(2, axis=0))


################################################################################


print(x.firstNonZero(axis=0))


################################################################################


print(x.lastNonZero(axis=0))


################################################################################


print(x.firstNonZero(axis=1))


################################################################################


print(x.lastNonZero(axis=1))


################################################################################


print(x.abs())


################################################################################
# Plotting
# ========
#
# We can easily plot the StatArray with its built in plotting functions.
# All plotting functions can take the matplotlib keywords

# The simplest is to just plot the array

Density = StatArray(np.random.randn(100),name="Density",units="$\frac{g}{cc}$")
Time = StatArray(np.linspace(0, 100, Density.size), name='Time', units='s')
Depth = StatArray(np.random.exponential(size=Density.size), name='Depth', units='m')


################################################################################


plt.figure()
Density.plot(linewidth=0.5, marker='x', markersize=1.0)

################################################################################
# We can quickly plot a bar graph.

plt.figure()
Density.bar()


################################################################################
# We can scatter the contents of the StatArray if it is 1D

plt.figure()
Density.scatter(alpha=0.7)


################################################################################
# Histogram Equalization
# ++++++++++++++++++++++
#
# A neat trick with colourmaps is histogram equalization.
# This approach forces all colours in the images to have an equal weight.
# This distorts the colour bar, but can really highlight the lower and higher
# ends of whatever you are plotting. Just add the equalize keyword!


plt.figure()
Density.scatter(alpha=0.7, equalize=True)


################################################################################
# Take the log base(x) of the data
#
# We can also take the data to a log, log10, log2, or a custom number!

plt.figure()
Density.scatter(alpha=0.7,edgecolor='k',log='e') # could also use log='e', log=2, log=x) where x is the base you require

################################################################################
# X and Y axes
#
# We can specify the x axis of the scatter plot.


plt.figure()
Density.scatter(x=Time, alpha=0.7, edgecolor='k')


################################################################################
# Notice that I never specified the y axis, so the y axis defaulted to the values in the StatArray. 
# In this case, any operations applied to the colours, are also applied to the y axis, e.g. log=10.  
# When I take the values of Density to log base 10, because I do not specify the y plotting locations, those locations are similarly affected.
#
# I can however force the y co-ordinates by specifying it as input. 
# In the second subplot I explicitly plot distance on the y axis. 
# In the first subplot, the y axis is the same as the colourbar.


plt.figure()
ax1 = plt.subplot(211)
Density.scatter(x=Time, alpha=0.7, edgecolor='k', log=10)
plt.subplot(212, sharex=ax1)
Density.scatter(x=Time, y=Depth, alpha=0.7, edgecolor='k', log=10)


################################################################################
# Point sizes
#
# Since the plotting functions take matplotlib keywords, I can also specify the size of each points.

################################################################################


s = np.ceil(100*(np.abs(np.random.randn(Density.size))))
plt.figure()
plt.tight_layout()
ax1 = plt.subplot(211)
Density.scatter(x=Time, y=Depth, s=s, alpha=0.7,edgecolor='k', sizeLegend=2)
plt.subplot(212, sharex=ax1)
#Density.scatter(x=Time, y=Depth, s=s, alpha=0.7,edgecolor='k', sizeLegend=[1.0, 100, 200, 300])
v = np.abs(Density)+1.0
Density.scatter(x=Time, y=Depth, s=s, alpha=0.7,edgecolor='k', sizeLegend=[1.0, 100, 200, 300], log=10)




################################################################################
# Of course we can still take the log, or equalize the colour histogram

plt.figure()
Density.scatter(x=Time, y=Depth, s=s, alpha=0.7,edgecolor='k',equalize=True,log=10)


################################################################################
# Typically pcolor only works with 2D arrays. The StatArray has a pcolor method that will pcolor a 1D array

plt.figure()
plt.subplot(221)
Density.pcolor()
plt.subplot(222)
Density.pcolor(y=Time)
plt.subplot(223)
Density.pcolor(y=Time, flipY=True)
plt.subplot(224)
Density.pcolor(y=Time, log=10, equalize=True)


################################################################################
# We can add grid lines, and add opacity to each element in the pcolor image
#
# This is useful if the colour values need to be scaled by another variable e.g. variance.


plt.figure()
plt.subplot(121)
Density.pcolor(grid=True, cmap='jet')
plt.subplot(122)
a = np.linspace(1.0, 0.0, Density.size)
Density.pcolor(grid=True, alpha=a, cmap='jet')


################################################################################
# We can plot a histogram of the StatArray

plt.figure()
Density.hist(100)


################################################################################
# We can write the StatArray to a HDF5 file.  HDF5 files are binary files that can include compression.  They allow quick and easy access to parts of the file, and can also be written to and read from in parallel!

with h5py.File('1Dtest.h5','w') as f:
    Density.toHdf(f,'test')


################################################################################
# We can then read the StatArray from the file
# Here x is a new variable, that is read in from the hdf5 file we just wrote.

x = hdfRead.readKeyFromFiles('1Dtest.h5','/','test')
print('x has the same values as Density? ',np.all(x == Density))
x[2] = 5.0 # Change one of the values in x
print('x has its own memory allocated (not a reference/pointer)? ',np.all(x == Density) == False)



################################################################################
# We can also define a 2D array

Density = StatArray(np.random.randn(50,100),"Density","$\frac{g}{cc}$")
Density.summary()


################################################################################
# The StatArray Class's functions work whether it is 1D or 2D
#
# We can still do a histogram

plt.figure()
Density.hist()



################################################################################
# And we can use pcolor to plot the 2D array

plt.figure()
ax = Density.pcolor()




################################################################################
# The StatArray comes with extra plotting options
#
# Here we specify the x and y axes for the 2D array using two other 1D StatArrays

plt.figure()
x = StatArray(np.arange(101),name='x Axis',units = 'mm')
y = StatArray(np.arange(51),name='y Axis',units = 'elephants')
ax=Density.pcolor(x=x, y=y)



################################################################################
# We can plot using a log10 scale, in this case, we have values that are less
# than or equal to 0.0.  Plotting with the log option will by default mask any
# of those values, and will let you know that it has done so!

plt.figure()
ax=Density.pcolor(x=x,y=y,log=2)


################################################################################
# A neat trick with colourmaps is histogram equalization.
# This approach forces all colours in the image to have an equal amount.
# This distorts the colours, but can really highlight the lower and higher
# ends of whatever you are plotting

plt.figure()
ax=Density.pcolor(x=x, y=y, equalize=True)


################################################################################
# We can equalize the log10 plot too :)

plt.figure()
ax=Density.pcolor(x=x,y=y,equalize=True, log=10)


################################################################################
# We can add opacity to each pixel in the image

a = StatArray(np.random.random(Density.shape), 'Opacity from 0.0 to 1.0')


################################################################################


plt.figure()
ax1 = plt.subplot(131)
ax = Density.pcolor(x=x, y=y, flipY=True, linewidth=0.1, noColorbar=True)
plt.subplot(132, sharex=ax1, sharey=ax1)
ax = Density.pcolor(x=x, y=y, alpha=a, flipY=True, linewidth=0.1, noColorbar=True)
plt.subplot(133, sharex=ax1, sharey=ax1)
ax = a.pcolor(x=x, y=y, flipY=True)


################################################################################
# If the array potentially has a lot of white space around the edges, we can trim the image

Density[:10, :] = 0.0
Density[-10:, :] = 0.0
Density[:, :10] = 0.0
Density[:, -10:] = 0.0
plt.figure()
plt.subplot(121)
Density.pcolor()
plt.subplot(122)
Density.pcolor(trim=0.0)


################################################################################
# Create a stacked area plot of a 2D StatArray

A = StatArray(np.abs(np.random.randn(13,100)), name='Variable', units="units")
x = StatArray(np.arange(100),name='x Axis',units = 'mm')
plt.figure()
ax1 = plt.subplot(211)
A.stackedAreaPlot(x=x, axis=1)
plt.subplot(212, sharex=ax1)
A.stackedAreaPlot(x=x, i=np.s_[[1,3,4],:], axis=1, labels=['a','b','c'])
