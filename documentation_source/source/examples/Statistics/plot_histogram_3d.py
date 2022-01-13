"""
Histogram 3D
------------

This 3D histogram class allows efficient updating of histograms, plotting and
saving as HDF5.

"""

#%%
import geobipy
from geobipy import StatArray
from geobipy import Histogram
import matplotlib.pyplot as plt
from geobipy import RectilinearMesh3D
import numpy as np


#%%
# Create some histogram bins in x and y
x = StatArray(np.linspace(-4.0, 4.0, 11), 'Variable 1')
y = StatArray(np.linspace(-4.0, 4.0, 21), 'Variable 2')
z = StatArray(np.linspace(-4.0, 4.0, 31), 'Variable 3')

mesh = RectilinearMesh3D(xEdges=x, yEdges=y, zEdges=z)

################################################################################
# Instantiate
H = Histogram(mesh=mesh)

################################################################################
# Generate some random numbers
a = np.random.randn(100000)
b = np.random.randn(100000)
c = np.random.randn(100000)
x = np.asarray([a, b, c])


################################################################################
# Update the histogram counts
H.update(a, b, c)

################################################################################
plt.figure()
plt.suptitle("Slice half way along each dimension")
for axis in range(3):
    plt.subplot(1, 3, axis+1)
    s = [5 if i  == axis else np.s_[:] for i in range(3)]
    _ = H[tuple(s)].pcolor(cmap='gray_r')

################################################################################
# Generate marginal histograms along an axis
plt.figure()
plt.suptitle("Marginals along each axis")
for axis in range(3):
    plt.subplot(1, 3, axis+1)
    _ = H.marginalize(axis=axis).plot()


################################################################################
# Take the mean estimate from the histogram
plt.figure()
plt.suptitle("Mean along each axis")
for axis in range(3):
    plt.subplot(1, 3, axis+1)
    _ = H.mean(axis=axis).pcolor()

################################################################################
# Take the median estimate from the histogram
plt.figure()
plt.suptitle("Median along each axis")
for axis in range(3):
    plt.subplot(1, 3, axis+1)
    _ = H.median(axis=axis).pcolor()

# ################################################################################
# # We can map the credible range to an opacity or transparency
# H.opacity()
# H.transparency()

H.animate(0, 'test.mp4')

plt.show()

# ################################################################################
# # We can plot the mesh in 3D!
# pv_mesh  = H.plot_pyvista(linewidth=0.5)
# pv_mesh.plot(show_edges=True, show_grid=True)
# %%
