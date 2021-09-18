"""
2D Rectilinear Mesh
-------------------
This 2D rectilinear mesh defines a grid with straight cell boundaries.

It can be instantiated in two ways.

The first is by providing the cell centres or
cell edges in two dimensions.

The second embeds the 2D mesh in 3D by providing the cell centres or edges in three dimensions.
The first two dimensions specify the mesh coordinates in the horiztontal cartesian plane
while the third discretizes in depth. This allows us to characterize a mesh whose horizontal coordinates
do not follow a line that is parallel to either the "x" or "y" axis.

"""

#%%
from geobipy import StatArray
from geobipy import RectilinearMesh2D
import matplotlib.pyplot as plt
import numpy as np


#%%
# Specify some cell centres in x and y
x = StatArray(np.arange(10.0), 'Easting', 'm')
y = StatArray(np.arange(10.0), 'Northing', 'm')
rm = RectilinearMesh2D(xCentres=x, yCentres=y)

################################################################################
# We can plot the grid lines of the mesh.
plt.figure()
_  = rm.plotGrid(linewidth=0.5)

###############################################################################
# Intersecting lines with a mesh
arr = np.zeros(rm.shape)

################################################################################
# Intersecting multisegment lines with a mesh
# arr = np.zeros(rm.shape)
# ix, iy = rm.line_indices([0.0, 3.0, 6.0, 9], [2.0, 6.0, -10.0, 10])
# arr[iy, ix] = 1
# plt.figure()
# rm.pcolor(values = arr)

################################################################################
# 2D Mesh embedded in 3D
# ++++++++++++++++++++++
z = StatArray(np.cumsum(np.arange(15.0)), 'Depth', 'm')
rm = RectilinearMesh2D(xCentres=x, yCentres=y, zCentres=z)

################################################################################
# Plot the x-y coordinates of the mesh
plt.figure()
_ = rm.plotXY()

################################################################################
# Again, plot the grid. This time the z-coordinate dominates the plot.
plt.figure()
_ = rm.plotGrid(xAxis='r', flipY=True, linewidth=0.5)

################################################################################
# We can pcolor the mesh by providing cell values.
xx, yy = np.meshgrid(rm.x.centres, rm.z.centres)
arr = StatArray(np.sin(np.sqrt(xx ** 2.0 + yy ** 2.0)), "Values")

plt.figure()
_ = rm.pcolor(arr, xAxis='r', grid=True, flipY=True, linewidth=0.5)

xG = rm.xGradientMatrix()
zG = rm.zGradientMatrix()

# dax = StatArray((xG * arr.flatten()).reshape((arr.shape[0], arr.shape[1]-1)))
# rm2 = rm[:, :9]

# plt.figure()
# rm2.pcolor(dax, xAxis='r', grid=True, flipY=True, linewidth=0.5)

# dax = StatArray((zG * arr.flatten()).reshape((arr.shape[0]-1, arr.shape[1])))

# plt.figure()
# dax.pcolor(grid=True, flipY=True, linewidth=0.5)

################################################################################
# Mask the x axis cells by a distance
rm_masked, x_indices, z_indices, arr2 = rm.mask_cells(xAxis='x', x_distance=0.4, values=arr)
plt.figure()
_ = rm_masked.pcolor(StatArray(arr2), grid=True, flipY=True)

################################################################################
# Mask the z axis cells by a distance
rm_masked, x_indices, z_indices, arr2 = rm.mask_cells(xAxis='x', z_distance=4.9, values=arr)
plt.figure()
_ = rm_masked.pcolor(StatArray(arr2), grid=True, flipY=True)

################################################################################
# Mask axes by a distance
rm_masked, x_indices, z_indices, arr2 = rm.mask_cells(xAxis='x', x_distance=0.4, z_distance=4.9, values=arr)
plt.figure()
_ = rm_masked.pcolor(StatArray(arr2), grid=True, flipY=True)

################################################################################
# We can perform some interval statistics on the cell values of the mesh
# Generate some values
a = np.repeat(np.arange(1.0, np.float64(rm.x.nCells+1))[:, np.newaxis], rm.z.nCells, 1).T

################################################################################
# Compute the mean over an interval for the mesh.
rm.intervalStatistic(a, intervals=[6.8, 12.4], axis=0, statistic='mean')

################################################################################
# Compute the mean over multiple intervals for the mesh.
rm.intervalStatistic(a, intervals=[6.8, 12.4, 20.0, 40.0], axis=0, statistic='mean')

################################################################################
# We can specify either axis
rm.intervalStatistic(a, intervals=[2.8, 4.2], axis=1, statistic='mean')

################################################################################
rm.intervalStatistic(a, intervals=[2.8, 4.2, 5.1, 8.4], axis=1, statistic='mean')

################################################################################
# Slice the 2D mesh to retrieve either a 2D mesh or 1D mesh
rm2 = rm[:5, :5]
rm3 = rm[:, 5]
rm4 = rm[5, :]

plt.figure()
plt.subplot(131)
rm2.plotGrid()
plt.subplot(132)
rm3.plotGrid()
plt.subplot(133)
rm4.plotGrid()

################################################################################
# Resample a grid
values = StatArray(np.random.randn(*rm.shape))
rm2, values2 = rm.resample(0.5, 0.5, values)



plt.figure()
plt.subplot(121)
rm.pcolor(values)
plt.subplot(122)
rm2.pcolor(values2)

################################################################################
# Axes in log space
# +++++++++++++++++
x = StatArray(np.logspace(-1, 4, 10), 'x')
y = StatArray(np.logspace(0, 3, 10), 'y')
rm = RectilinearMesh2D(xEdges=x, xlog=10, yEdges=y, ylog=10)

#################################################################
# We can plot the grid lines of the mesh.
plt.figure()
_  = rm.plotGrid(linewidth=0.5)

###############################################################################
# Intersecting lines with a mesh
x = np.r_[0.1, 1000.0]
y = np.r_[1.0, 1000.0]

###############################################################################
# Intersecting multisegment lines with a mesh
# arr = np.zeros(rm.shape)
# ix, iy = rm.line_indices([0.0, 3.0, 6.0, 9], [2.0, 6.0, -10.0, 10])
# arr[iy, ix] = 1
# plt.figure()
# rm.pcolor(values = arr)


################################################################################
import h5py
with h5py.File('rm2d.h5', 'w') as f:
    rm.toHdf(f, 'test')

with h5py.File('rm2d.h5', 'r') as f:
    rm2 = RectilinearMesh2D.fromHdf(f['test'])

values = StatArray(np.random.randn(*rm2.shape))

plt.figure()
rm2.pcolor(values)

plt.show()
