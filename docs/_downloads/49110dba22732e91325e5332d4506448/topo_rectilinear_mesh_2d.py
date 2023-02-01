"""
Topo Rectilinear Mesh 2D
------------------------
The Topo version of the rectilinear mesh has the same functionality as the
:ref:`Rectilinear Mesh 2D` but the top surface of the mesh can undulate.

"""

#%%
from geobipy import StatArray
from geobipy import RectilinearMesh2D
import matplotlib.pyplot as plt
import numpy as np


#%%
# Specify some cell centres in x and y
x = StatArray(np.arange(11.0), 'Easting', 'm')
y = StatArray(np.arange(11.0), 'Depth', 'm')
# Create a height profile for the mesh
height = StatArray(np.asarray([5,4,3,2,1,1,1,2,3,4,5])*3.0, 'Height', 'm')
# Instantiate the mesh
rm = RectilinearMesh2D(xEdges=x, yEdges=y, heightEdges=height)

################################################################################
# Plot only the grid lines of the mesh
plt.figure()
_ = rm.plotGrid(linewidth=0.5)

################################################################################
# Create some cell values
values = StatArray(np.random.random(rm.shape), 'Name', 'Units')

################################################################################
plt.figure()
_ = rm.pcolor(values, grid=True, linewidth=0.1, xAxis='x')

################################################################################
# Mask the x axis cells by a distance
rm_masked, x_indices, z_indices, arr2 = rm.mask_cells(xAxis='x', x_distance=0.4, values=values)
plt.figure()
_ = rm_masked.pcolor(StatArray(arr2), grid=True)

################################################################################
# Mask the z axis cells by a distance
rm_masked, x_indices, z_indices, arr2 = rm.mask_cells(xAxis='x', z_distance=0.4, values=values)
plt.figure()
_ = rm_masked.pcolor(StatArray(arr2), grid=True)

################################################################################
# Mask axes by a distance
rm_masked, x_indices, z_indices, arr2 = rm.mask_cells(xAxis='x', x_distance=0.4, z_distance=0.4, values=values)
plt.figure()
_ = rm_masked.pcolor(StatArray(arr2), grid=True)

################################################################################
# Compute the mean over an interval for the mesh.
rm.intervalStatistic(values, intervals=[6.8, 12.4], axis=0)

################################################################################
# Compute the mean over multiple intervals for the mesh.
rm.intervalStatistic(values, intervals=[6.8, 12.4, 20.0, 40.0], axis=0)


# ################################################################################
# # We can apply the interval statistics to either axis
# rm.intervalStatistic(values, intervals=[2.8, 4.2], axis=1)


# ################################################################################
# rm.intervalStatistic(values, intervals=[2.8, 4.2, 5.1, 8.4], axis=1)


# ################################################################################
# rm.ravelIndices([[3, 4], [5, 5]])


# ################################################################################
# rm.unravelIndex([35, 45])


# ################################################################################
# # 2D draped rectlinear mesh embedded in 3D
# # ++++++++++++++++++++++++++++++++++++++
# z = StatArray(np.cumsum(np.arange(10.0)), 'Depth', 'm')
# rm = RectilinearMesh2D(xCentres=x, yCentres=y, zCentres=z, heightCentres=height)
# values = StatArray(np.arange(rm.nCells, dtype=np.float64).reshape(rm.shape), 'Name', 'Units')


# ################################################################################
# plt.figure()
# rm.plotGrid(linewidth=1)

# ################################################################################
# # Plot the x-y co-ordinates
# plt.figure()
# rm.plotXY()

# ################################################################################
# # The pcolor function can now be plotted against distance
# plt.figure()
# rm.pcolor(values, grid=True, xAxis='r', linewidth=0.5)

################################################################################
import h5py
with h5py.File('rm2d.h5', 'w') as f:
    rm.toHdf(f, 'test')

with h5py.File('rm2d.h5', 'r') as f:
    rm2 = RectilinearMesh2D.fromHdf(f['test'])

plt.figure()
plt.subplot(121)
rm.plotGrid(linewidth=0.5)
plt.subplot(122)
rm2.plotGrid(linewidth=0.5)

################################################################################
# rm.toVTK('test', cellData=values)

plt.show()