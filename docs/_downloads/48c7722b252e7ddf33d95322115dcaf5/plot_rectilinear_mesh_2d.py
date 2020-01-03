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
arr = StatArray(np.random.random(rm.shape), 'Name', 'Units')

plt.figure()
_ = rm.pcolor(arr, xAxis='r', grid=True, flipY=True, linewidth=0.5)

################################################################################
# We can perform some interval statistics on the cell values of the mesh
# Generate some values
a = np.repeat(np.arange(1.0, np.float(rm.x.nCells+1))[:, np.newaxis], rm.z.nCells, 1).T


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
# rm.toVTK('test', cellData=StatArray(np.random.randn(z.size, x.size), "Name"))
