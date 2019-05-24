"""
Topo Rectilinear Mesh 2D
------------------------
"""

################################################################################


from geobipy import StatArray
from geobipy import TopoRectilinearMesh2D
import matplotlib.pyplot as plt
import numpy as np


################################################################################
# Create input StatArrays for the horizontal and vertical (x and y) mesh axes

x = StatArray(np.arange(10.0), 'Easting', 'm')
y = StatArray(np.arange(10.0), 'Northing', 'm')

################################################################################
# Create a height profile for the mesh

height = StatArray(np.asarray([5,4,3,2,1,1,2,3,4,5])*3.0, 'Height', 'm')


################################################################################
# Instantiate the mesh

rm = TopoRectilinearMesh2D(xCentres=x, yCentres=y, heightCentres=height)


################################################################################
# Plot only the grid lines of the mesh

plt.figure()
rm.plotGrid()

################################################################################
# Create an array of random numbers that we can pass to the mesh and perform operations

values = StatArray(np.random.random(rm.dims), 'Name', 'Units')

################################################################################
# Compute the mean over an interval for the mesh.

rm.intervalMean(values, intervals=[6.8, 12.4], axis=0)

################################################################################
# Compute the mean over multiple intervals for the mesh.

rm.intervalMean(values, intervals=[6.8, 12.4, 20.0, 40.0], axis=0)


################################################################################


rm.intervalMean(values, intervals=[2.8, 4.2], axis=1)


################################################################################


rm.intervalMean(values, intervals=[2.8, 4.2, 5.1, 8.4], axis=1)


################################################################################


rm.ravelIndices([[3, 4], [5, 5]])


################################################################################


rm.unravelIndex([35, 45])


################################################################################


plt.figure()
rm.pcolor(values, grid=True, linewidth=0.1, xAxis='x')


################################################################################
# Create a line with three dimensions.

z = StatArray(np.cumsum(np.arange(10.0)), 'Depth', 'm')


################################################################################


rm = TopoRectilinearMesh2D(xCentres=x, yCentres=y, zCentres=z, heightCentres=height)
values = StatArray(np.arange(rm.nCells, dtype=np.float).reshape(rm.dims), 'Name', 'Units')


################################################################################


plt.figure()
rm.plotGrid(linewidth=1)

################################################################################
# The pcolor function can now be plotted against distance

plt.figure()
rm.pcolor(values, grid=True, xAxis='r')

################################################################################
# And we can plot the x-y co-ordinates

plt.figure()
rm.plotXY()


################################################################################

# x = StatArray(np.arange(3.0), 'Easting', 'm')
# y = StatArray(np.arange(3.0), 'Northing', 'm')
# z = StatArray(np.cumsum(np.arange(4.0)), 'Depth', 'm')
# height = StatArray(np.asarray([1,2,3])*10.0, 'Height', 'm')


################################################################################


# rm = TopoRectilinearMesh2D(xCentres=x, yCentres=y, zCentres=z, heightCentres=height)


################################################################################


rm.toVTK('test', cellData=values)
