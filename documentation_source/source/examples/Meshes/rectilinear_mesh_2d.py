"""
2D Rectilinear Mesh
-------------------
"""

################################################################################

from geobipy import StatArray
from geobipy import RectilinearMesh2D
import matplotlib.pyplot as plt
import numpy as np


################################################################################


x = StatArray(np.arange(10.0), 'Easting', 'm')
y = StatArray(np.arange(10.0), 'Northing', 'm')


################################################################################


rm = RectilinearMesh2D(xCentres=x, yCentres=y)


################################################################################


plt.figure()
rm.plotGrid(flipY=True)


################################################################################


z = StatArray(np.cumsum(np.arange(15.0)), 'Depth', 'm')
rm = RectilinearMesh2D(xCentres=x, yCentres=y, zCentres=z)


################################################################################


plt.figure()
rm.plotGrid(xAxis='r', flipY=True)


################################################################################


a = np.repeat(np.arange(1.0, np.float(rm.x.nCells+1))[:, np.newaxis], rm.z.nCells, 1).T


################################################################################


a

################################################################################
# Compute the mean over an interval for the mesh.

rm.intervalStatistic(a, intervals=[6.8, 12.4], axis=0, statistic='mean')


################################################################################
# Compute the mean over multiple intervals for the mesh.

rm.intervalStatistic(a, intervals=[6.8, 12.4, 20.0, 40.0], axis=0, statistic='mean')


################################################################################


rm.intervalStatistic(a, intervals=[2.8, 4.2], axis=1, statistic='mean')


################################################################################


rm.intervalStatistic(a, intervals=[2.8, 4.2, 5.1, 8.4], axis=1, statistic='mean')


################################################################################


arr = StatArray(np.random.random(rm.shape), 'Name', 'Units')


################################################################################


plt.figure()
rm.pcolor(arr, xAxis='r', grid=True, flipY=True)


################################################################################


plt.figure()
rm.plotXY()


################################################################################


rm.toVTK('test', cellData=StatArray(np.random.randn(z.size, x.size), "Name"))
