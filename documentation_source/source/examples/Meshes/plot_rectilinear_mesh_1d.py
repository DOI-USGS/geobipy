"""
1D Rectilinear Mesh
-------------------
"""
#%%
from geobipy import StatArray
from geobipy import RectilinearMesh1D
import matplotlib.pyplot as plt
import numpy as np


#%%
# Instantiate a new 1D rectilinear mesh by specifying cell centres or edges.
# Here we use edges
x = StatArray(np.cumsum(np.arange(0.0, 10.0)), 'Depth', 'm')

################################################################################
rm = RectilinearMesh1D(cellEdges=x)


################################################################################
print(rm.cellCentres)

################################################################################
print(rm.cellEdges)

################################################################################
print(rm.internalCellEdges)

################################################################################
print(rm.cellWidths)

################################################################################
# Get the cell indices
print(rm.cellIndex(np.r_[1.0, 5.0, 20.0]))

################################################################################
# We can plot the grid of the mesh
plt.figure()
_ = rm.plotGrid(flipY=True)


################################################################################
# Or Pcolor the mesh showing. An array of cell values is used as the colour.
plt.figure()
arr = StatArray(np.random.randn(rm.nCells), "Name", "Units")
_ = rm.pcolor(arr, grid=True, flipY=True)


#%%
# Instantiate a new 1D rectilinear mesh by specifying cell centres or edges.
# Here we use edges
x = StatArray(np.logspace(-3, 3, 10), 'Depth', 'm')

################################################################################
rm = RectilinearMesh1D(cellEdges=x, log=10)


################################################################################
# Access property describing the mesh
print(rm.cellCentres)

################################################################################
print(rm.cellEdges)

################################################################################
print(rm.internalCellEdges)

################################################################################
print(rm.cellWidths)

################################################################################
# Get the cell indices
print(rm.cellIndex(np.r_[0.03, 5.0, 200.0]))

################################################################################
# We can plot the grid of the mesh
plt.figure()
_ = rm.plotGrid(flipY=True)


################################################################################
# Or Pcolor the mesh showing. An array of cell values is used as the colour.
plt.figure()
arr = StatArray(np.random.randn(rm.nCells), "Name", "Units")
_ = rm.pcolor(arr, grid=True, flipY=True)

