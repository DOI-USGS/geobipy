"""
3D Rectilinear Model
-------------------
This 3D rectilinear model defines a grid with straight cell boundaries.

"""

#%%
from geobipy import StatArray
from geobipy import RectilinearMesh3D
from geobipy import Model
import matplotlib.pyplot as plt
import numpy as np
import h5py


#%%
# Specify some cell centres in x and y
x = StatArray(np.arange(10.0), 'Easting', 'm')
y = StatArray(np.arange(11.0), 'Northing', 'm')
z = StatArray(np.arange(12.0), 'Depth', 'm')

xx, yy = np.meshgrid(x.internalEdges(), y.internalEdges())
height = StatArray(np.sin(np.sqrt(xx ** 2.0 + yy ** 2.0)), "Height")

rm = RectilinearMesh3D(xEdges=x, yEdges=y, zEdges=z, height=height)

values = StatArray(np.repeat(height[None, :, :], rm.z.nCells, 0), "Values")

mod = Model(mesh=rm, values = values)

################################################################################
# We can plot the mesh in 3D!
pv = mod.pyvista_mesh()

mod.to_vtk('Model3D.vtk')

with h5py.File('Model3D.h5', 'w') as f:
    mod.toHdf(f, 'model')

with h5py.File('Model3D.h5', 'r') as f:
    mod2 = Model().fromHdf(f['model'])


mod[:, :, 5]
mod.to_vtk('slice.vtk')

