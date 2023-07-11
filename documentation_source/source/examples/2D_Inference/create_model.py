"""
Nothing file
"""

import numpy as np

def create_model(model_type):
   from geobipy import StatArray, Distribution, RectilinearMesh1D, RectilinearMesh2D_stitched, Model
   n_points = 119
   zwedge = np.linspace(50.0, 1.0, n_points)
   zdeep = np.linspace(75.0, 500.0, n_points)

   resistivities = {'glacial' : np.r_[100, 10, 30],   # Glacial sediments, sands and tills
                  'saline_clay' : np.r_[100, 10, 1],    # Easier bottom target, uncommon until high salinity clay is 5-10 ish
                  'resistive_dolomites' : np.r_[50, 500, 50],   # Glacial sediments, resistive dolomites, marine shale.
                  'resistive_basement' : np.r_[100, 10, 10000],# Resistive Basement
                  'coastal_salt_water' : np.r_[1, 100, 20],    # Coastal salt water upper layer
                  'ice_over_salt_water' : np.r_[10000, 100, 1] # Antarctica glacier ice over salt water
   }
   conductivity = StatArray(1.0/resistivities[model_type], name="Conductivity", units='$\\frac{S}{m}$')

   # Create distributions for three lithology classes
   lithology_distribution = Distribution('MvLogNormal',
                                    mean=conductivity,
                                    variance=[0.5,0.5,0.5],
                                    linearSpace=True)

   x = RectilinearMesh1D(centres=StatArray(np.arange(n_points, dtype=np.float64), name='x'))
   mesh = RectilinearMesh2D_stitched(3, x=x)
   mesh.nCells[:] = 3
   mesh.y_edges[:, 1] = -zwedge
   mesh.y_edges[:, 2] = -zdeep
   mesh.y_edges[:, 3] = -np.inf
   mesh.y_edges.name, mesh.y_edges.units = 'Elevation', 'm'

   return Model(mesh=mesh, values=np.repeat(lithology_distribution.mean[None, :], n_points, 0))