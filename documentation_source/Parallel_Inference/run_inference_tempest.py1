"""
1D Inference of Tempest Data
----------------------------

All plotting in GeoBIPy can be carried out using the 3D inference class

"""
#%%
import matplotlib.pyplot as plt
from geobipy import serial_geobipy
from geobipy import Inference3D
import numpy as np
import os
import shutil

#%%
# Running GeoBIPy to invert data
# ++++++++++++++++++++++++++++++
#
# Define some directories and paths

#%%
# The directory where HDF files will be stored
output_directory = "..//..//..//supplementary//time_domain_inversion//results"
#%%

for filename in os.listdir(output_directory):
    file_path = os.path.join(output_directory, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

#%%
# The parameter file defines the set of user parameters needed to run geobipy.
parameter_file = "tempest_options"
#%%

# # Here are the contents of the user parameter file.
with open(parameter_file, 'r') as f:
    print(f.read())

#%%
# To run geobipy in serial, simply call that function.
# Here we specify index 0 to only carry out a shortened inversion of a single
# data point for time considerations.
# You will notice however that the HDF files are created for multiple lines
# inside the data file.
serial_geobipy(parameter_file, output_directory, index=0)


#%%
# Plotting the results for a single data point
# ++++++++++++++++++++++++++++++++++++++++++++
#
# For the sake of plotting, we refer to a previously completed inversion.
# For space considerations we do not include those HDF files in this repository
# and simply use them for plotting.

#%%
results_3d = Inference3D(directory=output_directory, system_file_path="..//..//..//supplementary//data")
#%%
# We can grab the results for a single index or fiducial
results_1d = results_3d.inference_1d(fiducial=results_3d.lines[3].fiducials[0])

#%%
results_1d.plot()

plt.show()
