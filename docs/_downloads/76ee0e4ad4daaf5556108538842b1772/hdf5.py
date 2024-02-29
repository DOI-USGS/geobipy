"""
Using HDF5 within GeoBIPy
-------------------------

Inference for large scale datasets in GeoBIPy is handled using MPI and distributed memory systems.
A common bottleneck with large parallel algorithms is the input output of information to disk.
We use HDF5 to read and write data in order to leverage the parallel capabililties of the HDF5 API.

Each object within GeoBIPy has a create_hdf, write_hdf, and read_hdf routine.

"""
import numpy as np
import h5py
from geobipy import StatArray

#%%
# StatArray

# Instantiate a StatArray
x = StatArray(np.arange(10.0), name = 'an Array', units = 'some units')

# Write the StatArray to a HDF file.
with h5py.File("x.h5", 'w') as f:
    x.toHdf(f, "x")

# Read the StatArray back in.
with h5py.File("x.h5", 'r') as f:
    y = StatArray.fromHdf(f, 'x')

print('x', x)
print('y', y)

#%%
# There are actually steps within the "toHdf" function.
# First, space is created within the HDF file and second, the data is written to that space
# These functions are split because during the execution of a parallel enabled program,
# all the space within the HDF file needs to be allocated before we can write to the file
# using multiple cores.

# Write the StatArray to a HDF file.
with h5py.File("x.h5", 'w') as f:
    x.createHdf(f, "x")
    x.writeHdf(f, "x")

# Read the StatArray back in.
with h5py.File("x.h5", 'r') as f:
    y = StatArray.fromHdf(f, 'x')

print('x', x)
print('y', y)

#%%
# The create and write HDF methods also allow extra space to be allocated so that
# the extra memory can be written later, perhaps by multiple cores.
# Here we specify space for 2 arrays, the memory is stored contiguously as a numpy array.
# We then write to only the first index.

# Write the StatArray to a HDF file.
with h5py.File("x.h5", 'w') as f:
    x.createHdf(f, "x", nRepeats=2)
    x.writeHdf(f, "x", index=0)

# Read the StatArray back in.
with h5py.File("x.h5", 'r') as f:
    y = StatArray.fromHdf(f, 'x', index=0)

print('x', x)
print('y', y)


#%%
# The duplication can also be a shape.

# Write the StatArray to a HDF file.
with h5py.File("x.h5", 'w') as f:
    x.createHdf(f, "x", nRepeats=(2, 2))
    x.writeHdf(f, "x", index=(0, 0))

# Read the StatArray back in.
with h5py.File("x.h5", 'r') as f:
    y = StatArray.fromHdf(f, 'x', index=(0, 0))

print('x', x)
print('y', y)

#%%
# Similarly, we can duplicate a 2D array with an extra 2D duplication

x = StatArray(np.random.randn(2, 2), name = 'an Array', units = 'some units')
# Write the StatArray to a HDF file.
with h5py.File("x.h5", 'w') as f:
    x.createHdf(f, "x", nRepeats=(2, 2))
    x.writeHdf(f, "x", index=(0, 0))

# Read the StatArray back in.
with h5py.File("x.h5", 'r') as f:
    y = StatArray.fromHdf(f, 'x', index=(0, 0))

print('x', x)
print('y', y)