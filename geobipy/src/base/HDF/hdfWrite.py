import numpy as np

def write_nd(arr, h5obj, myName, index=None):
    """Writes a numpy array to a preallocated dataset in a h5py group object

    Parameters
    ----------
    h5obj : h5py._hl.files.File or h5py._hl.group.Group
        A HDF file or group object to write the contents to. The dataset must have already been allocated in the file.
    myName : str
        The name of the h5py dataset key inside the h5py object. e.g. '/group1/group1a/dataset'
    index : slice, optional
        Specifies the index'th entry of the data to return. If the group was created using a createHDF procedure in parallel with the nRepeats option, index specifies the index'th entry from which to read the data.

    """
    if (not index is None):
        assert all([isinstance(x,(slice, tuple, int, np.integer)) for x in index]), ValueError('indices must be an integer or a numpy slice. e.g. np.s_[0:10]')

    nd = np.ndim(arr)
    assert nd <= 6, ValueError('The number of dimensions to write must be <= 6')

    # Pull the group
    grp = h5obj[myName]

    # If the user specifies an index, they should already know dimensions etc. so write and return
    if (not index is None):
        ds = grp[index]
    else:
        ds = grp

    assert np.size(ds.shape) == nd, ValueError("arr is being written to a dataset of different shape")

    # If the value is a scalar, write and return
    if (nd == 0):
        ds[0] = arr
        return

    # Write the entire array to appropriate locations in memory. No way to combine this into a function for N dimensions
    # Automatically fills memory from the beginning index in each dimension.
    print('a',arr)
    print('b',ds)
    if (nd == 1):
        ds[:arr.size] = arr
    print('c',ds)

    s = arr.shape
    if(nd == 2):
        ds[:s[0], :s[1]] = arr

    elif(nd == 3):
        ds[:s[0], :s[1], :s[2]] = arr

    elif(nd == 4):
        ds[:s[0], :s[1], :s[2], :s[3]] = arr

    elif(nd == 5):
        ds[:s[0], :s[1], :s[2], :s[3], :s[4]] = arr

    elif(nd == 6):
        ds[:s[0], :s[1], :s[2], :s[3], :s[4], :s[5]] = arr
