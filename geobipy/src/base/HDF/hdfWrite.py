import numpy as np

def writeNumpy(arr, h5obj, myName, index=None):
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
            assert isinstance(index,(slice, tuple, int, np.integer)), ValueError('index must be an integer or a numpy slice. e.g. np.s_[0:10]')

    nd = np.ndim(arr)
    assert nd <= 6, ValueError('The number of dimensions to write must be <= 6')

    # If the user specifies an index, they should already know dimensions etc. so write and return
    if (not index is None):
        h5obj[myName][index] = arr
        return

    # If the value is a scalar, write and return
    if (nd == 0):
        h5obj[myName][0] = arr
        return

    # Write the entire array to appropriate locations in memory. No way to combine this into a function for N dimensions
    # Automatically fills memory from the beginning index in each dimension.
    if (nd == 1):
        h5obj[myName][:arr.size] = arr
        return

    s = arr.shape
    if(nd == 2):
        h5obj[myName][:s[0], :s[1]] = arr
        return

    if(nd == 3):
        h5obj[myName][:s[0], :s[1], :s[2]] = arr
        return

    if(nd == 4):
        h5obj[myName][:s[0], :s[1], :s[2], :s[3]] = arr
        return

    if(nd == 5):
        h5obj[myName][:s[0], :s[1], :s[2], :s[3], :s[4]] = arr
        return

    if(nd == 6):
        h5obj[myName][:s[0], :s[1], :s[2], :s[3], :s[4], :s[5]] = arr
        return
