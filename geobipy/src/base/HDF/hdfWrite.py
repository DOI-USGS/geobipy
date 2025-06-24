from numpy import atleast_1d, ndim, s_, size, integer
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

    if index is None:
        return write_nd_nonindexed(arr, h5obj, myName)
    else:
        return write_nd_indexed(arr, h5obj, myName, index)



def write_nd_nonindexed(arr, h5obj, myName):

    nd = ndim(arr)
    # assert nd <= 6, ValueError('The number of dimensions to write must be <= 6')

    # Pull the group
    ds = h5obj[myName]

    # assert size(ds.shape) == nd, ValueError("arr is being written to a dataset of different shape")

    # If the value is a scalar, write and return
    if (nd == 0):
        ds[0] = arr
        return

    # Write the entire array to appropriate locations in memory. No way to combine this into a function for N dimensions
    # Automatically fills memory from the beginning index in each dimension.

    slic = tuple([s_[:size(arr, axis=i)] for i in range(nd)])
    ds[slic] = arr

def write_nd_indexed(arr, h5obj, myName, index):

    # Pull the group
    ds = h5obj[myName]

    # assert size(ds.shape) == nd, ValueError("arr is being written to a dataset of different shape")
    ndi = size(index)
    s = arr.size
    nda = ndim(arr)
    if nda >= 2:
        s = arr.shape

    assert nda <= 6, ValueError('Can only write arr of 6 dimensions or less')
    assert ndi <= 3, ValueError("Can only use indices of 3 dimensions or less")

    if (nda == ndi) and (nda == ds.ndim):
        ds[index] = arr
        return

    assert ndi + nda == ds.ndim or (ndi == 1 and nda == 1), ValueError("index must have {} dimensions".format(ds.ndim - nda))

    i = index
    if (nda == 0):
        ds[i, 0] = arr
    else:
        i = i.item() if i.size == 1 else i
        slic = tuple([i] + [s_[:size(arr, axis=j)] for j in range(nda)])
        ds[slic] = arr
