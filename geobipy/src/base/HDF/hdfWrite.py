from numpy import ndim, size, integer
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
    assert nd <= 6, ValueError('The number of dimensions to write must be <= 6')

    # Pull the group
    ds = h5obj[myName]

    # assert size(ds.shape) == nd, ValueError("arr is being written to a dataset of different shape")

    # If the value is a scalar, write and return
    if (nd == 0):
        ds[0] = arr
        return

    # Write the entire array to appropriate locations in memory. No way to combine this into a function for N dimensions
    # Automatically fills memory from the beginning index in each dimension.
    if (nd == 1):
        ds[:arr.size] = arr

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
    if ndi == 0:
        if (nda == 0):
            ds[i, 0] = arr

        elif (nda == 1):
            ds[i, :s] = arr

        elif(nda == 2):
            ds[i, :s[0], :s[1]] = arr

        elif(nda == 3):
            ds[i, :s[0], :s[1], :s[2]] = arr

        elif(nda == 4):
            ds[i, :s[0], :s[1], :s[2], :s[3]] = arr

        elif(nda == 5):
            ds[i, :s[0], :s[1], :s[2], :s[3], :s[4]] = arr

        elif(nda == 6):
            ds[i, :s[0], :s[1], :s[2], :s[3], :s[4], :s[5]] = arr

    elif ndi == 1:
        if not isinstance(i, (int, integer)):
            i = i[0]
        if (nda == 1):
            if ndim(ds) == 1:
                ds[i] = arr
            else:
                ds[i, :s] = arr

        elif(nda == 2):
            ds[i, :s[0], :s[1]] = arr

        elif(nda == 3):
            ds[i, :s[0], :s[1], :s[2]] = arr

        elif(nda == 4):
            ds[i, :s[0], :s[1], :s[2], :s[3]] = arr

        elif(nda == 5):
            ds[i, :s[0], :s[1], :s[2], :s[3], :s[4]] = arr

        elif(nda == 6):
            ds[i, :s[0], :s[1], :s[2], :s[3], :s[4], :s[5]] = arr


    elif ndi == 2:
        if (nda == 1):
            ds[i[0], i[1], :s] = arr

        elif(nda == 2):
            ds[i[0], i[1], :s[0], :s[1]] = arr

        elif(nda == 3):
            ds[i[0], i[1], :s[0], :s[1], :s[2]] = arr

        elif(nda == 4):
            ds[i[0], i[1], :s[0], :s[1], :s[2], :s[3]] = arr

        elif(nda == 5):
            ds[i[0], i[1], :s[0], :s[1], :s[2], :s[3], :s[4]] = arr

        elif(nda == 6):
            ds[i[0], i[1], :s[0], :s[1], :s[2], :s[3], :s[4], :s[5]] = arr


    elif ndi == 3:
        if (nda == 1):
            ds[i[0], i[1], i[2], :s] = arr

        elif(nda == 2):
            ds[i[0], i[1], i[2], :s[0], :s[1]] = arr

        elif(nda == 3):
            ds[i[0], i[1], i[2], :s[0], :s[1], :s[2]] = arr

        elif(nda == 4):
            ds[i[0], i[1], i[2], :s[0], :s[1], :s[2], :s[3]] = arr

        elif(nda == 5):
            ds[i[0], i[1], i[2], :s[0], :s[1], :s[2], :s[3], :s[4]] = arr

        elif(nda == 6):
            ds[i[0], i[1], i[2], :s[0], :s[1], :s[2], :s[3], :s[4], :s[5]] = arr
