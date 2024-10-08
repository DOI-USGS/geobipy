# import h5py
from numpy import array
from h5py import File, Group
from ...base import utilities as cf
#from .. import Error as Err


def find(filename, tag):
    """Find the locations of all groups with 'tag' in their path.

    Parameters
    ----------
    filename : str
        HDF5 file name
    tag : str
        Sub string that appears in the group name.

    Returns
    -------
    out : list
        List of paths into the HDF5 file.

    """
    def find_inner(hdf_file, tag):
        def h5py_iterator(g, tag, prefix=''):
            for key in g.keys():
                item = g[key]
                path = f'{prefix}/{key}'
                if tag in path: # test for dataset
                    yield item, path
                else:
                    if isinstance(item, Group): # test for group (go down)
                        yield from h5py_iterator(item, tag, path)

        for _, path in h5py_iterator(f, tag):
            yield path

    locs = []
    with File(filename, 'r') as f:
        for path in find_inner(f, tag):
            locs.append(path)
    return locs


def read_groups_with_tag(filename, tag, index=None, **kwargs):
    """Reads all groups with 'tag' in their path into memory.

    Parameters
    ----------
    filename : str
        HDF5 file name
    tag : str
        Sub string that appears in the group name.

    Returns
    -------
    out : list
        List of geobipy classes.

    """
    locs = find(filename, tag)
    classes = []
    with File(filename, 'r') as f:
        for loc in locs:
            classes.append(read_item(f[loc], index=index, **kwargs))
    return classes

def read_all(fName):
    """Reads all the entries written to a HDF file

    Iterates through the highest set of keys in the hdf5 file, and reads each one to a list. If each entry has an attached .readHdf procedure, that will be used to read in an object (Those objects imported at the top of this file can be successfully read in using this attached procedure.) If an entry is a numpy array, that will be the return type.  This function will read in the entire file! Use this with caution if you are using large files.

    Parameters
    ----------
    fName : str
        A path and/or file name.

    Returns
    -------
    out : list
        A list of the read in items from the hdf5 file.

    """
    items = {}  # Empty list of items
    with File(fName, 'r') as hf:
        for key in list(hf.keys()):
            grp = hf.get(key)  # Get the name of the item
            tmp = read_item(grp)
            items[key] = tmp  # Add the object to the list.
    return items

def readKeyFromFiles(fNames, groupName, key, index=None, **kwargs):
    """Reads in the keys from multiple files

    Iterates over filenames, group names, and keys and reads them from a HDF5 file

    Parameters
    ----------
    fNames : str or list of str
        The path(s) and/or file name(s)
    groupName : str or list of str
        The group(s) path within the hdf5 file(s) to read from. i.e. '/group1/group1a'
    key : str or list of str
        The key(s) in the group to read
    index : slice, optional
        Specifies the index'th entry of the data to return. If the group was created using a createHDF procedure in parallel with the nRepeats option, index specifies the index'th entry from which to read the data.

    Other Parameters
    ----------------
    Any other parameters in **kwargs are optional but may be necessary if an object's .fromHDF() procedure requires extra arguments. Refer to the object you wish to read in to determine whether extra arguments are necessary.

    Returns
    -------
    out : object or list
        Returns the read in entries as a list if there are multiple or as a single object if there is only one.

    """
    items = []
    if isinstance(fNames, str):
        fNames = [fNames]
    for f in fNames:
        with File(f, 'r') as hf:
            items.append(readKeyFromFile(hf, f, groupName, key, index=index, **kwargs))
    if (len(items) == 1): items = items[0] # Return unlisted item if single
    return items

def readKeyFromFile(h5obj, fName, groupName, key, index=None, **kwargs):
    """Reads in the keys from a file

    Iterates over group names and keys and reads them from a HDF5 file

    Parameters
    ----------
    h5obj : h5py._hl.files.File or h5py._hl.group.Group
        An opened hdf5 handle or a h5py group object
    fName : str
        The path and/or file name to the file that was opened
    groupName : str or list of str
        The group(s) path within the hdf5 file to read from. i.e. '/group1/group1a'
    key : str or list of str
        The key(s) in the group to read
    index : slice, optional
        Specifies the index'th entry of the data to return. If the group was created using a createHDF procedure in parallel with the nRepeats option, index specifies the index'th entry from which to read the data.

    Other Parameters
    ----------------
    Any other parameters in **kwargs are optional but may be necessary if an object's .fromHDF() procedure requires extra arguments. Refer to the object you wish to read in to determine whether extra arguments are necessary.

    Returns
    -------
    out : object or list
        Returns the read in entries as a list if there are multiple or as a single object if there is only one.

    """
    items = []
    if isinstance(groupName, str):
        groupName = [groupName]
    if isinstance(key, str):
        key = [key]

    for g in groupName:
        for k in key:
            h = g + '/' + k
            grp = h5obj.get(h)
            assert (not grp is None), ValueError('Could not read '+h+' from file '+fName)
            tmp = read_item(grp, index=index, **kwargs)
            items.append(tmp)

    if (len(items) == 1): items = items[0] # Return unlisted item if single
    return items


def read_item(h5obj, index=None, **kwargs):
    """Read an object from a HDF file

    This function provides a flexible way to read in either a numpy hdf5 entry, or an object in this package.  The objects in this package may have an attached .createHdf and writeHdf procedure.  If so, this function will read in those objects and return that object.  If the entry is instead a numpy array, a numpy array will be returned.

    Parameters
    ----------
    hObj : h5py._hl.dataset.Dataset or h5py._hl.group.Group
        A h5py object from which to read entries.
    index : slice, optional
        Specifies the index'th entry of the data to return. If the group was created using a createHDF procedure in parallel with the nRepeats option, index specifies the index'th entry from which to read the data.

    Other Parameters
    ----------------
    Any other parameters in **kwargs are optional but may be necessary if an object's .fromHDF() procedure requires extra arguments. Refer to the object you wish to read in to determine whether extra arguments are necessary.

    Returns
    -------
    out : object or numpy.ndarray
        An object that has a .fromHdf() procedure or a numpy array of the returned variable.

    """
    from ...classes.core.DataArray import DataArray
    from ...classes.statistics.StatArray import StatArray
    from ...classes.statistics.Histogram import Histogram
    from ...classes.data.datapoint.FdemDataPoint import FdemDataPoint
    from ...classes.data.datapoint.TdemDataPoint import TdemDataPoint
    from ...classes.data.datapoint.Tempest_datapoint import Tempest_datapoint
    from ...classes.data.dataset.FdemData import FdemData
    from ...classes.data.dataset.TdemData import TdemData
    from ...classes.data.dataset.TempestData import TempestData
    from ...classes.model.Model import Model
    from ...classes.mesh.RectilinearMesh1D import RectilinearMesh1D
    from ...classes.mesh.RectilinearMesh2D import RectilinearMesh2D
    from ...classes.mesh.RectilinearMesh2D_stitched import RectilinearMesh2D_stitched
    from ...classes.mesh.RectilinearMesh3D import RectilinearMesh3D
    from ...inversion.Inference1D import Inference1D
    from ...inversion.Inference2D import Inference2D
    from ...classes.system.CircularLoop import CircularLoop
    # from ...classes.system.CircularLoops import CircularLoops
    s = h5obj.attrs.get('repr')
    if (not s is None):

        # Put temporary conversions here for items
        if 'StatArray' in s:
            if (not 'dtype=' in s):
                s = s.replace(',np.',',dtype=np.')

        elif '(0.0,0.0,0.0,0.0)' in s:
            s = s.replace('(0.0,0.0,0.0,0.0)', '')
            # s += '()'

        if '()' in s:
            s = s.replace('()', '')

        if 'CircularLoops' in s:
            s = s.replace('CircularLoops', 'CircularLoop')

        item = eval(cf.safeEval(s))
        tmp = item.fromHdf(h5obj, index=index, **kwargs)
        return tmp
    try:
        if (index is None):
            return array(h5obj)
        else:
            return array(h5obj)[index]
    except:
        raise ValueError("Could not read group "+str(h5obj)+" from hdf file. \n Check whether an index must be specified in the read hdf function you are using")
    return None
