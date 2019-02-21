""" Module containing custom MPI functions """
import numpy as np
import sys
from os import getpid
from time import time
#from ...base.Error import Error as Err


def print(aStr='', end='\n'):
    """Prints the str to sys.stdout and flushes the buffer so that printing is immediate

    Parameters
    ----------
    aStr : str
        A string to print.
    end : str
        string appended after the last value, default is a newline.

    """
    sys.stdout.write(aStr + end)
    sys.stdout.flush()


def rankPrint(world, aStr="", end='\n', rank=0):
    """Prints only from the specified MPI rank

    Parameters
    ----------
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    aStr : str
        A string to print.
    end : str
        string appended after the last value, default is a newline.
    rank : int
        The rank to print from, default is the master rank, 0.

    """
    if (world.rank == rank):
        if not isinstance(aStr, str):
            aStr = str(aStr)
        print(aStr, end)


def banner(world, aStr=None, end='\n', rank=0):
    """Prints a String with Separators above and below

    Parameters
    ----------
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    aStr : str
        A string to print.
    end : str
        string appended after the last value, default is a newline.
    rank : int
        The rank to print from, default is the master rank, 0.

    """
    if (aStr is None):
        return
    rankPrint(world, aStr="=" * 78, end=end, rank=rank)
    rankPrint(world, aStr=aStr, end=end, rank=rank)
    rankPrint(world, aStr="=" * 78, end=end, rank=rank)


def orderedPrint(world, this, title=None):
    """Prints numbers from each rank in order of rank

    This routine will print an item from each rank in order of rank.  
    This routine is SLOW due to lots of communication, but is useful for illustration purposes, or debugging. 
    Do not use this in production code!  The title is used in a banner

    Parameters
    ----------
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    this : array_like
        Variable to print, must exist on every rank in the communicator.
    title : str, optional
        Creates a banner to separate output with a clear indication of what is being written.

    """
    if (world.rank > 0):
        world.send(this, dest=0, tag=14)
    else:
        banner(world, title)
        print('Rank 0 {}'.format(this))
        for i in range(1, world.size):
            tmp = world.recv(source=i, tag=14)
            print("Rank {} {}".format(i, tmp))


def helloWorld(world):
    """Print hello from every rank in an MPI communicator

    Parameters
    ----------
    world : mpi4py.MPI.Comm
        MPI parallel communicator.

    """
    size = world.size
    rank = world.rank
    orderedPrint(world, '/ {}'.format(rank + 1, size), "Hello From!")


def getParallelPrng(world, timeFunction):
    """Generate a random seed using time and the process id

    Returns
    -------
    seed : int
        The seed on each core

    """

    i = getpid()
    t = timeFunction()
    seed = np.int64(np.abs(((t*181)*((i-83)*359))%104729))

    prng = np.random.RandomState(seed)

    return prng


def loadBalance_shrinkingArrays(N, nChunks):
    """Splits the length of an array into a number of chunks. Load balances the chunks in a shrinking arrays fashion.

    Given a length N, split N up into nChunks and return the starting index and size of each chunk. After being split equally among the chunks, the remainder is split so that the first remainder chunks get +1 in size. e.g. N=10, nChunks=3 would return starts=[0,4,7] chunks=[4,3,3]

    Parameters
    ----------
    N : int
        A size to split into chunks.
    nChunks : int
        The number of chunks to split N into.

    Returns
    -------
    starts : ndarray of ints
        The starting indices of each chunk.
    chunks : ndarray of ints
        The size of each chunk.

    """
    chunks = np.zeros(nChunks, dtype=np.int)
    chunks[:] = N / nChunks
    Nmod = np.int(N % nChunks)
    chunks[:Nmod] += 1
    starts = np.cumsum(chunks) - chunks[0]
    if (Nmod > 0):
        starts[Nmod:] += 1
    return starts, chunks


def _isendDtype(value, dest, world):
    """Gets the data type of an object and sends it. 

    Must be used within an if statement. 
    if (world.rank == source): _sendDtype()
    Must be accompanied by _irecvDtype on the dest rank.

    Parameters
    ----------
    value : object 
        For numpy arrays and numpy scalars, a numpy data type will be sent.
        For arbitrary objects, the attached __class__.__name__ will be sent.
        For lists, the data type will be list
    dest : int
        Rank to send to.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.

    Returns
    -------
    out : object
        The data type.

    """
    try:
        tmp = str(value.dtype)  # Try to get the dtype attribute
    except:
        tmp = str(value.__class__.__name__)  # Otherwise use the type finder
    world.isend(tmp, dest=dest)
    return tmp


def _irecvDtype(source, world):
    """Receives a data type. 

    Must be used within an if statement. 
    if (world.rank == dest): _recvDtype()
    Must be accompanied by _isendDtype on the source rank.

    Parameters
    ----------
    self : object
        For numpy arrays and numpy scalars, a numpy data type will be received.
        For arbitrary objects, the attached __class__.__name__ will be received.
        For lists, the data type will be list
    source : int
        Receive from source
    world : mpi4py.MPI.Comm
        MPI parallel communicator.

    Returns
    -------
    out : object
        The data type.

    """
    req = world.irecv(source=source)
    tmp = req.wait()

    if (tmp == 'list'):
        return 'list'
    return eval('np.{}'.format(tmp))  # Return the evaluated string


def Isend(self, dest, world, dType=None, nDim=None, shape=None):
    """Isend a numpy array. Auto determines data type and shape. Must be accompanied by Irecv on the dest rank.

    """

    # Send the data type
    if dType is None:
        dType = _isendDtype(self, dest=dest, world=world)

    assert (not dType == 'list'), TypeError("Cannot Send/Recv a list")

    # Broadcast the number of dimensions
    if nDim is None:
        nDim = Isend_1int(np.ndim(self), dest=dest, world=world)

    if (nDim == 0):  # For a single number
        this = np.full(1, self, dtype=dType)  # Initialize on each worker
        req = world.Isend(this, dest=dest)  # Broadcast

    elif (nDim == 1):  # For a 1D array
        if shape is None:
            shape = Isend_1int(np.size(self), dest=dest, world=world)  # Broadcast the array size
        req = world.Isend(self, dest=dest)  # Broadcast

    elif (nDim > 1):  # nD Array
        if shape is None:
            world.Isend(np.asarray(self.shape), dest=dest)  # Broadcast the shape
        req = world.Isend(self, dest=dest)  # Broadcast

    return req


def Irecv(source, world, dType=None, nDim=None, shape=None):
    """Irecv a numpy array. Auto determines data type and shape. Must be accompanied by Isend on the source rank.

    """

    if dType is None:
        dType = _irecvDtype(source, world)

    assert not dType == 'list', TypeError("Cannot Send/Recv a list")

    if nDim is None:
        nDim = Irecv_1int(source, world)

    if (nDim == 0):  # For a single number
        this = np.empty(1, dtype=dType)  # Initialize on each worker
        req = world.Irecv(this, source=source)  # Broadcast
        req.Wait()
        return this[0]
    elif (nDim == 1): # For a 1D array
        if shape is None:
            shape = Irecv_1int(source=source, world=world)
        this = np.empty(shape, dtype=dType)
        req = world.Irecv(this, source=source)
        req.Wait()
        return this
    elif (nDim > 1): # Nd Array
        if shape is None:
            shape = np.empty(nDim, dtype=np.int)
            req = world.Irecv(shape, source=source)
            req.Wait()
        this = np.empty(shape, dtype=dType)
        req = world.Irecv(this, source=source)
        req.Wait()
        return this


def Isend_1int(self, dest, world):
    """Send a single integer. Must be accompanied by Irecv_1int on the dest rank.

    Parameters
    ----------
    self : int
        The integer to Send.
    dest : int
        Rank to receive
    world : mpi4py.MPI.Comm
        MPI parallel communicator.

    Returns
    -------
    out : int
        The sent integer.

    """

    #Examples
    #--------
    #Given an integer instantiated on the master rank 0, in order to broadcast it, I must also instantiate a variable with the same name on all other ranks.

    #>>> import numpy as np
    #>>> from mpi4py import MPI
    #>>> from geobipy.src.base import MPI as myMPI
    #>>> world = MPI.COMM_WORLD
    #>>> if world.rank == 0:
    #>>>     i = 5
    #>>> # Instantiate on all other ranks before broadcasting
    #>>> else:
    #>>>     i=None
    #>>> i = myMPI.Bcast(i, world)

    #"""
    this = np.full(1, self, np.int64)
    world.Isend(this, dest=dest)
    return this[0]


def Irecv_1int(source, world):
    """Recv a single integer. Must be accompanied by Isend_1int on the source rank.

    Parameters
    ----------
    self : int
        Integer to Recv
    source : int
        Receive from this rank.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.

    Returns
    -------
    out : int
        The received integer.

    """
    this = np.empty(1, np.int64)
    req = world.Irecv(this, source=source)
    req.Wait()
    return this[0]


def IsendToLeft(self, world, wrap=True):
    """ISend an array to the rank left of world.rank.

    """

    dest = world.size - 1 if world.rank == 0 else world.rank - 1
    Isend(self, dest = dest, world = world)


def IsendToRight(self, world, wrap=True):
    """ISend an array to the rank left of world.rank.

    """

    dest = 0 if world.rank == world.size - 1 else world.rank + 1
    Isend(self, dest = dest, world=world)


def IrecvFromRight(world, wrap=True):
    """IRecv an array from the rank right of world.rank.

    """
    source = 0 if world.rank == world.size - 1 else world.rank + 1
    return Irecv(source=source, world=world)


def IrecvFromLeft(world, wrap=True):
    """Irecv an array from the rank left of world.rank.

    """
    source = world.size - 1 if world.rank == 0 else world.rank - 1
    return Irecv(source=source, world=world)
     
    

def Bcast(self, world, root=0, dType=None, nDim=None, shape=None):
    """Broadcast a string or a numpy array

    Broadcast a string or a numpy array from a root rank to all ranks in an MPI communicator. Must be called collectively.
    In order to call this function collectively, the variable 'self' must be instantiated on every rank. See the example section for more details.

    Parameters
    ----------
    self : str or numpy.ndarray
        A string or numpy array to broadcast from root.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    root : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : same type as self
        The broadcast object on every rank.

    Raises
    ------
    TypeError
        If self is a list, tell the user to use the specific Bcast_list function.  While it has less code and seems like it might be faster, MPI actually pickles the list, broadcasts that binary stream, and unpickles on the other side. For a large number of lists, this can take a long time. This way, the user is made aware of the time benefits of using numpy arrays.

    Examples
    --------
    Given a numpy array instantiated on the master rank 0, in order to broadcast it, I must also instantiate a variable with the same name on all other ranks.

    >>> import numpy as np
    >>> from mpi4py import MPI
    >>> from geobipy.src.base import MPI as myMPI
    >>> world = MPI.COMM_WORLD
    >>> if world.rank == 0:
    >>>     x=StatArray(np.arange(10))
    >>> # Instantiate on all other ranks before broadcasting
    >>> else:
    >>>     x=None
    >>> y = myMPI.Bcast(x, world)
    >>>
    >>> # A string example
    >>> if (world.rank == 0):
    >>>     s = 'some string'  # This may have been read in through an input file for production code
    >>> else:
    >>>     s = ''
    >>> s = myMPI.Bcast(s,world)

    """
    if (type(self) == str):
        this = None
        if (world.rank == root):
            this = self
        this = world.bcast(this, root=root)
        return this

    # Broadcast the data type
    if dType is None:
        dType = bcastType(self, world, root=root)

    assert dType != 'list', TypeError("Use MPI.Bcast_list for lists")

    # Broadcast the number of dimensions
    if nDim is None:
        nDim = Bcast_1int(np.ndim(self), world, root=root)

    if (nDim == 0):  # For a single number
        this = np.empty(1, dtype=dType)  # Initialize on each worker
        if (world.rank == root):
            this[0] = self  # Assign on the master
        world.Bcast(this)  # Broadcast
        return this[0]

    if (nDim == 1):  # For a 1D array
        if shape is None:
            shape = Bcast_1int(np.size(self), world, root=root)  # Broadcast the array size
        this = np.empty(shape, dtype=dType)
        if (world.rank == root):  # Assign on the root
            this[:] = self
        world.Bcast(this, root=root)  # Broadcast
        return this

    if (nDim > 1):  # nD Array
        if shape is None:
            shape = Bcast(np.asarray(self.shape), world, root=root)  # Broadcast the shape
        this = np.empty(shape, dtype=dType)
        if (world.rank == root):  # Assign on the root
            this[:] = self
        world.Bcast(this, root=root)  # Broadcast
        return this


def bcastType(self, world, root=0):
    """Gets the type of an object and broadcasts it to every rank in an MPI communicator.

    Adaptively broadcasts the type of an object. Must be called collectively.

    Parameters
    ----------
    self : object
        For numpy arrays and numpy scalars, a numpy data type will be broadcast.
        For arbitrary objects, the attached __class__.__name__ will be broadcast.
        For lists, the data type will be list
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    root : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : object
        The data type broadcast to every rank including the rank broadcast from.

    """
    if (world.rank == root):
        try:
            tmp = str(self.dtype)  # Try to get the dtype attribute
        except:
            tmp = str(self.__class__.__name__)  # Otherwise use the type finder
    else:
        tmp = None  # Initialize tmp on all workers

    tmp = world.bcast(tmp, root=root)  # Bcast out to all
    if (tmp == 'list'):
        return 'list'
    return eval('np.{}'.format(tmp))  # Return the evaluated string


def Bcast_1int(self, world, root=0):
    """Broadcast a single integer

    In order to broadcast scalar values using the faster numpy approach, the value must cast into a 1D ndarray. Must be called collectively.

    Parameters
    ----------
    self : int
        The integer to broadcast.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    root : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : int
        The broadcast integer.

    Examples
    --------
    Given an integer instantiated on the master rank 0, in order to broadcast it, I must also instantiate a variable with the same name on all other ranks.

    >>> import numpy as np
    >>> from mpi4py import MPI
    >>> from geobipy.src.base import MPI as myMPI
    >>> world = MPI.COMM_WORLD
    >>> if world.rank == 0:
    >>>     i = 5
    >>> # Instantiate on all other ranks before broadcasting
    >>> else:
    >>>     i=None
    >>> i = myMPI.Bcast(i, world)

    """
    if (world.rank == root):
        this = np.full(1, self, np.int64)
    else:
        this = np.empty(1, np.int64)
    world.Bcast(this, root=root)
    return this[0]


def Bcast_list(self, world, root=0):
    """Broadcast a list by pickling, sending, and unpickling.  This is slower than using numpy arrays and uppercase (Bcast) mpi4py routines. Must be called collectively.

    Parameters
    ----------
    self : list
        A list to broadcast.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    root : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : list
        The broadcast list on every MPI rank.

    """
    this = world.bcast(self, root=root)
    return this


def Scatterv(self, starts, chunks, world, axis=0, root=0):
    """ScatterV an array to all ranks in an MPI communicator.

    Each rank gets a chunk defined by a starting index and chunk size. Must be called collectively. The 'starts' and 'chunks' must be available on every MPI rank. Must be called collectively. See the example for more details.

    Parameters
    ----------
    self : numpy.ndarray
        A numpy array to broadcast from root.
    starts : array of ints
        1D array of ints with size equal to the number of MPI ranks. Each element gives the starting index for a chunk to be sent to that core. e.g. starts[0] is the starting index for rank = 0.
    chunks : array of ints
        1D array of ints with size equal to the number of MPI ranks. Each element gives the size of a chunk to be sent to that core. e.g. chunks[0] is the chunk size for rank = 0.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    axis : int, optional
        Axis along which to Scatterv to the ranks if self is a 2D numpy array. Default is 0
    root : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : numpy.ndarray
        A chunk of self on each MPI rank with size chunk[world.rank].

    Examples
    --------
    >>> import numpy as np
    >>> from mpi4py import MPI
    >>> from geobipy.src.base import MPI as myMPI
    >>> world = MPI.COMM_WORLD
    >>> # Globally define a size N
    >>> N = 1000
    >>> # On each rank, compute the starting indices and chunk size for the given world.
    >>> starts,chunks=loadBalance_shrinkingArrays(N, world.size)
    >>> # Create an array on the master rank
    >>> if (world.rank == 0):
    >>>     x = np.arange(N)
    >>> else:
    >>>     x = None
    >>> # Scatter the array x among ranks.
    >>> myChunk = myMPI.Scatterv(x, starts, chunks, world, root=0)

    """
    # Brodacast the type
    dType = bcastType(self, world, root=root)

    assert dType != 'list', TypeError("Use Scatterv_list for lists!")

    return Scatterv_numpy(self, starts, chunks, dType, world, axis, root)


def Scatterv_list(self, starts, chunks, world, root=0):
    """Scatterv a list by pickling, sending, receiving, and unpickling.  This is slower than using numpy arrays and uppercase (Scatterv) mpi4py routines. Must be called collectively.

    Parameters
    ----------
    self : list
        A list to scatterv.
    starts : array of ints
        1D array of ints with size equal to the number of MPI ranks. Each element gives the starting index for a chunk to be sent to that core. e.g. starts[0] is the starting index for rank = 0.
    chunks : array of ints
        1D array of ints with size equal to the number of MPI ranks. Each element gives the size of a chunk to be sent to that core. e.g. chunks[0] is the chunk size for rank = 0.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    root : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : list
        A chunk of self on each MPI rank with size chunk[world.rank].

    """
    for i in range(world.size):
        if (i != root):
            if (world.rank == root):
                this = self[starts[i]:starts[i] + chunks[i]]
                world.send(this, dest=i)
            if (world.rank == i):
                this = world.recv(source=root)
                return this
    if (world.rank == root):
        return self[:chunks[root]]


def Scatterv_numpy(self, starts, chunks, dType, world, axis=0, root=0):
    """ScatterV a numpy array to all ranks in an MPI communicator.

    Each rank gets a chunk defined by a starting index and chunk size. Must be called collectively. The 'starts' and 'chunks' must be available on every MPI rank. See the example for more details. Must be called collectively.

    Parameters
    ----------
    self : numpy.ndarray
        A numpy array to broadcast from root.
    starts : array of ints
        1D array of ints with size equal to the number of MPI ranks. Each element gives the starting index for a chunk to be sent to that core. e.g. starts[0] is the starting index for rank = 0.
    chunks : array of ints
        1D array of ints with size equal to the number of MPI ranks. Each element gives the size of a chunk to be sent to that core. e.g. chunks[0] is the chunk size for rank = 0.
    dType : type
        The type of the numpy array being scattered. Must exist on all ranks.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    axis : int, optional
        Axis along which to Scatterv to the ranks if self is a 2D numpy array. Default is 0
    root : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : numpy.ndarray
        A chunk of self on each MPI rank with size chunk[world.rank].

    """
    # Broadcast the number of dimensions
    nDim = Bcast_1int(np.ndim(self), world, root=root)
    if (nDim == 1):  # For a 1D Array
        this = np.empty(chunks[world.rank], dtype=dType)
        world.Scatterv([self, chunks, starts, None], this[:], root=root)
        return this

    # For a 2D Array
    # MPI cannot send and receive arrays of more than one dimension.  Therefore higher dimensional arrays must be unpacked to 1D, and then repacked on the other side.
    if (nDim == 2):
        s = Bcast_1int(np.size(self, 1 - axis), world, root=root)
        tmpChunks = chunks * s
        tmpStarts = starts * s
        self_unpk = None
        if (world.rank == root):
            if (axis == 0):
                self_unpk = np.reshape(self, np.size(self))
            else:
                self_unpk = np.reshape(self.T, np.size(self))
        this_unpk = np.empty(tmpChunks[world.rank], dtype=dType)
        world.Scatterv([self_unpk, tmpChunks, tmpStarts, None], this_unpk, root=root)
        this = np.reshape(this_unpk, [chunks[world.rank], s])
        
        return this.T if axis == 1 else this
