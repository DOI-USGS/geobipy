""" Module containing custom MPI functions """
import pickle
from os import getpid
from time import time
import sys
from numpy.linalg import norm
from numpy import abs, arange, asarray, cumsum, empty, float32, float64, full
from numpy import int32, int64, prod, reshape, unravel_index, s_, size, zeros
from numpy import ndim as npndim
import numpy as np
from ..classes.statistics import StatArray

class world3D(object):

    def __init__(self, shape, world):

        assert world.size >= 8, ValueError("Must have at least 8 chunks for 3D load balancing.")

        target = shape / norm(shape)
        best = None
        bestFit = 1e20
        for i in range(2, int32(world.size/2)+1):
            for j in range(2, int32(world.size/i)):
                k = int32(world.size/(i*j))
                nBlocks = asarray([i, j, k])
                total = prod(nBlocks)

                if total == world.size:
                    fraction = nBlocks / norm(nBlocks)
                    fit = norm(fraction - target)
                    if fit < bestFit:
                        best = nBlocks
                        bestFit = fit


        assert not best is None, Exception("Could not split {} into {} blocks. ".format(shape, world.size))

        self.xStarts, self.xChunkSizes = loadBalance1D_shrinkingArrays(shape[0], best[0])
        self.yStarts, self.ychunkSizes = loadBalance1D_shrinkingArrays(shape[1], best[1])
        self.zStarts, self.zChunkSizes = loadBalance1D_shrinkingArrays(shape[2], best[2])

        self.chunkShape = asarray([self.zChunks.size, self.yChunks.size, self.xChunks.size])
        self.chunkIndex = unravel_index(self.rank, self.chunkShape)

        self.world = world


    @property
    def xIndices(self):
        index = self.chunksIndex[2]
        i0 = self.xStarts[index]
        i1 = i0 + self.xChunkSizes[index]
        return s_[i0:i1]


    @property
    def yIndices(self):
        index = self.chunksIndex[1]
        i0 = self.yStarts[index]
        i1 = i0 + self.yChunkSizes[index]
        return s_[i0:i1]


    @property
    def zIndices(self):
        index = self.chunksIndex[0]
        i0 = self.zStarts[index]
        i1 = i0 + self.zChunkSizes[index]
        return s_[i0:i1]


def print(aStr='', end='\n', **kwargs):
    """Prints the str to sys.stdout and flushes the buffer so that printing is immediate

    Parameters
    ----------
    aStr : str
        A string to print.
    end : str
        string appended after the last value, default is a newline.

    """
    if not isinstance(aStr, str):
        aStr = str(aStr)
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
        print(aStr, end, flush=True)


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


def ordered_print(world, this, title=None):
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
    ordered_print(world, '/ {}'.format(rank + 1, size), "Hello From!")

def loadBalance1D_shrinkingArrays(N, nChunks):
    """Splits the length of an array into a number of chunks. Load balances the chunks in a shrinking arrays fashion.

    Given a length N, split N up into nChunks and return the starting index and size of each chunk.
    After being split equally among the chunks, the remainder is split so that the first remainder
    chunks get +1 in size. e.g. N=10, nChunks=3 would return starts=[0,4,7] chunks=[4,3,3]

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
    chunks = zeros(nChunks, dtype=int32)
    chunks[:] = N / nChunks
    Nmod = int32(N % nChunks)
    chunks[:Nmod] += 1
    starts = cumsum(chunks) - chunks[0]
    if (Nmod > 0):
        starts[Nmod:] += 1
    return starts, chunks


def loadBalance3D_shrinkingArrays(shape, nChunks):
    """Splits three dimensions among nChunks.

    The number of chunks honours the relative difference in the values of shape. e.g. if shape is [600, 600, 300], then the number of chunks will be larger for the
    first two dimensions, and less for the third.
    Once the chunks are obtained, the start indices and chunk sizes for each dimension are returned.

    Parameters
    ----------
    N : array_like
        A 3D shape to split.
    nChunks : int
        The number of chunks to split shape into.

    Returns
    -------
    starts : ndarray of ints
        The starting indices of each chunk.
    chunks : ndarray of ints
        The size of each chunk.

    """

    # Find the "optimal" three product whose prod equals nChunks
    # and whose relative amounts match as closely to shape as possible.

    assert nChunks >= 8, ValueError("Must have at least 8 chunks for 3D load balancing.")

    target = shape / norm(shape)
    best = None
    bestFit = 1e20
    for i in range(2, int32(nChunks/2)+1):
        for j in range(2, int32(nChunks/i)):
            k = int32(nChunks/(i*j))
            nBlocks = asarray([i, j, k])
            total = prod(nBlocks)

            if total == nChunks:
                fraction = nBlocks / norm(nBlocks)
                fit = norm(fraction - target)
                if fit < bestFit:
                    best = nBlocks
                    bestFit = fit


    assert not best is None, Exception("Could not split {} into {} blocks. ".format(shape, nChunks))

    starts0, chunks0 = loadBalance1D_shrinkingArrays(shape[0], best[0])
    starts1, chunks1 = loadBalance1D_shrinkingArrays(shape[1], best[1])
    starts2, chunks2 = loadBalance1D_shrinkingArrays(shape[2], best[2])

    return starts0, starts1, starts2, chunks0, chunks1, chunks2


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
    if tmp == 'int':
        tmp = 'int32'
    world.send(tmp, dest=dest)
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
    tmp = world.recv(source=source)

    if (tmp == 'list'):
        return 'list'
    return eval('np.{}'.format(tmp))  # Return the evaluated string


def Isend(self, dest, world, dtype=None, ndim=None, shape=None):
    """Isend a numpy array. Auto determines data type and shape. Must be accompanied by Irecv on the dest rank.

    """

    # Send the data type
    if dtype is None:
        dtype = _isendDtype(self, dest=dest, world=world)

    assert (not dtype == 'list'), TypeError("Cannot Send/Recv a list")

    # Broadcast the number of dimensions
    if ndim is None:
        ndim = Isend_1int(npndim(self), dest=dest, world=world)

    if (ndim == 0):  # For a single number
        this = full(1, self, dtype=dtype)  # Initialize on each worker
        world.Send(this, dest=dest)  # Broadcast

    elif (ndim == 1):  # For a 1D array
        if shape is None:
            shape = Isend_1int(size(self), dest=dest, world=world)  # Broadcast the array size
        world.Send(self, dest=dest)  # Broadcast

    elif (ndim > 1):  # nD Array
        if shape is None:
            world.Send(asarray(self.shape), dest=dest)  # Broadcast the shape
        world.Send(self, dest=dest)  # Broadcast


def Irecv(source, world, dtype=None, ndim=None, shape=None):
    """Irecv a numpy array. Auto determines data type and shape. Must be accompanied by Isend on the source rank.

    """

    if dtype is None:
        dtype = _irecvDtype(source, world)

    assert not dtype == 'list', TypeError("Cannot Send/Recv a list")

    if ndim is None:
        ndim = Irecv_1int(source, world)

    if (ndim == 0):  # For a single number
        this = empty(1, dtype=dtype)  # Initialize on each worker
        world.Recv(this, source=source)  # Broadcast
        this = this[0]
    elif (ndim == 1): # For a 1D array
        if shape is None:
            shape = Irecv_1int(source=source, world=world)
        this = empty(shape, dtype=dtype)
        world.Recv(this, source=source)
    elif (ndim > 1): # Nd Array
        if shape is None:
            shape = empty(ndim, dtype=int32)
            world.Recv(shape, source=source)
        this = empty(shape, dtype=dtype)
        world.Recv(this, source=source)

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
    this = full(1, self, int64)
    world.Send(this, dest=dest)
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
    this = empty(1, int64)
    req = world.Recv(this, source=source)
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



def Bcast(self, world, root=0, dtype=None, ndim=None, shape=None):
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
    >>>     x=StatArray(arange(10))
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
    if dtype is None:
        dtype = bcastType(self, world, root=root)

    assert dtype != 'list', TypeError("Use MPI.Bcast_list for lists")

    # Broadcast the number of dimensions
    if ndim is None:
        ndim = Bcast_1int(npndim(self), world, root=root)

    if (ndim == 0):  # For a single number
        this = empty(1, dtype=dtype)  # Initialize on each worker
        if (world.rank == root):
            this[0] = self  # Assign on the master
        world.Bcast(this)  # Broadcast
        return this[0]

    if (ndim == 1):  # For a 1D array
        if shape is None:
            shape = Bcast_1int(size(self), world, root=root)  # Broadcast the array size
        this = empty(shape, dtype=dtype)
        if (world.rank == root):  # Assign on the root
            this[:] = self
        world.Bcast(this, root=root)  # Broadcast
        return this

    if (ndim > 1):  # nD Array
        if shape is None:
            shape = Bcast(asarray(self.shape), world, root=root)  # Broadcast the shape
        this = empty(shape, dtype=dtype)
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
        this = full(1, self, int64)
    else:
        this = empty(1, int64)
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
    >>>     x = arange(N)
    >>> else:
    >>>     x = None
    >>> # Scatter the array x among ranks.
    >>> myChunk = myMPI.Scatterv(x, starts, chunks, world, root=0)

    """
    # Brodacast the type
    dtype = bcastType(self, world, root=root)

    assert dtype != 'list', TypeError("Use Scatterv_list for lists!")

    return Scatterv_numpy(self, starts, chunks, dtype, world, axis, root)


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

def data_type_map(data_type):
    from mpi4py import MPI

    if data_type == float32:
        return MPI.FLOAT
    elif data_type == float64:
        return MPI.DOUBLE
    elif data_type == int:
        return MPI.INT

def Scatterv_numpy(self, starts, chunks, dtype, world, axis=0, root=0):
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
    dtype : type
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
    ndim = Bcast_1int(npndim(self), world, root=root)

    if (ndim == 1):  # For a 1D Array
        this = empty(chunks[world.rank], dtype=dtype)
        world.Scatterv([self, chunks, starts, data_type_map(dtype)], this, root=root)
        return this

    # For a 2D Array
    # MPI cannot send and receive arrays of more than one dimension.  Therefore higher dimensional arrays must be unpacked to 1D, and then repacked on the other side.
    if (ndim == 2):
        s = Bcast_1int(size(self, 1 - axis), world, root=root)
        tmpChunks = chunks * s
        tmpStarts = starts * s
        self_unpk = None
        if (world.rank == root):
            if (axis == 0):
                self_unpk = reshape(self, size(self))
            else:
                self_unpk = reshape(self.T, size(self))
        this_unpk = empty(tmpChunks[world.rank], dtype=dtype)
        world.Scatterv([self_unpk, tmpChunks, tmpStarts, data_type_map(dtype)], this_unpk, root=root)
        this = reshape(this_unpk, [chunks[world.rank], s])

        return this.T if axis == 1 else this
