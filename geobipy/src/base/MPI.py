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

    This routine will print an item from each rank in order of rank.  This routine is SLOW due to lots of communication, but is useful for illustration purposes, or debugging. Do not use this in production code!  The title is used in a banner

    Parameters
    ----------
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    this : array_like
        Variable to print, must exist on every rank in the communicator.
    title : str, optional
        Creates a banner to separate output with a clear indication of what is being written.

    """
    try:
        this.shape
        item = this
    except:
        try:
            dtype = this.dtype
        except:
            dtype = type(this)

        item = np.zeros(1, dtype=dtype) + this

    if (world.rank > 0):
        world.Send(item, dest=0, tag=14)
    else:
        banner(world, title)
        print('0 ' + str(item))
        for i in range(1, world.size):
            tmp = np.empty(item.shape, dtype=item.dtype)
            world.Recv(tmp, source=i, tag=14)
            print(str(i) + ' ' + str(tmp))


def helloWorld(world):
    """Print hello from every rank in an MPI communicator

    Parameters
    ----------
    world : mpi4py.MPI.Comm
        MPI parallel communicator.

    """
    size = world.size
    rank = world.rank
    print('Hello! From Rank :' + str(rank + 1) + ' of ' + str(size))


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


def bcastType(self, world, source=0):
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
    source : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : object
        The data type broadcast to every rank including the rank broadcast from.

    """
    if (world.rank == source):
        try:
            tmp = self.dtype  # Try to get the dtype attribute
        except:
            tmp = self.__class__.__name__  # Otherwise use the type finder
    else:
        tmp = None  # Initialize tmp on all workers

    tmp = world.bcast(tmp, root=source)  # Bcast out to all
    if (str(tmp) == 'list'):
        return 'list'
    tmp = 'np.' + str(tmp)  # Prepend np. to create the numpy type
    return eval(tmp)  # Return the evaluated string


#def Sendrecv(self, fromRank, toRank, world):
#    """ Send and Recieve self from one to another rank
#
#    Parameters
#    ----------
#    self : object
#        An object t
#
#    """
#    if (type(self) == str):
#        if (world.rank == fromRank):
#            this = [self]
#            world.send(this, dest=toRank)
#        if (world.rank == toRank):
#            this = world.recv(source=fromRank)
#            print(this[0])
#            return this[0]


def Bcast(self, world, source=0):
    """Broadcast a string or a numpy array

    Broadcast a string or a numpy array from a source rank to all ranks in an MPI communicator. Must be called collectively.
    In order to call this function collectively, the variable 'self' must be instantiated on every rank. See the example section for more details.

    Parameters
    ----------
    self : str or numpy.ndarray
        A string or numpy array to broadcast from source.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    source : int, optional
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
        if (world.rank == source):
            this = self
        this = world.bcast(this, root=source)
        return this

    # Broadcast the data type
    myType = bcastType(self, world, source=source)

    assert myType != 'list', TypeError("Use MPI.Bcast_list for lists")

    # Broadcast the number of dimensions
    nDim = Bcast_1int(np.ndim(self), world, source=source)
    if (nDim == 0):  # For a single number
        this = np.zeros(1, dtype=myType)  # Initialize on each worker
        if (world.rank == source):
            this[0] = self  # Assign on the master
        world.Bcast(this)  # Broadcast
        return this[0]

    if (nDim == 1):  # For a 1D array
        N = Bcast_1int(np.size(self), world, source=source)  # Broadcast the array size
        if (world.rank == source):  # Assign on the source
            this = np.zeros(N, dtype=myType)
            this[:] = self
        else:  # Initialize on each worker
            this = np.empty(N, dtype=myType)
        world.Bcast(this, root=source)  # Broadcast
        return this

    if (nDim > 1):  # nD Array
        shape = Bcast(np.asarray(self.shape), world, source=source)  # Broadcast the shape
        if (world.rank == source):  # Assign on the source
            this = np.zeros(shape, dtype=myType)
            this[:] = self
        else:  # Initialize on each worker
            this = np.empty(shape, dtype=myType)
        world.Bcast(this, root=source)  # Broadcast
        return this


def Bcast_1int(self, world, source=0):
    """Broadcast a single integer

    In order to broadcast scalar values using the faster numpy approach, the value must cast into a 1D ndarray. Must be called collectively.

    Parameters
    ----------
    self : int
        The integer to broadcast.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    source : int, optional
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
    if (world.rank == source):
        this = np.zeros(1, np.int64) + self
    else:
        this = np.empty(1, np.int64)
    world.Bcast(this, root=source)
    return this[0]


def Bcast_list(self, world, source=0):
    """Broadcast a list by pickling, sending, and unpickling.  This is slower than using numpy arrays and uppercase (Bcast) mpi4py routines. Must be called collectively.

    Parameters
    ----------
    self : list
        A list to broadcast.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    source : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : list
        The broadcast list on every MPI rank.

    """
    this = world.bcast(self, root=source)
    return this


def Scatterv(self, starts, chunks, world, axis=0, source=0):
    """ScatterV an array to all ranks in an MPI communicator.

    Each rank gets a chunk defined by a starting index and chunk size. Must be called collectively. The 'starts' and 'chunks' must be available on every MPI rank. Must be called collectively. See the example for more details.

    Parameters
    ----------
    self : numpy.ndarray
        A numpy array to broadcast from source.
    starts : array of ints
        1D array of ints with size equal to the number of MPI ranks. Each element gives the starting index for a chunk to be sent to that core. e.g. starts[0] is the starting index for rank = 0.
    chunks : array of ints
        1D array of ints with size equal to the number of MPI ranks. Each element gives the size of a chunk to be sent to that core. e.g. chunks[0] is the chunk size for rank = 0.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    axis : int, optional
        Axis along which to Scatterv to the ranks if self is a 2D numpy array. Default is 0
    source : int, optional
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
    >>> myChunk = myMPI.Scatterv(x, starts, chunks, world, source=0)

    """
    # Brodacast the type
    myType = bcastType(self, world, source=source)

    assert myType != 'list', "Use Scatterv_list for lists!"

    return Scatterv_numpy(self, starts, chunks, myType, world, axis, source)


def Scatterv_list(self, starts, chunks, world, source=0):
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
    source : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : list
        A chunk of self on each MPI rank with size chunk[world.rank].

    """
    for i in range(world.size):
        if (i != source):
            if (world.rank == source):
                this = self[starts[i]:starts[i] + chunks[i]]
                world.send(this, dest=i)
            if (world.rank == i):
                this = world.recv(source=source)
                return this
    if (world.rank == source):
        return self[:chunks[source]]


def Scatterv_numpy(self, starts, chunks, myType, world, axis=0, source=0):
    """ScatterV a numpy array to all ranks in an MPI communicator.

    Each rank gets a chunk defined by a starting index and chunk size. Must be called collectively. The 'starts' and 'chunks' must be available on every MPI rank. See the example for more details. Must be called collectively.

    Parameters
    ----------
    self : numpy.ndarray
        A numpy array to broadcast from source.
    starts : array of ints
        1D array of ints with size equal to the number of MPI ranks. Each element gives the starting index for a chunk to be sent to that core. e.g. starts[0] is the starting index for rank = 0.
    chunks : array of ints
        1D array of ints with size equal to the number of MPI ranks. Each element gives the size of a chunk to be sent to that core. e.g. chunks[0] is the chunk size for rank = 0.
    myType : type
        The type of the numpy array being scattered. Must exist on all ranks.
    world : mpi4py.MPI.Comm
        MPI parallel communicator.
    axis : int, optional
        Axis along which to Scatterv to the ranks if self is a 2D numpy array. Default is 0
    source : int, optional
        The MPI rank to broadcast from. Default is 0.

    Returns
    -------
    out : numpy.ndarray
        A chunk of self on each MPI rank with size chunk[world.rank].

    """
    # Broadcast the number of dimensions
    nDim = Bcast_1int(np.ndim(self), world, source=source)
    if (nDim == 1):  # For a 1D Array
        this = np.zeros(chunks[world.rank], dtype=myType)
        world.Scatterv([self, chunks, starts, None], this[:], root=source)
        return this

    # For a 2D Array
    # MPI cannot send and receive arrays of more than one dimension.  Therefore higher dimensional arrays must be unpacked to 1D, and then repacked on the other side.
    if (nDim == 2):
        s = Bcast_1int(np.size(self, 1 - axis), world, source=source)
        tmpChunks = chunks * s
        tmpStarts = starts * s
        self_unpk = None
        if (world.rank == source):
            if (axis == 0):
                self_unpk = np.reshape(self, np.size(self))
            else:
                self_unpk = np.reshape(self.T, np.size(self))
        this_unpk = np.zeros(tmpChunks[world.rank], dtype=myType)
        world.Scatterv([self_unpk, tmpChunks, tmpStarts, None],
                       this_unpk, root=source)
        this = np.reshape(this_unpk, [chunks[world.rank], s])
        if (axis == 1):
            this = this.T
        return this
