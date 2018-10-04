import numpy as np
from ....classes.core.StatArray import StatArray
from ....classes.core.myObject import myObject
from ....base import MPI

class DataSet(myObject):
    """Used to contain predicted data, data, and standard deviations.

    Multiple DataSet classes are used within a Data class when different types are measured.

    DataSet(nPoints, nChannels, units)

    Parameters
    ----------
    nPoints : int
        Number of measurement points
    nChannels : int
        Number of data channels
    units : str
        Units of the data

    """

    def __init__(self, nPoints=1, nChannels=1, units=None):
        """ Initialize the DataSet """
        # StatArray of data
        self.D = StatArray([nPoints, nChannels], "Data", units, order='F')
        # StatArray of Standard Deviations
        self.Std = StatArray(np.ones([nPoints, nChannels]), "Standard Deviation", units, order='F')
        # Create predicted data
        self.P = StatArray([nPoints, nChannels], "Predicted Data", order='F')

        self.nPoints = nPoints
        self.nChannels = nChannels

    def Bcast(self, world, root=0):
        """Broadcast the DataSet using MPI 
        
        Parameters
        ----------
        world : mpi4py.MPI.COMM_WORLD
            MPI communicator
        root : int, optional
            The MPI rank to broadcast from. Default is 0.

        Returns
        -------
        out : geobipy.DataSet
            DataSet broadcast to each core in the communicator
        
        """
        nPoints = MPI.Bcast(self.nPoints, world, root=root)
        nChannels = MPI.Bcast(self.nChannels, world, root=root)
        this = DataSet(nPoints, nChannels)
        this.D = self.D.Bcast(world, root=root)
        this.Std = self.Std.Bcast(world, root=root)
        return this


    def Scatterv(self, starts, chunks, world, axis=0, root=0):
        """Scatterv the DataSet using MPI 
        
        Parameters
        ----------
        starts : array of ints
            1D array of ints with size equal to the number of MPI ranks. Each element gives the starting index for a chunk to be sent to that core. e.g. starts[0] is the starting index for rank = 0.
        chunks : array of ints
            1D array of ints with size equal to the number of MPI ranks. Each element gives the size of a chunk to be sent to that core. e.g. chunks[0] is the chunk size for rank = 0.
        world : mpi4py.MPI.Comm
            The MPI communicator over which to Scatterv.
        axis : int
            This axis is distributed amongst ranks.
        root : int, optional
            The MPI rank to broadcast from. Default is 0.

        Returns
        -------
        out : geobipy.DataSet
            The DataSet distributed amongst ranks.
        
        """
        nPoints = chunks[world.rank]
        nChannels = MPI.Bcast(self.nChannels, world, root=root)
        this = DataSet(nPoints, nChannels)
        this.D = self.D.Scatterv(starts, chunks, world, axis=axis, root=root)
        this.Std = self.Std.Scatterv(starts, chunks, world, axis=axis, root=root)
        return this