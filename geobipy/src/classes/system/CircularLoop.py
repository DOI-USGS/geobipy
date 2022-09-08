from copy import deepcopy
import numpy as np
from ...base import MPI as myMPI
from .EmLoop import EmLoop
from ..core import StatArray
from ..statistics.Distribution import Distribution

class CircularLoop(EmLoop):
    """Defines a circular loop for EM acquisition systems

    CircularLoop(orient, moment, x, y, z, pitch, roll, yaw, radius)

    Parameters
    ----------
    orient : str
        Orientation of the loop, 'x' or 'z'
    moment : int
        Moment of the loop
    x : float
        X location of the loop relative to an observation location
    y : float
        Y location of the loop relative to an observation location
    z : float
        Z location of the loop relative to an observation location
    pitch : float
        Pitch of the loop
    roll : float
        Roll of the loop
    yaw : float
        Yaw of the loop
    radius : float
        Radius of the loop

    """

    def __init__(self, orient="z", moment=1.0, x=0.0, y=0.0, z=0.0, elevation=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0, **kwargs):
        """ Initialize a loop in an EM system """

        super().__init__(x, y, z, elevation=elevation, pitch=pitch, roll=roll, yaw=yaw, **kwargs)
        # Radius of the loop
        self.radius = radius

    @property
    def area(self):
        return np.pi * self.radius * self.radius

    @property
    def radius(self):
        return self._radius    

    @radius.setter
    def radius(self, value):
        if not isinstance(value, StatArray.StatArray):
            value = np.float64(value)
        # assert isinstance(value, (StatArray.StatArray, float, np.float64)), TypeError("pitch must have type float")
        self._radius = StatArray.StatArray(value, 'Radius', 'm')

    @property
    def summary(self):
        """Print a summary"""
        msg = super().summary
        msg += "radius:\n{}".format("|   "+(self.radius.summary.replace("\n", "\n|   "))[:-4])
        return msg

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out.radius = deepcopy(self.radius, memo=memo)
        return out

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = super().createHdf(parent, name, withPosterior, add_axis, fillvalue)
        if add_axis is not None:
            grp.attrs['repr'] = 'CircularLoops'
        self.radius.createHdf(grp, 'radius', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        
    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        super().writeHdf(parent, name, withPosterior, index)

        grp = parent[name]
        self.radius.writeHdf(grp, 'radius', index=index)

    @classmethod
    def fromHdf(cls, grp, index=None):
        """ Reads in the object from a HDF file """

        out = super(CircularLoop, cls).fromHdf(grp, index)
        out.radius = StatArray.StatArray.fromHdf(grp['radius'], index=index)
        return out

    def Bcast(self, world, root=0):
        """Broadcast using MPI

        Parameters
        ----------
        world : mpi4py.MPI.COMM_WORLD
            An MPI communicator

        Returns
        -------
        out : CircularLoop
            A CircularLoop on each core

        """

        x = self.x.Bcast(world, root)
        y = self.y.Bcast(world, root)
        z = self.z.Bcast(world, root)
        pitch = self.pitch.Bcast(world, root)
        roll = self.roll.Bcast(world, root)
        yaw = self.yaw.Bcast(world, root)

        data = np.asarray([self._orient, self.moment, self.radius], dtype=np.float64)
        tData = myMPI.Bcast(data, world, root=root)

        return CircularLoop(orient=tData[0], moment=tData[1], x=x, y=y, z=z, pitch=pitch, roll=roll, yaw=yaw, radius=tData[2])

    def Isend(self, dest, world):

        super().Isend(dest, world)
        self.radius.Isend(dest, world)

    @classmethod
    def Irecv(cls, source, world):

        out = super(CircularLoop, cls).Irecv(source, world)
        out.radius = StatArray.StatArray.Irecv(source, world)

        return out
