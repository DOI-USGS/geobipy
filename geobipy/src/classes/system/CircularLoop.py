from copy import deepcopy
from numpy import asarray, float64, pi, size
from ...base import MPI as myMPI
from .EmLoop import EmLoop
from ..statistics import StatArray
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

    __slots__ = ('_radius')

    def __init__(self, x=None, y=None, z=None, elevation=None, orientation=None, moment=None, pitch=None, roll=None, yaw=None, radius=None, **kwargs):
        """ Initialize a loop in an EM system """

        super().__init__(x=x, y=y, z=z, elevation=elevation, orientation=orientation, moment=moment, pitch=pitch, roll=roll, yaw=yaw, **kwargs)
        self._radius = StatArray.StatArray(self._nPoints, "Radius", "m")

        # Radius of the loop
        self.radius = radius

    @property
    def area(self):
        return pi * self.radius * self.radius

    @property
    def radius(self):
        if size(self._radius) == 0:
            self._radius = StatArray.StatArray(self.nPoints, "Radius", "m")
        return self._radius

    @radius.setter
    def radius(self, values):
        if (values is not None):
            self.nPoints = size(values)

            if self._radius.size != self._nPoints:
                self._radius = StatArray.StatArray(values, "Radius", "m")
                return

            self._radius[:] = values

    @property
    def summary(self):
        """Print a summary"""
        msg = super().summary
        msg += "radius:\n{}".format("|   "+(self.radius.summary.replace("\n", "\n|   "))[:-4])
        return msg

    @property
    def address(self):
        out = super().address
        for x in [self.radius]:
            out = hstack([out, x.address.flatten()])

        return out

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out._radius = deepcopy(self.radius, memo=memo)
        return out

    def append(self, other):

        super().append(other)
        self._radius = self._radius.append(other.radius)

        return self

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

        data = asarray([self._orient, self.moment, self.radius], dtype=float64)
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
