from copy import deepcopy
from numpy import asarray, float64, pi, size, unique

from .CircularLoop import CircularLoop
from ...base import MPI as myMPI
from .EmLoops import EmLoops
from ..core import StatArray
from ..statistics.Distribution import Distribution

class CircularLoops(EmLoops):
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
    single = CircularLoop

    def __init__(self, orientation=None, moment=None, x=None, y=None, z=None, elevation=None, pitch=None, roll=None, yaw=None, radius=None, **kwargs):
        """ Initialize a loop in an EM system """

        super().__init__(x, y, z, elevation=elevation, orientation=orientation, moment=moment, pitch=pitch, roll=roll, yaw=yaw, **kwargs)
        # Radius of the loop
        self.radius = radius

    @property
    def area(self):
        return pi * self.radius * self.radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, values):
        if (values is None):
            values = self.nPoints
        else:
            if self.nPoints == 0:
                self.nPoints = size(values)
            assert size(values) == self.nPoints, ValueError("radius must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._radius = deepcopy(values)
                return

        self._radius = StatArray.StatArray(values, "Radius", "m")

    @property
    def summary(self):
        """Print a summary"""
        msg = ("{}"
                "Radius: {}\n"
               ).format(super().summary, self.radius)
        return msg

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out._radius = deepcopy(self.radius, memo=memo)

        return out

    def __getitem__(self, i):
        """Define get item

        Parameters
        ----------
        i : ints or slice
            The indices of the points in the pointcloud to return

        out : geobipy.PointCloud3D
            The potentially smaller point cloud

        """
        out = super().__getitem__(i)
        i = unique(i)
        out._radius = self.radius[i]
        return out

    def createHdf(self, parent, myName, withPosterior=True, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = super().createHdf(parent, myName, withPosterior, fillvalue)
        self.radius.createHdf(grp, 'radius', withPosterior=withPosterior, fillvalue=fillvalue)


    def writeHdf(self, parent, name, withPosterior):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        super().writeHdf(parent, name, withPosterior)

        grp = parent[name]
        self.radius.writeHdf(grp, 'radius', withPosterior)

    @classmethod
    def fromHdf(cls, grp, **kwargs):
        """ Reads in the object from a HDF file """

        if kwargs.get('index') is not None:
            return cls.single.fromHdf(grp, **kwargs)

        out = super(CircularLoops, cls).fromHdf(grp)
        out.radius = StatArray.StatArray.fromHdf(grp, 'radius')

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
        out = super(CircularLoops, cls).Irecv(source, world)
        out._radius = StatArray.StatArray.Irecv(source, world)
        return out
