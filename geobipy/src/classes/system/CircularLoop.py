from copy import deepcopy
import numpy as np
from ...base import MPI as myMPI
from .EmLoop import EmLoop
from ...base import utilities as cf
from ..core import StatArray
from ...base.HDF.hdfWrite import write_nd

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

    def __init__(self, orient="z", moment=1.0, x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0):
        """ Initialize a loop in an EM system """
        # Orientation of the loop dipole
        self.orient = orient
        # Dipole moment of the loop
        self._moment = moment
        # Not sure yet
        self._x = x
        # Not sure yet
        self._y = y
        # Not sure yet
        self._z = z
        # Pitch of the loop
        self._pitch = pitch
        # Roll of the loop
        self._roll = roll
        # Yaw of the loop
        self._yaw = yaw
        # Radius of the loop
        self._radius = radius

    @property
    def area(self):
        return np.pi * self.radius * self.radius

    @property
    def moment(self):
        return self._moment

    @property
    def pitch(self):
        return self._pitch

    @property
    def radius(self):
        return self._radius

    @property
    def roll(self):
        return self._roll

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def yaw(self):
        return self._yaw

    @property
    def z(self):
        return self._z


    @property
    def orient(self):
        if self._orient == 0.0:
            return 'x'
        elif self._orient == 1.0:
            return 'y'
        else:
            return 'z'

    @orient.setter
    def orient(self, value):
        if isinstance(value, str):
            assert value in ['x', 'y', 'z'], ValueError("orientation must be 'x', 'y', or 'z'")
            if value == 'x':
                self._orient = 0.0
            elif value == 'y':
                self._orient = 1.0
            else:
                self._orient = 2.0
        else:
            assert value in [0, 0.0, 1, 1.0, 2, 2.0], ValueError("orientation must be 0, 1, or 2")
            self._orient = np.float64(value)

    def __deepcopy__(self, memo={}):
        return CircularLoop(self.orient, self.moment, self.x, self.y, self.z, self.pitch, self.roll, self.yaw, self.radius)


    @property
    def summary(self):
        """Print a summary"""
        msg = ("EmLoop: \n"
               "Orientation: {}\n"
               "Moment: {}\n"
               "X: {}\n"
               "Y: {}\n"
               "Z: {}\n"
               "Pitch: {}\n"
               "Roll: {}\n"
               "Yaw: {}\n"
               "Radius: {}\n").format(self.orient, self.moment, self.x, self.y, self.z, self.pitch, self.roll, self.yaw)
        return msg

    # def hdf_name(self):
    #     """Create a reproducibility string that can be instantiated from a hdf file """
    #     return 'CircularLoop()'

    def createHdf(self, parent, name, nRepeats=None, fillvalue=None, withPosterior=False):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = self.create_hdf_group(parent, name)

        data = StatArray.StatArray(9).createHdf(grp, 'data', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, name, withPosterior=False, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        grp = parent[name]

        data = StatArray.StatArray(np.asarray([self._orient, self.moment, self.x, self.y, self.z, self.pitch, self.roll, self.yaw, self.radius], dtype=np.float64))
        data.writeHdf(grp, 'data', index=index)

    @classmethod
    def fromHdf(cls, grp, index=None):
        """ Reads in the object from a HDF file """

        item = grp['data']

        if not 'repr' in item.attrs:
            tmp = np.asarray(item[index, :])
        else:
            tmp = StatArray.StatArray.fromHdf(item, index=index)

        return cls(*tmp)


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

        data = np.asarray([self._orient, self.moment, self.x, self.y, self.z, self.pitch, self.roll, self.yaw, self.radius], dtype=np.float64)
        tData = myMPI.Bcast(data, world, root=root)

        return CircularLoop(*tData)


    def Isend(self, dest, world):
        data = np.asarray([self._orient, self.moment, self.x, self.y, self.z, self.pitch, self.roll, self.yaw, self.radius], dtype=np.float64)
        myMPI.Isend(data, dest=dest, ndim=1, shape=(9, ), dtype=np.float64, world=world)


    def Irecv(self, source, world):
        data = myMPI.Irecv(source=source, ndim=1, shape=(9, ), dtype=np.float64, world=world)
        return CircularLoop(*data)


    def __str__(self):
        """ Define print(self) """
        return 'CircularLoop("{0}",{1},{2},{3},{4},{5},{6},{7},{8})'.format(
            self.orient, self.moment,
            self.x,      self.y,    self.z,
            self.pitch,  self.roll, self.yaw, self.radius)

















