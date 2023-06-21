from copy import deepcopy
from numpy import asarray, int32, size, unique
from ..pointcloud.PointCloud3D import PointCloud3D
from ..core import StatArray
from .EmLoop import EmLoop
from abc import ABC, abstractclassmethod

class EmLoops(PointCloud3D, ABC):
    """Defines a loop in an EM system e.g. transmitter or reciever

    This is an abstract base class and should not be instantiated

    EmLoops()
    """
    single = EmLoop

    def __init__(self, x=None, y=None, z=None, elevation=None, orientation=None, moment=None, pitch=None, roll=None, yaw=None, **kwargs):
        super().__init__(x, y, z, elevation, **kwargs)

        # Orientation of the loop dipole
        self.orientation = orientation
        # Dipole moment of the loop
        self.moment = moment
        # Pitch of the loop
        self.pitch = pitch
        # Roll of the loop
        self.roll = roll
        # Yaw of the loop
        self.yaw = yaw

    @property
    def moment(self):
        if size(self._moment) == 0:
            self._moment = StatArray.StatArray(self.nPoints, "Moment", "")
        return self._moment

    @moment.setter
    def moment(self, values):
        if (values is None):
            values = self.nPoints
        else:
            if self.nPoints == 0:
                self.nPoints = size(values)
            assert size(values) == self.nPoints, ValueError("moment must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._moment = deepcopy(values)
                return

        self._moment = StatArray.StatArray(values, "Moment", "")

    @property
    def pitch(self):
        if size(self._pitch) == 0:
            self._pitch = StatArray.StatArray(self.nPoints, "Pitch", "$^{o}$")
        return self._pitch

    @pitch.setter
    def pitch(self, values):
        if (values is None):
            values = self.nPoints
        else:
            if self.nPoints == 0:
                self.nPoints = size(values)
            assert size(values) == self.nPoints, ValueError("pitch must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._pitch = deepcopy(values)
                return

        self._pitch = StatArray.StatArray(values, "Pitch", "$^{o}$")

    @property
    def roll(self):
        if size(self._roll) == 0:
            self._roll = StatArray.StatArray(self.nPoints, "Roll", "$^{o}$")
        return self._roll

    @roll.setter
    def roll(self, values):
        if (values is None):
            values = self.nPoints
        else:
            if self.nPoints == 0:
                self.nPoints = size(values)
            assert size(values) == self.nPoints, ValueError("roll must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._roll = deepcopy(values)
                return

        self._roll = StatArray.StatArray(values, "Roll", "$^{o}$")

    @property
    def size(self):
        return self.nPoints

    @property
    def yaw(self):
        if size(self._yaw) == 0:
            self._yaw = StatArray.StatArray(self.nPoints, "Yaw", "$^{o}$")
        return self._yaw

    @yaw.setter
    def yaw(self, values):
        if (values is None):
            values = self.nPoints
        else:
            if self.nPoints == 0:
                self.nPoints = size(values)
            assert size(values) == self.nPoints, ValueError("yaw must have size {}".format(self.nPoints))
            if (isinstance(values, StatArray.StatArray)):
                self._yaw = deepcopy(values)
                return

        self._yaw = StatArray.StatArray(values, "Yaw", "$^{o}$")

    @property
    def orientation(self):
        if size(self._orientation) == 0:
            self._orientation = StatArray.StatArray(self.nPoints, "Orientation", "")

        tmp = ('x', 'y', 'z')
        return [tmp[i] for i in self._orientation]

    @orientation.setter
    def orientation(self, values):
        if (values is None):
            values = self.nPoints
        else:
            tmp = {'x': 0, 'y':1, 'z':2}
            if self.nPoints == 0:
                self.nPoints = size(values)
            assert size(values) == self.nPoints, ValueError("orientation must have size {}".format(self.nPoints))
            values = asarray([tmp[x.replace(" ", "")] for x in values])

        self._orientation = StatArray.StatArray(values, "Orientation", dtype=int32)

    @property
    def summary(self):
        """Print a summary"""
        msg = super().summary
        msg += "moment:\n{}\n".format("|   "+(self.moment.summary.replace("\n", "\n|   "))[:-4])
        msg += "pitch:\n{}\n".format("|   "+(self.pitch.summary.replace("\n", "\n|   "))[:-4])
        msg += "roll:\n{}\n".format("|   "+(self.roll.summary.replace("\n", "\n|   "))[:-4])
        msg += "yaw:\n{}\n".format("|   "+(self.yaw.summary.replace("\n", "\n|   "))[:-4])
        return msg

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out._orientation = deepcopy(self._orientation, memo=memo)
        out._moment = deepcopy(self.moment, memo=memo)
        out._pitch = deepcopy(self.pitch, memo=memo)
        out._roll = deepcopy(self.roll, memo=memo)
        out._yaw = deepcopy(self.yaw, memo=memo)
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

        if not isinstance(i, slice):
            i = unique(i)

        if self.nPoints == 0:
            i = None

        out._orientation = self._orientation[i]
        out.moment = self.moment[i]
        out.pitch = self.pitch[i]
        out.roll = self.roll[i]
        out.yaw = self.yaw[i]
        return out


    def createHdf(self, parent, name, withPosterior=True, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        grp = super().createHdf(parent, name, withPosterior=True, fillvalue=None)
        self.pitch.createHdf(grp, 'pitch', withPosterior=withPosterior, fillvalue=fillvalue)
        self.roll.createHdf(grp, 'roll', withPosterior=withPosterior, fillvalue=fillvalue)
        self.yaw.createHdf(grp, 'yaw', withPosterior=withPosterior, fillvalue=fillvalue)

        self.moment.createHdf(grp, 'moment', withPosterior=withPosterior, fillvalue=fillvalue)
        self._orientation.createHdf(grp, 'orientation', withPosterior=withPosterior, fillvalue=fillvalue)

        return grp

    def writeHdf(self, parent, name, withPosterior=True):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        super().writeHdf(parent, name, withPosterior)

        grp = parent[name]
        self.pitch.writeHdf(grp, 'pitch')
        self.roll.writeHdf(grp, 'roll')
        self.yaw.writeHdf(grp, 'yaw')
        self.moment.writeHdf(grp, 'moment')
        self._orientation.writeHdf(grp, 'orientation')

    @classmethod
    def fromHdf(cls, grp, **kwargs):
        """ Reads in the object from a HDF file """

        if kwargs.get('index') is not None:
            return cls.single.fromHdf(grp, **kwargs)

        out = super(EmLoops, cls).fromHdf(grp)

        out.pitch = StatArray.StatArray.fromHdf(grp['pitch'])
        out.roll = StatArray.StatArray.fromHdf(grp['roll'])
        out.yaw = StatArray.StatArray.fromHdf(grp['yaw'])
        out.moment = StatArray.StatArray.fromHdf(grp['moment'])
        out._orientation = StatArray.StatArray.fromHdf(grp['orientation'])

        return out

    def Isend(self, dest, world):

        super().Isend(dest, world)

        self.pitch.Isend(dest, world)
        self.roll.Isend(dest, world)
        self.yaw.Isend(dest, world)
        self._orientation.Isend(dest, world)
        self.moment.Isend(dest, world)

    @classmethod
    def Irecv(cls, source, world):
        out = super(EmLoops, cls).Irecv(source, world)

        out._pitch = StatArray.StatArray.Irecv(source, world)
        out._roll = StatArray.StatArray.Irecv(source, world)
        out._yaw = StatArray.StatArray.Irecv(source, world)
        out._orientation = StatArray.StatArray.Irecv(source, world)
        out._moment = StatArray.StatArray.Irecv(source, world)

        return out
