import numpy as np
from copy import deepcopy
from ..core import StatArray
from ..pointcloud.Point import Point
from ..mesh.RectilinearMesh1D import RectilinearMesh1D
from ..statistics.Histogram import Histogram
from ..statistics.Distribution import Distribution
from abc import ABC

class EmLoop(Point, ABC):
    """Defines a loop in an EM system e.g. transmitter or reciever

    This is an abstract base class and should not be instantiated

    EmLoop()


    """

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=0.0, orientation='z', moment=1.0, pitch=0.0, roll=0.0, yaw=0.0, **kwargs):

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
        return self._moment

    @moment.setter
    def moment(self, value):
        if not isinstance(value, StatArray.StatArray):
            value = np.float64(value)
        # assert isinstance(value, (StatArray.StatArray, float, np.float64)), TypeError("pitch must have type float")
        self._moment = StatArray.StatArray(value, 'Moment')

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        if not isinstance(value, StatArray.StatArray):
            value = np.float64(value)

        if '_pitch' in self.__dict__:
            self._pitch[0] = value
        else:
            self._pitch = StatArray.StatArray(value, 'Pitch', '$^{o}$')

    @property
    def roll(self):
        return self._roll

    @roll.setter
    def roll(self, value):
        if not isinstance(value, StatArray.StatArray):
            value = np.float64(value)
        if '_roll' in self.__dict__:
            self._roll[0] = value
        else:
            self._roll = StatArray.StatArray(value, 'Roll', '$^{o}$')

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        if not isinstance(value, StatArray.StatArray):
            value = np.float64(value)

        if '_yaw' in self.__dict__:
            self._yaw[0] = value
        else:
            self._yaw = StatArray.StatArray(value, 'Yaw', '$^{o}$')

    @property
    def orientation(self):
        if self._orientation == 0:
            return 'x'
        elif self._orientation == 1:
            return 'y'
        else:
            return 'z'

    @orientation.setter
    def orientation(self, value):
        if isinstance(value, str):
            assert value in ['x', 'y', 'z'], ValueError("orientation must be 'x', 'y', or 'z'")
            if value == 'x':
                value = 0
            elif value == 'y':
                value = 1
            else:
                value = 2

        assert 0 <= value <= 2, ValueError("orientation must be 0, 1, or 2")
        self._orientation = StatArray.StatArray(1, dtype=np.int8) + value

    @property
    def summary(self):
        """Print a summary"""
        msg = super().summary

        msg += "orientation:\n{}\n".format("|   "+(self.orientation.replace("\n", "\n|   ")))
        msg += "moment:\n{}".format("|   "+(self.moment.summary.replace("\n", "\n|   "))[:-4])
        msg += "pitch:\n{}".format("|   "+(self.pitch.summary.replace("\n", "\n|   "))[:-4])
        msg += "roll:\n{}".format("|   "+(self.roll.summary.replace("\n", "\n|   "))[:-4])
        msg += "yaw:\n{}".format("|   "+(self.yaw.summary.replace("\n", "\n|   "))[:-4])

        return msg

    def __deepcopy__(self, memo={}):
        out = super().__deepcopy__(memo)
        out._orientation = deepcopy(self._orientation, memo=memo)
        out._moment = deepcopy(self.moment, memo=memo)
        out._pitch = deepcopy(self.pitch, memo=memo)
        out._roll = deepcopy(self.roll, memo=memo)
        out._yaw = deepcopy(self.yaw, memo=memo)
        return out

    def set_priors(self, x_prior=None, y_prior=None, z_prior=None, pitch_prior=None, roll_prior=None, yaw_prior=None, kwargs={}):

        super().set_priors(x_prior, y_prior, z_prior, kwargs=kwargs)

        if pitch_prior is not None:
            self.pitch.prior = pitch_prior
        if roll_prior is not None:
            self.roll.prior = roll_prior
        if yaw_prior is not None:
            self.yaw.prior = yaw_prior

    def set_proposals(self, x_proposal=None, y_proposal=None, z_proposal=None, pitch_proposal=None, roll_proposal=None, yaw_proposal=None, kwargs={}):

        super().set_proposals(x_proposal, y_proposal, z_proposal, kwargs=kwargs)

        if pitch_proposal is None:
            if kwargs.get('solve_pitch', False):
                pitch_proposal = Distribution('Normal', self.pitch.value, kwargs['pitch_proposal_variance'], prng=kwargs['prng'])

        self.pitch.proposal = pitch_proposal

        if roll_proposal is None:
            if kwargs.get('solve_roll', False):
                roll_proposal = Distribution('Normal', self.roll.value, kwargs['roll_proposal_variance'], prng=kwargs['prng'])

        self.roll.proposal = roll_proposal

        if yaw_proposal is None:
            if kwargs.get('solve_yaw', False):
                yaw_proposal = Distribution('Normal', self.yaw.value, kwargs['yaw_proposal_variance'], prng=kwargs['prng'])

        self.yaw.proposal = yaw_proposal

    def reset_posteriors(self):
        super().reset_posteriors()
        self.pitch.reset_posteriors()
        self.roll.reset_posteriors()
        self.yaw.reset_posteriors()

    def set_posteriors(self):

        super().set_posteriors()

        self.set_pitch_posterior()
        self.set_roll_posterior()
        self.set_yaw_posterior()

    def set_pitch_posterior(self):
        """

        """
        if self.pitch.hasPrior:
            mesh = RectilinearMesh1D(edges=StatArray.StatArray(self.pitch.prior.bins(199), name=self.pitch.name, units=self.pitch.units), relativeTo=self.pitch)
            self.pitch.posterior = Histogram(mesh=mesh)

    def set_roll_posterior(self):
        """

        """
        if self.roll.hasPrior:
            mesh = RectilinearMesh1D(edges = StatArray.StatArray(self.roll.prior.bins(), name=self.roll.name, units=self.roll.units), relativeTo=self.roll)
            self.pitch.posterior = Histogram(mesh=mesh)

    def set_yaw_posterior(self):
        """

        """
        if self.yaw.hasPrior:
            mesh = RectilinearMesh1D(edges=StatArray.StatArray(self.yaw.prior.bins(), name=self.yaw.name, units=self.yaw.units), relativeTo=self.yaw)
            self.pitch.posterior = Histogram(mesh=mesh)

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        grp = super().createHdf(parent, name, withPosterior=True, add_axis=add_axis, fillvalue=fillvalue)
        self.pitch.createHdf(grp, 'pitch', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.roll.createHdf(grp, 'roll', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.yaw.createHdf(grp, 'yaw', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        self.moment.createHdf(grp, 'moment', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self._orientation.createHdf(grp, 'orientation', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        return grp

    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        super().writeHdf(parent, name, withPosterior, index)

        grp = parent[name]
        self.pitch.writeHdf(grp, 'pitch', index=index)
        self.roll.writeHdf(grp, 'roll', index=index)
        self.yaw.writeHdf(grp, 'yaw', index=index)
        self.moment.writeHdf(grp, 'moment', index=index)
        self._orientation.writeHdf(grp, 'orientation', index=index)

    @classmethod
    def fromHdf(cls, grp, index=None):
        """ Reads in the object from a HDF file """

        out = super(EmLoop, cls).fromHdf(grp, index)

        out.pitch = StatArray.StatArray.fromHdf(grp['pitch'], index=index)
        out.roll = StatArray.StatArray.fromHdf(grp['roll'], index=index)
        out.yaw = StatArray.StatArray.fromHdf(grp['yaw'], index=index)
        out.moment = StatArray.StatArray.fromHdf(grp['moment'], index=index)
        out._orientation = StatArray.StatArray.fromHdf(grp['orientation'], index=index)

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
        out = super(EmLoop, cls).Irecv(source, world)

        out._pitch = StatArray.StatArray.Irecv(source, world)
        out._roll = StatArray.StatArray.Irecv(source, world)
        out._yaw = StatArray.StatArray.Irecv(source, world)
        out._orientation = StatArray.StatArray.Irecv(source, world)
        out._moment = StatArray.StatArray.Irecv(source, world)

        return out