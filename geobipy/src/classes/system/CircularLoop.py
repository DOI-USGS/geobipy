from copy import deepcopy
import numpy as np
from ...base import MPI as myMPI
from .EmLoop import EmLoop
from ...base import utilities as cf
from ..core import StatArray
from ..statistics.Histogram1D import Histogram1D
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

        super().__init__(x, y, z)
        # Orientation of the loop dipole
        self.orient = orient
        # Dipole moment of the loop
        self._moment = moment
        # Pitch of the loop
        self.pitch = pitch
        # Roll of the loop
        self.roll = roll
        # Yaw of the loop
        self.yaw = yaw
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

    @pitch.setter
    def pitch(self, value):
        self._pitch = StatArray.StatArray(np.float64(value), 'Pitch', '$^{o}$')

    @property
    def radius(self):
        return self._radius

    @property
    def roll(self):
        return self._roll

    @roll.setter
    def roll(self, value):
        self._roll = StatArray.StatArray(np.float64(value), 'Roll', '$^{o}$')

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        self._yaw = StatArray.StatArray(np.float64(value), 'Yaw', '$^{o}$')

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

    def __deepcopy__(self, memo={}):
        result = type(self).__new__(type(self))
        result.orient = deepcopy(self.orient)
        result._moment = deepcopy(self.moment)
        result._x = deepcopy(self.x)
        result._y = deepcopy(self.y)
        result._z = deepcopy(self.z)
        result._pitch = deepcopy(self.pitch)
        result._roll = deepcopy(self.roll)
        result._yaw = deepcopy(self.yaw)
        result._radius = deepcopy(self.radius)

        return result

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

    def set_posteriors(self):

        super().set_posteriors()

        self.set_pitch_posterior()
        self.set_roll_posterior()
        self.set_yaw_posterior()

    def set_pitch_posterior(self):
        """

        """
        if self.pitch.hasPrior:
            self.pitch.posterior = Histogram1D(edges = StatArray.StatArray(self.pitch.prior.bins(), name=self.pitch.name, units=self.pitch.units), relativeTo=self.pitch)

    def set_roll_posterior(self):
        """

        """
        if self.roll.hasPrior:
            self.roll.posterior = Histogram1D(edges = StatArray.StatArray(self.roll.prior.bins(), name=self.roll.name, units=self.roll.units), relativeTo=self.roll)

    def set_yaw_posterior(self):
        """

        """
        if self.yaw.hasPrior:
            self.yaw.posterior = Histogram1D(edges = StatArray.StatArray(self.yaw.prior.bins(), name=self.yaw.name, units=self.yaw.units), relativeTo=self.yaw)

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


    @classmethod
    def Irecv(cls, source, world):
        data = myMPI.Irecv(source=source, ndim=1, shape=(9, ), dtype=np.float64, world=world)
        return cls(*data)


    def __str__(self):
        """ Define print(self) """
        return 'CircularLoop("{0}",{1},{2},{3},{4},{5},{6},{7},{8})'.format(
            self.orient, self.moment,
            self.x,      self.y,    self.z,
            self.pitch,  self.roll, self.yaw, self.radius)
