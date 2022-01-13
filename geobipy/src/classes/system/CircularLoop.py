from copy import deepcopy
import numpy as np
from ...base import MPI as myMPI
from .EmLoop import EmLoop
from ..core import StatArray
from ..statistics.Distribution import Distribution
from ..mesh.RectilinearMesh1D import RectilinearMesh1D
from ..statistics.Histogram import Histogram

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

        super().__init__(x, y, z, elevation=elevation, **kwargs)
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
        if not isinstance(value, StatArray.StatArray):
            value = np.float64(value)
        # assert isinstance(value, (StatArray.StatArray, float, np.float64)), TypeError("pitch must have type float")
        self._pitch = StatArray.StatArray(value, 'Pitch', '$^{o}$')

    @property
    def radius(self):
        return self._radius

    @property
    def roll(self):
        return self._roll

    @roll.setter
    def roll(self, value):
        if not isinstance(value, StatArray.StatArray):
            value = np.float64(value)
        # assert isinstance(value, (StatArray.StatArray, float, np.float64)), TypeError("roll must have type float")
        self._roll = StatArray.StatArray(value, 'Roll', '$^{o}$')

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        if not isinstance(value, StatArray.StatArray):
            value = np.float64(value)
        # assert isinstance(value, (StatArray.StatArray, float, np.float64)), TypeError("yaw must have type float")
        self._yaw = StatArray.StatArray(value, 'Yaw', '$^{o}$')

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
        out = super().__deepcopy__(memo)
        out.orient = deepcopy(self.orient, memo=memo)
        out._moment = deepcopy(self.moment, memo=memo)
        out._pitch = deepcopy(self.pitch, memo=memo)
        out._roll = deepcopy(self.roll, memo=memo)
        out._yaw = deepcopy(self.yaw, memo=memo)
        out._radius = deepcopy(self.radius, memo=memo)

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

    def set_posteriors(self):

        super().set_posteriors()

        self.set_pitch_posterior()
        self.set_roll_posterior()
        self.set_yaw_posterior()

    def set_pitch_posterior(self):
        """

        """
        if self.pitch.hasPrior:
            mesh = RectilinearMesh1D(edges=StatArray.StatArray(self.pitch.prior.bins(), name=self.pitch.name, units=self.pitch.units), relativeTo=self.pitch)
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

    def createHdf(self, parent, myName, withPosterior=True, add_axis=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = super().createHdf(parent, myName, withPosterior, add_axis, fillvalue)
        if add_axis is not None:
            grp.attrs['repr'] = 'CircularLoop'

        self.pitch.createHdf(grp, 'pitch', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.roll.createHdf(grp, 'roll', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.yaw.createHdf(grp, 'yaw', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        data = StatArray.StatArray(3).createHdf(grp, 'data', add_axis=add_axis, fillvalue=fillvalue)


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

        data = StatArray.StatArray(np.asarray([self._orient, self.moment, self.radius], dtype=np.float64))
        data.writeHdf(grp, 'data', index=index)

    @classmethod
    def fromHdf(cls, grp, index=None):
        """ Reads in the object from a HDF file """

        out = super(CircularLoop, cls).fromHdf(grp, index)

        out.pitch = StatArray.StatArray.fromHdf(grp['pitch'], index=index)
        out.roll = StatArray.StatArray.fromHdf(grp['roll'], index=index)
        out.yaw = StatArray.StatArray.fromHdf(grp['yaw'], index=index)

        tmp = StatArray.StatArray.fromHdf(grp['data'], index=index)
        out._orient = tmp[0]
        out._moment = tmp[1]
        out._radius = tmp[2]

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
        data = np.asarray([self._orient, self.moment, self.radius], dtype=np.float64)
        myMPI.Isend(data, dest=dest, ndim=1, shape=(3, ), dtype=np.float64, world=world)

        self.pitch.Isend(dest, world)
        self.roll.Isend(dest, world)
        self.yaw.Isend(dest, world)


    @classmethod
    def Irecv(cls, source, world):

        out = super(CircularLoop, cls).Irecv(source, world)

        data = myMPI.Irecv(source=source, ndim=1, shape=(3, ), dtype=np.float64, world=world)

        out.pitch = StatArray.StatArray.Irecv(source, world)
        out.roll = StatArray.StatArray.Irecv(source, world)
        out.yaw = StatArray.StatArray.Irecv(source, world)

        out.orient = data[0]
        out._moment = data[1]
        out._radius = data[2]

        return out


    def __str__(self):
        """ Define print(self) """
        return 'CircularLoop("{0}",{1},{2},{3},{4},{5},{6},{7},{8})'.format(
            self.orient, self.moment,
            self.x,      self.y,    self.z,
            self.pitch,  self.roll, self.yaw, self.radius)
