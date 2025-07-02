from numpy import asarray, ceil, float64, hstack, int8, int32, minimum, size, unique, unravel_index
from copy import deepcopy
from matplotlib.figure import Figure
from matplotlib.pyplot import gcf
from ..statistics import StatArray
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

    __slots__ = ('_orientation', '_moment', '_pitch', '_roll', '_yaw')

    def __init__(self, x=None, y=None, z=None, elevation=None, orientation=None, moment=None, pitch=None, roll=None, yaw=None, **kwargs):

        super().__init__(x, y, z, elevation, **kwargs)

        self._orientation = StatArray.StatArray(self.nPoints, "Orientation", "", dtype=int32)
        self._moment = StatArray.StatArray(self.nPoints, "Moment", "")
        self._pitch  = StatArray.StatArray(self.nPoints, "Pitch", "$^{o}$")
        self._roll   = StatArray.StatArray(self.nPoints, "Roll", "$^{o}$")
        self._yaw    = StatArray.StatArray(self.nPoints, "Yaw", "$^{o}$")

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

        # if not isinstance(i, slice):
        #     i = unique(i)

        _ = self.orientation
        out._orientation = self._orientation[i]
        out.moment = self.moment[i]
        out.pitch = self.pitch[i]
        out.roll = self.roll[i]
        out.yaw = self.yaw[i]
        return out

    @property
    def addressof(self):
        msg = super().addressof
        msg += "pitch:\n{}".format(("|   "+self.pitch.addressof.replace("\n", "\n|   "))[:-4])
        msg += "roll:\n{}".format(("|   "+self.roll.addressof.replace("\n", "\n|   "))[:-4])
        msg += "yaw:\n{}".format(("|   "+self.yaw.addressof.replace("\n", "\n|   "))[:-4])
        return msg

    @property
    def address(self):
        out = super().address
        for x in [self.pitch, self.roll, self.yaw]:
            out = hstack([out, x.address.flatten()])

        return out

    @property
    def hasPosterior(self):
        return self.n_posteriors > 0

    @property
    def moment(self):
        if size(self._moment) == 0:
            self._moment = StatArray.StatArray(self.nPoints, "Moment", "")
        return self._moment

    @moment.setter
    def moment(self, values):
        if (values is not None):
            self.nPoints = size(values)
            if self._moment.size != self._nPoints:
                self._moment = StatArray.StatArray(values, "Moment", "")
                return

            self._moment[:] = values

    @property
    def pitch(self):
        if size(self._pitch) == 0:
            self._pitch = StatArray.StatArray(self.nPoints, "Pitch", "$^{o}$")
        return self._pitch

    @pitch.setter
    def pitch(self, values):
        if (values is not None):
            self.nPoints = size(values)
            if self._pitch.size != self._nPoints:
                self._pitch = StatArray.StatArray(values, "Pitch", "$^{o}$")
                return

            self._pitch[:] = values

    @property
    def priors(self):
        return super().priors | {k:v.prior for k,v in zip(['pitch', 'roll', 'yaw'], (self.pitch, self.roll, self.yaw)) if v.hasPrior}

    @property
    def roll(self):
        if size(self._roll) == 0:
            self._roll = StatArray.StatArray(self.nPoints, "Roll", "$^{o}$")
        return self._roll

    @roll.setter
    def roll(self, values):
        if (values is not None):
            self.nPoints = size(values)
            if self._roll.size != self._nPoints:
                self._roll = StatArray.StatArray(values, "Roll", "$^{o}$")
                return

            self._roll[:] = values

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
        if (values is not None):
            self.nPoints = size(values)
            if self._yaw.size != self._nPoints:
                self._yaw = StatArray.StatArray(values, "Yaw", "$^{o}$")
                return

            self._yaw[:] = values

    @property
    def orientation(self):
        if size(self._orientation) == 0:
            self._orientation = StatArray.StatArray(self.nPoints, "Orientation", "", dtype=int32)

        tmp = ('x', 'y', 'z')
        return [tmp[i] for i in self._orientation]

    @orientation.setter
    def orientation(self, values):
        if (values is not None):
            tmp = {'x': 0, 'y':1, 'z':2}

            self.nPoints = size(values)

            values = asarray([tmp[x.replace(" ", "")] for x in values])

            if self._orientation.size != self._nPoints:
                self._orientation = StatArray.StatArray(values, "Orientation", dtype=int32)
                return

            self._orientation[:] = values

    @property
    def n_posteriors(self):
        return super().n_posteriors + self.pitch.hasPosterior + self.roll.hasPosterior + self.yaw.hasPosterior

    @property
    def summary(self):
        """Print a summary"""
        msg = super().summary

        msg += "orientation:\n{}\n".format("|   "+(self._orientation.summary.replace("\n", "\n|   ")))
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

    def append(self, other):

        super().append(other)

        self._orientation = self._orientation.append(other._orientation)
        self._moment = self._moment.append(other.moment)
        self._pitch = self._pitch.append(other.pitch)
        self._roll = self._roll.append(other.roll)
        self._yaw = self._yaw.append(other.yaw)

        return self


    def perturb(self):
        """Propose a new point given the attached propsal distributions
        """
        super().perturb()

        if self.pitch.hasPosterior:
                self.pitch.perturb(imposePrior=True, log=True)
                # Update the mean of the proposed elevation
                self.pitch.proposal.mean = self.pitch

        if self.roll.hasPosterior:
                self.roll.perturb(imposePrior=True, log=True)
                # Update the mean of the proposed elevation
                self.roll.proposal.mean = self.roll

        if self.yaw.hasPosterior:
                self.yaw.perturb(imposePrior=True, log=True)
                # Update the mean of the proposed elevation
                self.yaw.proposal.mean = self.yaw

    @property
    def probability(self):
        probability = super().probability

        if self.pitch.hasPrior:
            probability += self.pitch.probability(log=True)

        if self.roll.hasPrior:
            probability += self.roll.probability(log=True)

        if self.yaw.hasPrior:
            probability += self.yaw.probability(log=True)

        return probability

    def set_priors(self, x_prior=None, y_prior=None, z_prior=None, pitch_prior=None, roll_prior=None, yaw_prior=None, **kwargs):

        super().set_priors(x_prior, y_prior, z_prior, **kwargs)

        if pitch_prior is None:
            if kwargs.get('solve_pitch', False):
                pitch_prior = Distribution('Uniform',
                                            self.pitch - kwargs['maximum_pitch_change'],
                                            self.pitch + kwargs['maximum_pitch_change'],
                                            prng=kwargs.get('prng'))
        self.pitch.prior = pitch_prior

        if roll_prior is None:
            if kwargs.get('solve_roll', False):
                roll_prior = Distribution('Uniform',
                                            self.roll - kwargs['maximum_roll_change'],
                                            self.roll + kwargs['maximum_roll_change'],
                                            prng=kwargs.get('prng'))
        self.roll.prior = roll_prior

        if yaw_prior is None:
            if kwargs.get('solve_yaw', False):
                yaw_prior = Distribution('Uniform',
                                            self.yaw - kwargs['maximum_yaw_change'],
                                            self.yaw + kwargs['maximum_yaw_change'],
                                            prng=kwargs.get('prng'))
        self.yaw.prior = yaw_prior

    def set_proposals(self, x_proposal=None, y_proposal=None, z_proposal=None, pitch_proposal=None, roll_proposal=None, yaw_proposal=None, **kwargs):

        super().set_proposals(x_proposal, y_proposal, z_proposal, **kwargs)

        if pitch_proposal is None:
            if kwargs.get('solve_pitch', False):
                pitch_proposal = Distribution('Normal', self.pitch.item(), kwargs['pitch_proposal_variance'], prng=kwargs.get('prng'))

        self.pitch.proposal = pitch_proposal

        if roll_proposal is None:
            if kwargs.get('solve_roll', False):
                roll_proposal = Distribution('Normal', self.roll.item(), kwargs['roll_proposal_variance'], prng=kwargs.get('prng'))

        self.roll.proposal = roll_proposal

        if yaw_proposal is None:
            if kwargs.get('solve_yaw', False):
                yaw_proposal = Distribution('Normal', self.yaw.item(), kwargs['yaw_proposal_variance'], prng=kwargs.get('prng'))

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
            mesh = RectilinearMesh1D(edges=StatArray.StatArray(self.pitch.prior.bins(199), name=self.pitch.name, units=self.pitch.units), relative_to=self.pitch)
            self.pitch.posterior = Histogram(mesh=mesh)

    def set_roll_posterior(self):
        """

        """
        if self.roll.hasPrior:
            mesh = RectilinearMesh1D(edges = StatArray.StatArray(self.roll.prior.bins(), name=self.roll.name, units=self.roll.units), relative_to=self.roll)
            self.roll.posterior = Histogram(mesh=mesh)

    def set_yaw_posterior(self):
        """

        """
        if self.yaw.hasPrior:
            mesh = RectilinearMesh1D(edges=StatArray.StatArray(self.yaw.prior.bins(), name=self.yaw.name, units=self.yaw.units), relative_to=self.yaw)
            self.yaw.posterior = Histogram(mesh=mesh)

    def update_posteriors(self):

        super().update_posteriors()
        self.pitch.update_posterior()
        self.roll.update_posterior()
        self.yaw.update_posterior()

    def _init_posterior_plots(self, gs):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """
        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        shp = (minimum(3, self.n_posteriors), ceil(self.n_posteriors / 3).astype(int32))
        splt = gs.subgridspec(*shp, wspace=0.3, hspace=1.0)

        ax = {}
        if super().hasPosterior:
            ax['point'] = super()._init_posterior_plots(splt[:super().n_posteriors, 0])

        k = 0
        for l, c in zip(('pitch', 'roll', 'yaw'), [self.pitch, self.roll, self.yaw]):
            if c.hasPosterior:
                j = super().n_posteriors + k
                s = unravel_index(j, shp, order='F')
                ax[l] = c._init_posterior_plots(splt[s[0], s[1]])
                k += 1

        return ax

    def plot_posteriors(self, axes=None, **kwargs):

        if axes is None:
            axes = kwargs.pop('fig', gcf())

        if not isinstance(axes, dict):
            axes = self._init_posterior_plots(axes)

        assert len(axes) == self.n_posteriors, ValueError("Must have length {} list of axes for the posteriors. self._init_posterior_plots can generate them.".format(self.n_posteriors))

        if super().hasPosterior:
            super().plot_posteriors(axes['point'], **kwargs)

        pitch_kwargs = kwargs.pop('pitch_kwargs', {})
        roll_kwargs = kwargs.pop('roll_kwargs', {})
        yaw_kwargs = kwargs.pop('yaw_kwargs', {})

        overlay = kwargs.pop('overlay', None)

        for l, c, kw in zip(('pitch', 'roll', 'yaw'), [self.pitch, self.roll, self.yaw], [pitch_kwargs, roll_kwargs, yaw_kwargs]):
            if c.hasPosterior:
                c.plot_posteriors(ax = axes[l], **kw)

        if overlay is not None:
            self.overlay_on_posteriors(overlay, axes, pitch_kwargs, roll_kwargs, yaw_kwargs)

    def overlay_on_posteriors(self, overlay, axes, pitch_kwargs={}, roll_kwargs={}, yaw_kwargs={}, **kwargs):

        assert isinstance(overlay, EmLoop), TypeError("overlay must have type EmLoop")

        super().overlay_on_posteriors(overlay, axes['point'], **kwargs)

        for l, c, o, kw in zip(('pitch', 'roll', 'yaw'),
                               [self.pitch, self.roll, self.yaw],
                               [overlay.pitch, overlay.roll, overlay.yaw],
                               [pitch_kwargs, roll_kwargs, yaw_kwargs]):
            if c.hasPosterior:
                c.posterior.plot_overlay(value = o, ax = axes[l], **kw)

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        grp = super().createHdf(parent, name, withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
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
        self.pitch.writeHdf(grp, 'pitch', withPosterior=withPosterior, index=index)
        self.roll.writeHdf(grp, 'roll', withPosterior=withPosterior, index=index)
        self.yaw.writeHdf(grp, 'yaw', withPosterior=withPosterior, index=index)
        self.moment.writeHdf(grp, 'moment', withPosterior=withPosterior, index=index)
        self._orientation.writeHdf(grp, 'orientation', withPosterior=withPosterior, index=index)

    @classmethod
    def fromHdf(cls, grp, index=None):
        """ Reads in the object from a HDF file """

        out = super(EmLoop, cls).fromHdf(grp, index)

        out._pitch = StatArray.StatArray.fromHdf(grp['pitch'], index=index)
        out._roll = StatArray.StatArray.fromHdf(grp['roll'], index=index)
        out._yaw = StatArray.StatArray.fromHdf(grp['yaw'], index=index)
        out._moment = StatArray.StatArray.fromHdf(grp['moment'], index=index)
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