""" @Point_Class
Module describing a Point defined by x,y,z c-ordinates
"""
from abc import ABC, abstractmethod
from copy import deepcopy

from ..mesh.RectilinearMesh1D import RectilinearMesh1D
from ..statistics.Histogram import Histogram
from ..statistics.Distribution import Distribution
from ..core.myObject import myObject
from ..core import StatArray
import numpy as np
from matplotlib.figure import Figure
from matplotlib.pyplot import gcf


class Point(myObject, ABC):
    """ Class defining a point in 3D Euclidean space """

    def __init__(self, x=0.0, y=0.0, z=0.0, elevation=0.0, **kwargs):

        """ Initialize the class """

        # x coordinate
        self.x = x
        # y coordinate
        self.y = y
        # z coordinate
        self.z = z

        self.elevation = elevation

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, value):
        self._elevation = StatArray.StatArray(value, 'Elevation', 'm')

    @property
    def hasPosteriors(self):
        return (self.x.hasPosterior + self.y.hasPosterior + self.z.hasPosterior) > 0

    @property
    def n_posteriors(self):
        return self.x.hasPosterior + self.y.hasPosterior + self.z.hasPosterior

    @property
    def summary(self):
        """Summary of self """
        msg =  "{}\n".format(type(self).__name__)
        msg += "x:\n{}".format("|   "+(self.x.summary.replace("\n", "\n|   "))[:-4])
        msg += "y:\n{}".format("|   "+(self.y.summary.replace("\n", "\n|   "))[:-4])
        msg += "z:\n{}".format("|   "+(self.z.summary.replace("\n", "\n|   "))[:-4])
        msg += "elevation:\n{}".format("|   "+(self.elevation.summary.replace("\n", "\n|   "))[:-4])

        return msg

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = StatArray.StatArray(value, 'Easting', 'm')

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = StatArray.StatArray(value, 'Northing', 'm')

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        self._z = StatArray.StatArray(value, 'Height', 'm')

    def __add__(self, other):
        """ Add two points together """
        P = deepcopy(self)
        P._x += other.x
        P._y += other.y
        P._z += other.z
        P._elevation += other.elevation
        return P

    def __deepcopy__(self, memo={}):
        """ Define a deepcopy routine """
        out = type(self).__new__(type(self))
        out._x = deepcopy(self.x, memo=memo)
        out._y = deepcopy(self.y, memo=memo)
        out._z = deepcopy(self.z, memo=memo)
        out._elevation = deepcopy(self.elevation, memo=memo)
        return out

    def __sub__(self, other):
        """ Subtract two points """
        P = deepcopy(self)
        P._x -= other.x
        P._y -= other.y
        P._z -= other.z
        P._elevation -= other.elevation
        return P

    def distance(self, other, **kwargs):
        """Get the Lp norm distance between two points. """
        return np.linalg.norm(np.asarray([self.x, self.y, self.z]) - np.asarray([other.x, other.y, other.z]), **kwargs)

    def move(self, dx, dy, dz):
        """ Move the point by [dx,dy,dz] """
        self._x += dx
        self._y += dy
        self._z += dz
        self._elevation += dz
        return self

    def perturb(self):
        """Propose a new point given the attached propsal distributions
        """

        for c in [self.x, self.y, self.z]:
            if c.hasPosterior:
                c.perturb(imposePrior=True, log=True)
                # Update the mean of the proposed elevation
                c.proposal.mean = c

    @property
    def probability(self):
        """Evaluate the probability for the EM data point given the specified attached priors

        Parameters
        ----------
        rEerr : bool
            Include the relative error when evaluating the prior
        aEerr : bool
            Include the additive error when evaluating the prior
        height : bool
            Include the elevation when evaluating the prior
        calibration : bool
            Include the calibration parameters when evaluating the prior
        verbose : bool
            Return the components of the probability, i.e. the individually evaluated priors

        Returns
        -------
        out : np.float64
            The evaluation of the probability using all assigned priors

        Notes
        -----
        For each boolean, the associated prior must have been set.

        Raises
        ------
        TypeError
            If a prior has not been set on a requested parameter

        """
        probability = np.float64(0.0)

        if self.x.hasPrior:
            probability += self.x.probability(log=True)

        if self.y.hasPrior:
            probability += self.y.probability(log=True)

        if self.z.hasPrior:
            probability += self.z.probability(log=True)

        return probability

    def set_priors(self, x_prior=None, y_prior=None, z_prior=None, **kwargs):

        if x_prior is None:
            if kwargs.get('solve_x', False):
                x_prior = Distribution('Uniform', self.x - kwargs['maximum_x_change'], self.x + kwargs['maximum_x_change'], prng=kwargs.get('prng'))

        if y_prior is None:
            if kwargs.get('solve_y', False):
                y_prior = Distribution('Uniform', self.y - kwargs['maximum_y_change'], self.y + kwargs['maximum_y_change'], prng=kwargs.get('prng'))

        if z_prior is None:
            if kwargs.get('solve_z', False):
                z_prior = Distribution('Uniform', self.z - kwargs['maximum_z_change'], self.z + kwargs['maximum_z_change'], prng=kwargs.get('prng'))

        self.x.prior = x_prior
        self.y.prior = y_prior
        self.z.prior = z_prior

    def set_proposals(self, x_proposal=None, y_proposal=None, z_proposal=None, **kwargs):

        if x_proposal is None:
            if kwargs.get('solve_x', False):
                x_proposal = Distribution('Normal', self.x.value, kwargs['x_proposal_variance'], prng=kwargs['prng'])

        if y_proposal is None:
            if kwargs.get('solve_y', False):
                y_proposal = Distribution('Normal', self.y.value, kwargs['y_proposal_variance'], prng=kwargs['prng'])

        if z_proposal is None:
            if kwargs.get('solve_z', False):
                z_proposal = Distribution('Normal', self.z.value, kwargs['z_proposal_variance'], prng=kwargs['prng'])

        self.x.proposal = x_proposal
        self.y.proposal = y_proposal
        self.z.proposal = z_proposal

    def reset_posteriors(self):
        self.x.reset_posteriors()
        self.y.reset_posteriors()
        self.z.reset_posteriors()

    def set_posteriors(self):

        self.set_x_posterior()
        self.set_y_posterior()
        self.set_z_posterior()

    def set_x_posterior(self):
        """

        """
        if self.x.hasPrior:
            mesh = RectilinearMesh1D(edges = StatArray.StatArray(self.x.prior.bins(), name=self.x.name, units=self.x.units), relativeTo=self.x)
            self.x.posterior = Histogram(mesh=mesh)

    def set_y_posterior(self):
        """

        """
        if self.y.hasPrior:
            mesh = RectilinearMesh1D(edges = StatArray.StatArray(self.y.prior.bins(), name=self.y.name, units=self.y.units), relativeTo=self.y)
            self.y.posterior = Histogram(mesh=mesh)

    def set_z_posterior(self):
        """

        """
        if self.z.hasPrior:
            mesh = RectilinearMesh1D(edges = StatArray.StatArray(self.z.prior.bins(), name=self.z.name, units=self.z.units), relativeTo=self.z)
            self.z.posterior = Histogram(mesh=mesh)

    def update_posteriors(self):
        self.x.update_posterior()
        self.y.update_posterior()
        self.z.update_posterior()

    def _init_posterior_plots(self, gs=None):
        """Initialize axes for posterior plots

        Parameters
        ----------
        gs : matplotlib.gridspec.Gridspec
            Gridspec to split

        """
        if gs is None:
            gs = Figure()

        if isinstance(gs, Figure):
            gs = gs.add_gridspec(nrows=1, ncols=1)[0, 0]

        n_posteriors = self.x.hasPosterior + self.y.hasPosterior + self.z.hasPosterior
        splt = gs.subgridspec(n_posteriors, 1, wspace=0.3, hspace=1.0)

        ax = []
        i = 0
        for c in [self.x, self.y, self.z]:
            if c.hasPosterior:
                ax.append(c._init_posterior_plots(splt[i]))
                i += 1

        return ax

    def plot_posteriors(self, axes=None, **kwargs):

        if axes is None:
            axes = kwargs.pop('fig', gcf())

        if not isinstance(axes, list):
            axes = self._init_posterior_plots(axes)

        n_posteriors = self.x.hasPosterior + self.y.hasPosterior + self.z.hasPosterior
        assert len(axes) == n_posteriors, ValueError("Must have length {} list of axes for the posteriors. self._init_posterior_plots can generate them.".format(n_posteriors))

        x_kwargs = kwargs.pop('x_kwargs', {})
        y_kwargs = kwargs.pop('y_kwargs', {})
        z_kwargs = kwargs.pop('z_kwargs', {})

        overlay = kwargs.pop('overlay', None)
        if not overlay is None:
            x_kwargs['overlay'] = overlay.x
            y_kwargs['overlay'] = overlay.y
            z_kwargs['overlay'] = overlay.z

        if ~self.x.hasPosterior & ~self.y.hasPosterior & self.z.hasPosterior:
            z_kwargs['transpose'] = z_kwargs.get('transpose', True)

        i = 0
        for c, kw in zip([self.x, self.y, self.z], [x_kwargs, y_kwargs, z_kwargs]):
            if c.hasPosterior:
                c.plotPosteriors(ax = axes[i], **kw)
                i += 1

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = self.create_hdf_group(parent, name)
        self.x.createHdf(grp, 'x', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.y.createHdf(grp, 'y', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.z.createHdf(grp, 'z', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)
        self.elevation.createHdf(grp, 'elevation', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue)

        if add_axis is not None:
            grp.attrs['repr'] = "PointCloud3D"

        return grp

    def writeHdf(self, parent, name, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        grp = parent[name]
        self.x.writeHdf(grp, 'x',  withPosterior=withPosterior, index=index)
        self.y.writeHdf(grp, 'y',  withPosterior=withPosterior, index=index)
        self.z.writeHdf(grp, 'z',  withPosterior=withPosterior, index=index)
        self.elevation.writeHdf(grp, 'elevation',  withPosterior=withPosterior, index=index)

    @classmethod
    def fromHdf(cls, grp, index=None, **kwargs):
        """ Reads the object from a HDF group """

        x = StatArray.StatArray.fromHdf(grp['x'], index=index)
        y = StatArray.StatArray.fromHdf(grp['y'], index=index)
        z = StatArray.StatArray.fromHdf(grp['z'], index=index)
        elevation = StatArray.StatArray.fromHdf(grp['elevation'], index=index)

        return cls(x=x, y=y, z=z, elevation=elevation, **kwargs)

    def Isend(self, dest, world):
        self.x.Isend(dest, world)
        self.y.Isend(dest, world)
        self.z.Isend(dest, world)
        self.elevation.Isend(dest, world)

    @classmethod
    def Irecv(cls, source, world, **kwargs):
        x = StatArray.StatArray.Irecv(source, world)
        y = StatArray.StatArray.Irecv(source, world)
        z = StatArray.StatArray.Irecv(source, world)
        e = StatArray.StatArray.Irecv(source, world)
        return cls(x=x, y=y, z=z, elevation=e, **kwargs)
