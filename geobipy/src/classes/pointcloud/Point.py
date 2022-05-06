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

    def set_priors(self, x_prior=None, y_prior=None, z_prior=None, kwargs={}):
        if x_prior is not None:
            self.x.prior = x_prior
        if y_prior is not None:
            self.y.prior = y_prior
        if z_prior is not None:
            self.z.prior = z_prior

    def set_proposals(self, x_proposal=None, y_proposal=None, z_proposal=None, kwargs={}):

        if x_proposal is None:
            if kwargs.get('solve_x', False):
                x_proposal = Distribution('Normal', self.x.value, kwargs['x_proposal_variance'], prng=kwargs['prng'])
        self.x.proposal = x_proposal

        if y_proposal is None:
            if kwargs.get('solve_y', False):
                y_proposal = Distribution('Normal', self.y.value, kwargs['y_proposal_variance'], prng=kwargs['prng'])
        self.y.proposal = y_proposal

        if z_proposal is None:
            if kwargs.get('solve_z', False):
                z_proposal = Distribution('Normal', self.z.value, kwargs['z_proposal_variance'], prng=kwargs['prng'])
        self.z.proposal = z_proposal

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
