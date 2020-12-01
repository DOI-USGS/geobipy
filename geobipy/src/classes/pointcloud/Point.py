""" @Point_Class
Module describing a Point defined by x,y,z c-ordinates
"""
from ..core import StatArray
import numpy as np


class Point(object):
    """ Class defining a point in 3D Euclidean space """

    def __init__(self, x=0.0, y=0.0, z=0.0):

        """ Initialize the class """

        # x coordinate
        self.x = x
        # y coordinate
        self._y = StatArray.StatArray(y, 'Northing', 'm')
        # z coordinate
        self._z = StatArray.StatArray(z, 'Height', 'm')


    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = StatArray.StatArray(value, 'Easting', 'm')


    @property
    def y(self):
        return self._y

    @x.setter
    def y(self, value):
        self._y = StatArray.StatArray(value, 'Northing', 'm')

    @property
    def z(self):
        return self._z

    @x.setter
    def z(self, value):
        self._z = StatArray.StatArray(value, 'Height', 'm')


    def __add__(self, other):
        """ Add two points together """
        P = self.deepcopy()
        P._x += other.x
        P._y += other.y
        P._z += other.z
        return P


    def __sub__(self, other):
        """ Subtract two points """
        P = self.deepcopy()
        P._x -= other.x
        P._y -= other.y
        P._z -= other.z
        return P


    def distance(self, other, **kwargs):
        """Get the Lp norm distance between two points. """
        return np.linalg.norm(np.asarray([self.x, self.y, self.z])-np.asarray([other.x, other.y, other.z]), **kwargs)


    def deepcopy(self):
        return self.__deepcopy__()


    def __deepcopy__(self):
        """ Define a deepcopy routine """
        return Point(self.x, self.y, self.z)


    def move(self, dx, dy, dz):
        """ Move the point by [dx,dy,dz] """
        self._x += dx
        self._y += dy
        self._z += dz
        return self


    # def __str__(self):
    #     """ Prints the x,y,z co-ordinates of a point """
    #     return "Point({}, {}, {})".format(self.x[0], self.y[0], self.z[0])


    def Isend(self, dest, world):
        tmp = np.empty(3, np.float64)
        tmp[:] = np.hstack([self.x, self.y, self.z])
        world.Isend(tmp, dest=dest)


    def Irecv(self, source, world):
        tmp = np.empty(3, np.float64)
        req = world.Irecv(tmp, source=source)
        req.Wait()
        return Point(tmp[0], tmp[1], tmp[2])
