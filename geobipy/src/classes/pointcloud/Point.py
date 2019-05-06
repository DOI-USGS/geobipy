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
        self._x = StatArray.StatArray(x, 'Easting', 'm')
        # y coordinate
        self._y = StatArray.StatArray(y, 'Northing', 'm')
        # z coordinate
        self._z = StatArray.StatArray(z, 'Height', 'm')


    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z


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

    
    def norm(self, other, **kwargs):
        """Get the Lp norm distance between two points. """
        return np.linalg.norm(np.asarray([self._x, self._y, self._z])-np.asarray([other._x, other._y, other._z]), **kwargs)


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
