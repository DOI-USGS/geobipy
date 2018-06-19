""" @Point_Class
Module describing a Point defined by x,y,z c-ordinates
"""
import numpy as np


class Point(object):
    """ Class defining a point in 3D Euclidean space """

    def __init__(self, x=0, y=0, z=0):
        """ Initialize the class """
        self.p = np.array([x, y, z])

    def __add__(self, other):
        """ Add two points together """
        P = Point()
        P.p = self.p + other.p
        return P

    def __sub__(self, other):
        """ Subtract two points """
        P = Point()
        P.p = self.p - other.p
        return P

    def move(self, dx, dy, dz):
        """ Move the point by [dx,dy,dz] """
        self.p[0] += dx
        self.p[1] += dy
        self.p[2] += dz
        return self

    def __str__(self):
        """ Prints the x,y,z co-ordinates of a point """
        return "x:{0},y:{1},z:{2}".format(
            str(self.p[0]), str(self.p[1]), str(self.p[2]))

    def distance(self, other):
        """ Computes the Euclidean distance between two points """
        return np.linalg.norm(self.p - other.p)

    def distance_from_origin(self):
        """ Compute the Euclidean distance from the point to (0,0,0) """
        return np.linalg.norm(self.p)
