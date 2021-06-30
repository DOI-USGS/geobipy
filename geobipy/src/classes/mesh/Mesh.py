""" @Mesh_Class
Module describing a Mesh
"""
from ...classes.core.myObject import myObject


class Mesh(myObject):
    """Abstract Base Class

    This is an abstract base class for additional meshes

    See Also
    ----------
    geobipy.RectilinearMesh1D
    geobipy.RectilinearMesh2D
    geobipy.RectilinearMesh3D

    """

    def __init__(self):
        """ABC method"""
        NotImplementedError()
