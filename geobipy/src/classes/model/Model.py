""" @Model_Class
Module describing a Model
"""
from ...classes.core.myObject import myObject


class Model(myObject):
    """Abstract Model Class

    This is an abstract base class for additional model classes

    See Also
    ----------
    geobipy.Model1D
    geobipy.Model2D

    """

    def __init__(self):
        """ABC method"""
        NotImplementedError()

