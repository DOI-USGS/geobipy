""" @RectilinearMesh1D_Class
Module describing a 1D Rectilinear Mesh class with x axis specified
"""
from ...classes.core.myObject import myObject
from ...classes.core.StatArray import StatArray
import numpy as np
from ...base import customPlots as cp
from ...base.customFunctions import safeEval


class RectilinearMesh1D(myObject):
    """ Class defining a 2D rectilinear mesh whose cells are rectangular with linear sides """


    def __init__(self, x=None, name=None, units=None, dtype=np.float64):
        """ Initialize a 2D Rectilinear Mesh
        """
        tmp = [x, name, units]
        if (all([i is None for i in tmp])):
            return
        ## StatArray of the 2D mesh values
        self.arr = StatArray(x.size, name, units, dtype=dtype)
#        assert isinstance(x,StatArray), 'x must be an StatArray'
        ## StatArray of the x axis values
        self.x = x.deepcopy()

        # Set some extra variables for speed
        # Is the discretization regular
        self.isRegular = self.x.isRegular()
        # Get the increment
        self.dx = self.x[1] - self.x[0]


    def hdfName(self):
        """ Reprodicibility procedure """
        return('RectilinearMesh1D()')


    def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self.x.toHdf(grp, 'x')
        self.y.createHdf(grp, 'arr', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        self.arr.writeHdf(parent, myName+'/arr',  index=index)


    def toHdf(self, h5obj, myName):
        """ Write the StatArray to an HDF object
        h5obj: :An HDF File or Group Object.
        """
        # Create a new group inside h5obj
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self.x.toHdf(grp, 'x')
        self.arr.toHdf(grp, 'arr')


    def fromHdf(self, grp, index=None):
        """ Reads in the object froma HDF file """
        item = grp.get('x')
        obj = eval(safeEval(item.attrs.get('repr')))
        x = obj.fromHdf(item)
        res = RectilinearMesh1D(x)
        item = grp.get('arr')
        obj = eval(safeEval(item.attrs.get('repr')))
        if (index is None):
            res.arr = obj.fromHdf(item)
        else:
            res.arr = obj.fromHdf(item, index=np.s_[index,:])
        return res


    def summary(self):
        """ Print a summary of self """
        self.x.summary()
        self.arr.summary()

