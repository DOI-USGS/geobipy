""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
from copy import deepcopy
from ...classes.core.myObject import myObject
from ...classes.core.StatArray import StatArray
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from ...base import customPlots as cp
from ...base.customFunctions import safeEval
from ...base.customFunctions import _logSomething, isInt


class RectilinearMesh2D(myObject):
    """ Class defining a 2D rectilinear mesh whose cells are rectangular with linear sides """


    def __init__(self, x=None, y=None, name=None, units=None, dtype=np.float64):
        """ Initialize a 2D Rectilinear Mesh
        """
        if (x is None):
            return
        
        assert np.ndim(x) == 1, TypeError('x must only have 1 dimension')
        assert np.ndim(y) == 1, TypeError('y must only have 1 dimension')
        assert isinstance(x,StatArray), TypeError('x must be of type StatArray')
        assert isinstance(y,StatArray), TypeError('y must be of type StatArray')
    
        # StatArray of the 2D mesh values
        self.arr = StatArray([y.size, x.size], name, units, dtype=dtype)

        # StatArray of the x axis values
        self.x = x.deepcopy()
        # StatArray of the y axis values
        self.y = y.deepcopy()

        # Set some extra variables for speed
        self.xIsRegular = self.x.isRegular()
        self.yIsRegular = self.y.isRegular()
        self.dx = self.x[1] - self.x[0]
        self.xWidth = np.abs(np.diff(self.x))
        self.xWidth = self.xWidth.append(self.xWidth[-1])
        self.dy = self.y[1] - self.y[0]
        self.yWidth = np.abs(np.diff(self.y))
        self.yWidth = self.yWidth.append(self.yWidth[-1])


    def aggregate(self, y1 = None, y2 = None):
        """ Aggregate the rows over the interval [y1 y2]
            if y1 is instead a sequence, aggregate the hitmap for the intervals defined by the sequence, y2 is ignored"""

        if (y1 is None):
            y1 = self.y[0]
        if (y2 is None):
            y2 = self.y[-1]

        if (np.size(y1) == 1):
            # Don't need to check for depth being shallower than zGrid[0] since the sortedsearch with return 0
            assert y1 <= self.y[-1], 'aggregate: y1 is greater than max - '+str(self.y[-1])
            assert y2 <= self.y[-1], 'aggregate: y2 is greater than max - '+str(self.y[-1])

            cell1 = self.y.searchsorted(y1)
            cell2 = self.y.searchsorted(y2)
            vals = np.mean(self.arr[cell1:cell2+1,:],axis = 0)
        else:
            vals = self.intervalMean(y1,0)
        return vals


    def deepcopy(self):
        return deepcopy(self)


    def __deepcopy__(self, memo):
        """ Define the deepcopy for the StatArray """
        other = RectilinearMesh2D(self.x, self.y, self.name, self.units, self.arr.dtype)
        other.arr[:,:] += self.arr
        return other


    def getCellEdges(self, dim):
        """ Gets the cell edges in the given dimension """
        if (dim == 0):
            return self.y.edges()
        else:
            return self.x.edges()


    def getMeanInterval(self):
        """ Gets the mean of the array """
        t = np.sum(np.repeat(self.x[np.newaxis,:],self.arr.shape[0],0) * self.arr,1)
        s = self.arr.sum(1)
        if all(s == 0.0): return t
        return t / s


    def isSame(self, other):
        """ Determines if two grids contain the same axes and size """
        if (not self.hasSameSize(other)):
            return False
        if (self.x != other.x):
            return False
        if (self.y != other.y):
            return False


    def hasSameSize(self, other):
        """ Determines if the meshes have the same dimension sizes """
        if self.arr.shape != other.arr.shape:
            return False
        if self.x.size != other.x.size:
            return False
        if self.y.size != other.y.size:
            return False
        return True


    def getConfidenceIntervals(self, percent, log=None):
        """ Gets the confidence intervals for the specified dimension """
        total = self.arr[0, :].sum()

        p = 0.01 * percent
        tmp = np.cumsum(self.arr, 1)
        ixM = np.argmin(np.abs(tmp - 0.5 * total), 1)
        ix1 = np.argmin(np.abs(tmp - ((1.0 - p) * total)), 1)
        ix2 = np.argmin(np.abs(tmp - p * total), 1)
        sigMed = self.x[ixM]
        sigLow = self.x[ix1]
        sigHigh = self.x[ix2]

        if (not log is None):
            sigMed, dum = _logSomething(sigMed, log=log)
            sigLow, dum = _logSomething(sigLow, log=log)
            sigHigh, dum = _logSomething(sigHigh, log=log)

        return (sigMed, sigLow, sigHigh)


    def intervalMean(self, newGrid, dim):
        """ Compute the mean for the intervals given by newGrid along dimension dim """
        if (dim == 0):
            res = np.zeros([np.size(newGrid)-1,self.x.size])
            r = range(self.x.size)
            for i in r:
                bins = binned_statistic(self.y,self.arr[:,i],bins = newGrid)
                res[:,i] = bins.statistic
        else:
            res = np.zeros([self.arr.shape[0],np.size(newGrid)-1])
            r = range(self.y.size)
            for i in r:
                bins = binned_statistic(self.x,self.arr[i,:],bins = newGrid)
                res[i,:] = bins.statistic
        return res


    def regrid(self, newGrid, dim):
        """ Regrid a dimension to the new intervals given by newGrid """
        arr = self.intervalMean(newGrid, dim)
        if (dim == 0):
            x = self.x
            y = StatArray(newGrid[:-1]+0.5*np.diff(newGrid),self.y.name,self.y.units)
        else:
            x = StatArray(newGrid[:-1]+0.5*np.diff(newGrid),self.x.name,self.x.units)
            y = self.y

        res = RectilinearMesh2D(x, y, name=self.arr.name, units=self.arr.units)
        res.arr[:,:] = arr
        return res


    def normBySum(self, dim):
        """ Normalizes the parameters by the sum along dimension dim """
        s = np.sum(self.arr, dim)
        if (dim == 0):
            self.arr = self.arr / np.repeat(s[np.newaxis, :], np.size(self.arr, dim), dim)
        elif (dim == 1):
            self.arr = self.arr / np.repeat(s[:, np.newaxis], np.size(self.arr, dim), dim)
        self.arr.name = 'Relative frequency'


    def plot(self,title='',invX=False,logX=False,flipY=False,**kwargs):
        """ Plot the 2D Rectilinear Mesh values """
        ax = plt.gca()
        plt.cla()
        if (invX):
            plt.pcolormesh(1.0 / self.x,self.y,np.asarray(self.arr), **kwargs)
        else:
            plt.pcolormesh(self.x, self.y, np.asarray(self.arr), **kwargs)
            cp.xlabel(self.x.getNameUnits())
        cp.ylabel(self.y.getNameUnits())
        cp.title(title)
        if (flipY):
            ax = plt.gca()
            lim = ax.get_ylim()
            if (lim[1] > lim[0]):
                ax.set_ylim(lim[::-1])
        if logX:
            plt.xscale('log')
        plt.colorbar()

    
    def pcolor(self, **kwargs):
        """ Plot the Histogram2D as an image """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        
        self.arr.pcolor(x=x, y=y, **kwargs)


    def hdfName(self):
        """ Reprodicibility procedure """
        return('Rmesh2D([1,1])')


    def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self.arr.createHdf(grp, 'arr', nRepeats=nRepeats, fillvalue=fillvalue)
        self.x.createHdf(grp,'x', nRepeats=nRepeats, fillvalue=fillvalue)
        self.y.createHdf(grp,'y', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        ai = None
        bi = None
        if not index is None:
            assert isInt(index), ValueError('index must be an integer')
            ai = np.s_[index,:,:]
            bi = np.s_[index,:]

        self.arr.writeHdf(parent, myName+'/arr',  index=ai)
        self.x.writeHdf(parent, myName+'/x',  index=bi)
        self.y.writeHdf(parent, myName+'/y',  index=bi)


    def toHdf(self, h5obj, myName):
        """ Write the StatArray to an HDF object
        h5obj: :An HDF File or Group Object.
        """
        # Create a new group inside h5obj
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self.arr.toHdf(grp, 'arr')
        self.x.toHdf(grp, 'x')
        self.y.toHdf(grp, 'y')


    def fromHdf(self, grp, index=None):
        """ Reads in the object from a HDF file """

        ai=None
        bi=None
        if (not index is None):
            assert isInt(index), ValueError('index must be an integer')
            ai = np.s_[index,:,:]
            bi = np.s_[index,:]

        item = grp.get('arr')
        obj = eval(safeEval(item.attrs.get('repr')))
        arr = obj.fromHdf(item, index=ai)
        item = grp.get('x')
        obj = eval(safeEval(item.attrs.get('repr')))
        x = obj.fromHdf(item, index=bi)
        item = grp.get('y')
        obj = eval(safeEval(item.attrs.get('repr')))
        y = obj.fromHdf(item, index=bi)
        tmp = RectilinearMesh2D(x, y)
        tmp.arr = arr
        return tmp

    def xRange(self):
        """ Get the range of x

        Returns
        -------
        out : numpy.float64
            The range of x

        """

        return np.float64(self.x[-1] - self.x[0])

    def yRange(self):
        """ Get the range of y

        Returns
        -------
        out : numpy.float64
            The range of y

        """

        return np.float64(self.y[-1] - self.y[0])
