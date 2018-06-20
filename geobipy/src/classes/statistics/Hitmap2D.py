""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
#from ...base import Error as Err
import numpy as np
from ...base.customFunctions import safeEval
from ..core.StatArray import StatArray
from .Histogram2D import Histogram2D
from ...base.customFunctions import _logSomething, isInt

class Hitmap2D(Histogram2D):
    """ Class defining a 2D hitmap whose cells are rectangular with linear sides """

    def varianceCutoff(self, percent=67.0):
        """ Get the cutoff value along y axis from the bottom up where the variance is percent*max(variance) """
        p = 0.01*percent
        s = (np.repeat(self.x[np.newaxis,:],np.size(self.arr,0),0) * self.arr).std(axis = 1)
        mS = s.max()
        iC = s.searchsorted(p*mS,side='right')-1

        return self.y[iC]


    def getConfidenceRange(self, percent=95.0, log=None):
        """ Get the range of confidence with depth """
        sMed, sLow, sHigh = self.getConfidenceIntervals(percent, log=log)

        return sHigh - sLow


    def getOpacity(self, percent=95.0):
        """ Return an opacity with depth between 0 and 1 based on the 95% confidence inverval of the hitmap """
        opacity = self.getConfidenceRange(percent=percent)
        maxes = np.max(opacity)
        if (maxes == 0.0): return opacity
        opacity /= maxes
        opacity = 1.0 - opacity
        return opacity


    def getOpacityLevel(self, percent):
        """ Get the index along axis 1 from the bottom up that corresponds to the percent opacity """
        p = 0.01*percent
        op = self.getOpacity()[::-1]
        nz = op.size - 1
        iC = 0
        while op[iC] < p and iC < nz:
            iC +=1
        return self.y[op.size - iC -1]


    def hdfName(self):
        """ Reprodicibility procedure """
        return('Hitmap2D()')

    def fromHdf(self, grp, index=None):
        """ Reads in the object froma HDF file """

        ai=None
        bi=None
        if (not index is None):
            assert isInt(index), TypeError('index must be an integer')
            ai = np.s_[index,:,:]
            bi = np.s_[index,:]

        item = grp.get('arr')
        obj = eval(safeEval(item.attrs.get('repr')))
        arr = obj.fromHdf(item, index=ai)
        item = grp.get('x')
        this = eval(safeEval(item.attrs.get('repr')))
        x = this.fromHdf(item, index=bi)
        item = grp.get('y')
        this = eval(safeEval(item.attrs.get('repr')))
        y = this.fromHdf(item, index=bi)
        tmp = Hitmap2D(x, y)
        tmp.arr = arr
        return tmp
