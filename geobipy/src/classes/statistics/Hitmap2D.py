""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
#from ...base import Error as Err
from copy import deepcopy
import numpy as np
from ...base import customFunctions as cF
from ..core import StatArray
from ..mesh.RectilinearMesh1D import RectilinearMesh1D
from .Histogram2D import Histogram2D

class Hitmap2D(Histogram2D):
    """ Class defining a 2D hitmap whose cells are rectangular with linear sides """

    def deepcopy(self):
        return deepcopy(self)


    def __deepcopy__(self, memo):
        """ Define the deepcopy. """

        if self.xyz:
            out = Hitmap2D(xBins=self.xBins, yBins=self.yBins, zBins=self.zBins)
        else:
            out = Hitmap2D(xBins=self.xBins, yBins=self.yBins)
        out._counts = self._counts.deepcopy()

        return out


    def varianceCutoff(self, percent=67.0):
        """ Get the cutoff value along y axis from the bottom up where the variance is percent*max(variance) """
        p = 0.01*percent
        s = (np.repeat(self.xBinCentres[np.newaxis,:],np.size(self.counts,0),0) * self.counts).std(axis = 1)
        mS = s.max()
        iC = s.searchsorted(p*mS,side='right')-1

        return self.yBinCentres[iC]


    def getOpacityLevel(self, percent, log=None):
        """ Get the index along axis 1 from the bottom up that corresponds to the percent opacity """
        p = 0.01*percent
        op = self.opacity(log=log)[::-1]
        nz = op.size - 1
        iC = 0
        while op[iC] < p and iC < nz:
            iC +=1
        return self.y.cellCentres[op.size - iC -1]


    def hdfName(self):
        """ Reprodicibility procedure """
        return('Hitmap2D()')


    def fromHdf(self, grp, index=None):
        """ Reads in the object froma HDF file """

        ai=None
        bi=None
        if (not index is None):
            assert cF.isInt(index), TypeError('index must be an integer {}'.format(index))
            ai = np.s_[index, :, :]
            bi = np.s_[index, :]

        item = grp.get('arr')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        arr = obj.fromHdf(item, index=ai)

        item = grp.get('x')
        this = eval(cF.safeEval(item.attrs.get('repr')))
        x = this.fromHdf(item, index=bi)
        item = grp.get('y')
        this = eval(cF.safeEval(item.attrs.get('repr')))
        y = this.fromHdf(item, index=bi)

        tmp = Hitmap2D(xBins=x.cellEdges, yBins=y.cellEdges)
        tmp._counts[:, :] = arr

        return tmp
