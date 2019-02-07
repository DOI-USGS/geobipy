""" @LineResults
Class to handle the HDF5 result files for a line of data.
 """
#from ..base import Error as Err
import numpy as np
import numpy.ma as ma
import h5py
from ..classes.core.myObject import myObject
from ..classes.core.StatArray import StatArray
from ..classes.statistics.Distribution import Distribution
from ..classes.statistics.Histogram1D import Histogram1D
from ..classes.statistics.Histogram2D import Histogram2D
from ..classes.statistics.Hitmap2D import Hitmap2D
from ..classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from ..classes.mesh.TopoRectilinearMesh2D import TopoRectilinearMesh2D
from ..base.HDF import hdfRead
from ..base import customPlots as cP
import matplotlib.pyplot as plt
from os.path import split
from ..base import fileIO as fIO
from geobipy.src.inversion.Results import Results

try:
    from pyvtk import VtkData, UnstructuredGrid, CellData, Scalars
except:
    pass

class LineResults(myObject):
    """ Class to define results from EMinv1D_MCMC for a line of data """
    def __init__(self, fName=None, sysPath=None, hdfFile=None):
        """ Initialize the lineResults """
        if (fName is None): return

        self.addErr = None
        self.best = None
        self.bestData = None
        self.bestModel = None
        self.burnedIn = None
        self.doi = None
        self.elevation = None
        self.facies = None
        self.iDs = None
        self.interfaces = None
        self.hitMap = None
        self.k = None
        self.mean = None
        self.nPoints = None
        self.nSys = None
        self.opacity = None
        self.range = None
        self.relErr = None
        self.sysPath=sysPath
        self.totErr = None
        self.x = None
        self.y = None
        self.z = None
        self.depthGrid = None

        self.mesh = None

        self.fName = fName
        self.line = split(fName)[1]
        self.hdfFile = None
        if (hdfFile is None): # Open the file for serial read access
            self.open()
            self.getIDs()
        else:
            self.hdfFile = hdfFile
            self.getIDs()

        self.getDataLocations()


    def open(self):
        """ Check whether the file is open """
        try:
            self.hdfFile.attrs
        except:
            self.hdfFile = h5py.File(self.fName,'r+')


    def close(self):
        """ Check whether the file is open """
        if (self.hdfFile is None): return
        try:
            self.hdfFile.close()
        except:
            pass # Already closed


    def crossplotErrors(self, system=0, **kwargs):
        """ Create a crossplot of the relative errors against additive errors for the most probable data point, for each data point along the line """
        self.getAdditiveError()
        self.getRelativeError()

        self.getNsys()
        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','none')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.0)

        if (self.nSys > 1):
            r = range(self.nSys)
            for i in r:
                fc = cP.wellSeparated[i+2]
                cP.plot(x=self.relErr[:,i], y=self.addErr[:,i], c=fc,
                    alpha = 0.7,label='System ' + str(i + 1), **kwargs)

            plt.legend()

        else:
            fc = cP.wellSeparated[2]
            cP.plot(x=self.relErr, y=self.addErr, c=fc,
                    alpha = 0.7,label='System ' + str(1), **kwargs)

        cP.xlabel(self.relErr.getNameUnits())
        cP.ylabel(self.addErr.getNameUnits())
        


    def getMesh(self):
        """Get the 2D topo fitting rectilinear mesh. """
        
        if (not self.mesh is None):
            return
        self.getDataLocations()
        if (self.hitMap is None):
            tmp = self.getAttribute('hitmap/y', index=0)
        else:
            tmp = self.hitMap.y

        self.mesh = TopoRectilinearMesh2D(xCentres=self.x, yCentres=self.y, zEdges=tmp.edges(min=0.0), heightCentres=self.elevation)


    def getAdditiveError(self):
        """ Get the Additive error of the best data points """
        if (not self.addErr is None): return
        self.addErr = self.getAttribute('Additive Error')


    def getBestData(self, **kwargs):
        """ Get the best data """

        if (not self.bestData is None): return
        self.bestData = self.getAttribute('best data', **kwargs)


    def getBestParameters(self):
        """ Get the best model of the parameters """
        if (not self.best is None): return
        self.best = StatArray(self.getAttribute('bestinterp'), dtype=np.float64)
        self.best.name = "Best resistivity"
        self.best.units = "$\Omega m$"


    def getDataLocations(self):
        """Get the co-ordinates of the observation locations. """
        self.getX()
        self.getY()
        self.getHeight()
        self.getElevation()


    def getDOI(self,percent=67.0):
        """ Get the DOI of the line depending on a percentage variance cutoff for each data point """
        #if (not self.doi is None): return
        self.getOpacity()
        self.getMesh()
        p = 0.01*(100.0 - percent)

        self.doi = np.zeros(self.nPoints)
        zSize = self.opacity.shape[1]-1
        r = range(self.nPoints)
        for i in r:
            op = self.opacity[i,:][::-1]
            iC = 0
            while op[iC] < p and iC < zSize:
                iC +=1
            self.doi[i]=self.depthGrid.cellCentres[zSize - iC]


    def getElevation(self):
        """ Get the elevation of the data points """
        if (not self.elevation is None): return
        self.elevation = StatArray(np.asarray(self.getAttribute('elevation')), 'Elevation', 'm')


    def getHeight(self):
        """Get the height of the observations. """
        if (not self.z is None): return
        self.z = self.getAttribute('z')


    def getHitMap(self):
        """ Get the hitmaps for each data point """
        print('Be careful with .getHitMap(),  it could require a lot of memory.')
        if (not self.hitMap is None): return
        self.hitMap = self.getAttribute('Hit Map', index=0)


    def getIDs(self):
        """ Get the id numbers of the data points in the line results file """
        if (not self.iDs is None): return

        self.iDs = np.asarray(self.hdfFile.get('ids'))
        self.nPoints = self.iDs.size


    def getInterfaces(self, cut=0.0):
        """ Get the layer interfaces from the layer depth histograms """
        if (not self.interfaces is None): return

        tmp = self.getAttribute('layer depth histogram')

        maxCount = tmp.counts.max()
        self.interfaces = tmp.counts / np.float64(maxCount)
        self.interfaces[self.interfaces < cut] = np.nan
        self.interfaces.name = "Interfaces"


    def getKlayers(self):
        """ Get the number of layers in the best model for each data point """
        if (not self.k is None): return
        self.k = StatArray(self.getAttribute('# Layers'), '# of Cells')


    def getMeanParameters(self):
        """ Get the mean model of the parameters """
        if (not self.mean is None): return
        self.mean = StatArray(self.getAttribute('meaninterp'), dtype=np.float64)
        self.mean.name = "Mean resistivity"
        self.mean.units = "$\Omega m$"


    def getNsys(self):
        """ Get the number of systems """
        if (not self.nSys is None): return
        self.nSys = self.getAttribute('# of systems')


    def getOpacity(self, percent=95.0, multiplier=0.5, log='e'):
        """ Get the model parameter opacity using the confidence intervals """
        if (not self.opacity is None): return

        self.getMesh()
        self.opacity = StatArray(np.zeros([self.mesh.dims[1], self.mesh.dims[0]]), 'Opacity')

        a = np.asarray(self.hdfFile['hitmap/arr/data'])
        try:
            b = np.asarray(self.hdfFile['hitmap/x/data'])
        except:
            b = np.asarray(self.hdfFile['hitmap/x/x/data'])

        try:
            c = np.asarray(self.hdfFile['hitmap/y/data'])
        except:
            c = np.asarray(self.hdfFile['hitmap/y/y/data'])

        yTmp = StatArray(c[0, :])
        h = Hitmap2D(xBinCentres = StatArray(b[0, :]), yBinCentres = yTmp)
        h._counts[:, :] = a[0, :, :]
        self.opacity[0, :] = h.confidenceRange(percent=percent, log=log)

        for i in range(1, self.nPoints):
            h.x.xBinCentres = b[i, :]
            h._counts[:, :] = a[i, :, :]
            self.opacity[i, :] = h.confidenceRange(percent=percent, log=log)

        self.opacity = self.opacity.T

        high = multiplier * (self.opacity.max() - self.opacity.min())

        self.opacity[self.opacity > high] = high

        self.opacity /= high

        self.opacity = 1.0 - self.opacity


    def getRelativeError(self):
        """ Get the Relative error of the best data points """
        if (not self.relErr is None): return
        self.relErr = self.getAttribute('Relative Error')


    def getResults(self, index=None, fid=None):
        """ Obtain the results for the given iD number """

        assert not (index is None and fid is None), Exception("Please specify either an integer index or a fiducial.")
        assert index is None or fid is None, Exception("Only specify either an integer index or a fiducial.")

        if not fid is None:
            assert fid in self.iDs, "The HDF file was not initialized to contain the results for this datapoints results "
            # Get the point index
            i = self.iDs.searchsorted(fid)
        else:
            i = index

        aFile = self.hdfFile

        s = np.s_[i, :]

        R = Results()

        R.iPlot = np.array(aFile.get('iplot'))
        R.plotMe = np.array(aFile.get('plotme'))
        R.limits = np.array(aFile.get('limits'))
        R.invertPar = np.array(aFile.get('invertPar'))
        R.nMC = np.array(aFile.get('nmc'))
        R.nSystems = np.array(aFile.get('nsystems'))
        R.ratex = hdfRead.readKeyFromFile(aFile,'','/','ratex')

        R.i = hdfRead.readKeyFromFile(aFile,'','/','i', index=i)
        R.iBurn = hdfRead.readKeyFromFile(aFile,'','/','iburn', index=i)
        R.burnedIn = hdfRead.readKeyFromFile(aFile,'','/','burnedin', index=i)
        R.doi = hdfRead.readKeyFromFile(aFile,'','/','doi', index=i)
        R.multiplier = hdfRead.readKeyFromFile(aFile,'','/','multiplier', index=i)
        R.rate = hdfRead.readKeyFromFile(aFile,'','/','rate', index=s)
        R.PhiDs = hdfRead.readKeyFromFile(aFile,'','/','phids', index=s)
        R.Hitmap = hdfRead.readKeyFromFile(aFile,'','/','hitmap', index=i)
        R.bestD = hdfRead.readKeyFromFile(aFile,'','/','bestd', index=i, sysPath=self.sysPath)
        #R.currentD = hdfRead.readKeyFromFile(aFile,'','/','currentd', index=i, sysPath=self.sysPath)
        R.bestModel = hdfRead.readKeyFromFile(aFile,'','/','bestmodel', index=i)
        R.bestModel.maxDepth = np.log(R.Hitmap.y.cellCentres[-1])
        R.kHist = hdfRead.readKeyFromFile(aFile,'','/','khist', index=i)
        R.DzHist = hdfRead.readKeyFromFile(aFile,'','/','dzhist', index=i)
        R.MzHist = hdfRead.readKeyFromFile(aFile,'','/','mzhist', index=i)


        # R.DzHist.bins -= (R.DzHist.bins[int(R.DzHist.bins.size/2)] - R.bestD.z[0])

        R.relErr = []
        R.addErr = []
        for j in range(R.nSystems):
            R.relErr.append(hdfRead.readKeyFromFile(aFile,'','/','relerr'+str(j), index=i))
            R.addErr.append(hdfRead.readKeyFromFile(aFile,'','/','adderr'+str(j), index=i))


        R.invTime=np.array(aFile.get('invtime')[i])
        R.saveTime=np.array(aFile.get('savetime')[i])

        # Initialize a list of iteration number
        R.iRange = StatArray(np.arange(2 * R.nMC), name="Iteration #", dtype=np.int64)

        R.verbose = False

        return R


    def getTotalError(self):
        """ Get the total error of the best data points """
        if (not self.totErr is None): return

        self.totErr = self.getAttribute('Total Error')


    def getX(self):
        """ Get the X co-ordinates (Easting) """
        if (not self.x is None):
            return
        self.x = self.getAttribute('x')
        if self.x.name in [None, '']:
            self.x.name = 'Easting'
        if self.x.units in [None, '']:
            self.x.units = 'm'


    def getY(self):
        """ Get the Y co-ordinates (Easting) """
        if (not self.y is None):
            return
        self.y = self.getAttribute('y')

        if self.y.name in [None, '']:
            self.y.name = 'Northing'
        if self.y.units in [None, '']:
            self.y.units = 'm'


    def plotAllBestData(self, **kwargs):
        """ Plot a channel of data as points """

        self.setAlonglineAxis(self.plotAgainst)

        self.getBestData()

        cP.pcolor(self.bestData.p.T, x=self.xPlot, y=StatArray(np.arange(self.bestData.p.shape[1]), name='Channel'), **kwargs)


    def plotBestDataChannel(self, channel=None, **kwargs):
        """ Plot a channel of the best predicted data as points """

        self.setAlonglineAxis(self.plotAgainst)

        self.getBestData(sysPath = self.sysPath)

        if channel is None:
            channel = np.s_[:]

        cP.plot(self.xPlot, self.bestData.p[:, channel], **kwargs)


    def plotDataElevation(self, **kwargs):
        """ Adds the data elevations to a plot """

        self.getDataLocations()

        axis = kwargs.pop('axis', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color','k')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.5)

        xtmp = self.mesh.getXAxis(axis, centres=False)

        cP.plot(xtmp, self.z.edges() + self.elevation.edges(), **kwargs)

        if (labels):
            cP.xlabel(xtmp.getNameUnits())
            cP.ylabel('Elevation (m)')


    def plotDoi(self, percent=67.0, **kwargs):

        self.setAlonglineAxis(self.plotAgainst)
        self.getElevation()
        self.getDOI(percent)

        labels = kwargs.pop('labels', True)
        c = kwargs.pop('color','k')
        lw = kwargs.pop('linewidth',0.5)

        cP.plot(self.xPlot, self.elevation - self.doi, color=c, linewidth=lw, **kwargs)

        if (labels):
            cP.xlabel(self.xPlot.getNameUnits())
            cP.ylabel('Elevation (m)')


    def plotElevation(self, **kwargs):

        self.getMesh()

        axis = kwargs.pop('axis', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color','k')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.5)

        self.mesh.plotHeight(axis=axis, **kwargs)

        # if (labels):
        #     cP.xlabel(xtmp.getNameUnits())
        #     cP.ylabel('Elevation (m)')


    def plotElevationDistributions(self, **kwargs):
        """ Plot the horizontally stacked elevation histograms for each data point along the line """

        self.getMesh()
        tmp = self.getAttribute('elevation histogram')

        axis = kwargs.pop('axis', 'x')

        xtmp = self.mesh.getXAxis(axis)

        c = tmp.counts.T
        c = np.divide(c, np.max(c,0), casting='unsafe')
        x = np.zeros(xtmp.size+1)
        d = np.diff(xtmp)
        c.pcolor(xtmp, y=tmp.bins, **kwargs)
        cP.title('Data Elevation posterior distributions')


    def plotHighlightedObservationLocations(self, iDs, **kwargs):

        self.setAlonglineAxis(self.plotAgainst)
        # Get the data heights
        if (self.z is None):
            self.z = self.getAttribute('z')
        self.getElevation()

        labels = kwargs.pop('labels', True)
        m = kwargs.pop('marker','*') # Downward pointing arrow
        c = kwargs.pop('color',cP.wellSeparated[1])
        ls = kwargs.pop('linestyle','none')
        mec = kwargs.pop('markeredgecolor','k')
        mew = kwargs.pop('markeredgewidth','0.1')

        i = self.iDs.searchsorted(iDs)

        tmp=self.z.reshape(self.z.size) + self.elevation

        plt.plot(self.xPlot[i], tmp[i], color=c, marker=m, markeredgecolor=mec, linestyle=ls, markeredgewidth=mew, **kwargs)

        if (labels):
            cP.xlabel(self.xPlot.getNameUnits())
            cP.ylabel('Elevation (m)')


    def plotKlayers(self, **kwargs):
        """ Plot the number of layers in the best model for each data point """
        self.getMesh()
        self.getKlayers()
        
        axis = kwargs.pop('axis', 'x')
        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth', 1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','none')

        xtmp = self.mesh.getXAxis(axis)
        self.k.plot(xtmp, **kwargs)
        # cP.ylabel(self.k.getNameUnits())
        cP.title('# of Layers in Best Model')


    def plotKlayersDistributions(self, **kwargs):
        """ Plot the horizontally stacked elevation histograms for each data point along the line """

        self.getMesh()
        tmp = self.getAttribute('layer histogram')

        axis = kwargs.pop('axis', 'x')

        xtmp = self.mesh.getXAxis(axis)

        c = tmp.counts.T
        c = np.divide(c, np.max(c,0), casting='unsafe')
        ax = c.pcolor(xtmp, y=tmp.bins, **kwargs)
        cP.title('# of Layers posterior distributions')


    def plotAdditiveError(self, **kwargs):
        """ Plot the relative errors of the data """
        self.getAdditiveError()
        self.getNsys()

        axis = kwargs.pop('axis', 'x')
        m = kwargs.pop('marker','o')
        ms = kwargs.pop('markersize',5)
        mfc = kwargs.pop('markerfacecolor',None)
        mec = kwargs.pop('markeredgecolor','k')
        mew = kwargs.pop('markeredgewidth',1.0)
        ls = kwargs.pop('linestyle','-')
        lw = kwargs.pop('linewidth',1.0)

        xtmp = self.mesh.getXAxis(axis, centres=True)

        if (self.nSys > 1):
            r = range(self.nSys)
            for i in r:
                fc = cP.wellSeparated[i+2]
                cP.plot(xtmp, y=self.addErr[:,i],
                    marker=m,markersize=ms,markerfacecolor=mfc,markeredgecolor=mec,markeredgewidth=mew,
                    linestyle=ls,linewidth=lw,c=fc,
                    alpha = 0.7,label='System ' + str(i + 1), **kwargs)
            plt.legend()
        else:
            fc = cP.wellSeparated[2]
            cP.plot(xtmp, y=self.addErr,
                    marker=m,markersize=ms,markerfacecolor=mfc,markeredgecolor=mec,markeredgewidth=mew,
                    linestyle=ls,linewidth=lw,c=fc,
                    alpha = 0.7,label='System ' + str(1), **kwargs)

        # cP.xlabel(xtmp.getNameUnits())
        # cP.ylabel(self.addErr.getNameUnits())
        


    def plotAdditiveErrorDistributions(self, system=0, **kwargs):
        """ Plot the distributions of additive errors as an image for all data points in the line """
        self.getMesh()
        self.getNsys()
        tmp=self.getAttribute('Additive error histogram')

        axis = kwargs.pop('axis', 'x')

        xtmp = self.mesh.getXAxis(axis)

        if self.nSys > 1:
            c = tmp[system].counts.T
            c = np.divide(c, np.max(c,0), casting='unsafe')
            c.pcolor(xtmp, y=tmp[system].bins, **kwargs)
            cP.title('Additive error posterior distributions for system {}'.format(system))
        else:
            c = tmp.counts.T
            c = np.divide(c, np.max(c,0), casting='unsafe')
            c.pcolor(xtmp, y=tmp.bins, **kwargs)
            cP.title('Additive error posterior distributions')


    def plotError2DJointProbabilityDistribution(self, datapoint, system=0, **kwargs):
        """ For a given datapoint, obtains the posterior distributions of relative and additive error and creates the 2D joint probability distribution """

        # Read in the histogram of relative error for the data point
        rel=self.getAttribute('Relative error histogram', index=datapoint)
        # Read in the histogram of additive error for the data point
        add=self.getAttribute('Additive error histogram', index=datapoint)

        joint = Histogram2D()
        joint.create2DjointProbabilityDistribution(rel[system],add[system])

        joint.pcolor(**kwargs)


    def plotInterfaces(self, cut=0.0, useVariance=True, **kwargs):
        """ Plot a cross section of the layer depth histograms. Truncation is optional. """

        self.getMesh()
        self.getInterfaces(cut=cut)

        kwargs['noColorbar'] = kwargs.pop('noColorbar', True)

        if useVariance:
            self.getOpacity()
            kwargs['alpha'] = self.opacity

        pm = self.mesh.pcolor(self.interfaces.T, **kwargs)


    def plotObservedDataChannel(self, channel=None, **kwargs):
        """ Plot a channel of the observed data as points """

        self.setAlonglineAxis(self.plotAgainst)

        self.getBestData()

        if channel is None:
            channel = np.s_[:]

        # print(np.min(self.bestData.d[:, channel]))

        cP.plot(self.xPlot, self.bestData.d[:, channel], **kwargs)


    def plotOpacity(self, log='e', **kwargs):
        """ Plot the opacity """

        self.getMesh()
        self.getOpacity(log=log)
        kwargs.pop('log', None)

        self.mesh.pcolor(values = self.opacity, **kwargs)


    def plotRelativeErrorDistributions(self, system=0, **kwargs):
        """ Plot the distributions of relative errors as an image for all data points in the line """
        self.getMesh()
        self.getNsys()

        tmp=self.getAttribute('Relative error histogram')

        axis = kwargs.pop('axis', 'x')

        xtmp = self.mesh.getXAxis(axis)

        if self.nSys > 1:
            c = tmp[system].counts.T
            c = np.divide(c, np.max(c,0), casting='unsafe')
            c.pcolor(xtmp, y=tmp[system].bins, **kwargs)
            cP.title('Relative error posterior distributions for system {}'.format(system))
        else:
            c = tmp.counts.T
            c = np.divide(c, np.max(c,0), casting='unsafe')
            c.pcolor(xtmp, y=tmp.bins, **kwargs)
            cP.title('Relative error posterior distributions')


    def plotRelativeError(self, **kwargs):
        """ Plot the relative errors of the data """
        self.getMesh()
        self.getRelativeError()
        self.getNsys()

        axis = kwargs.pop('axis', 'x')
        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','-')
        kwargs['linewidth'] = kwargs.pop('linewidth',1.0)

        xtmp = self.mesh.getXAxis(axis)

        if (self.nSys > 1):
            r = range(self.nSys)
            for i in r:
                kwargs['c'] = cP.wellSeparated[i+2]
                self.relErr[:, i].plot(xtmp,
                    alpha = 0.7, label='System {}'.format(i + 1), **kwargs)
            plt.legend()
        else:
            kwargs['c'] = cP.wellSeparated[2]
            self.relErr.plot(xtmp,
                    alpha = 0.7, label='System {}'.format(1), **kwargs)


    def plotTotalError(self, channel, **kwargs):
        """ Plot the relative errors of the data """

        self.getMesh()
        self.getTotalError()

        axis = kwargs.pop('axis', 'x')

        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','-')
        kwargs['linewidth'] = kwargs.pop('linewidth',1.0)

        xtmp = self.mesh.getXAxis(axis)

#        if (self.nSys > 1):
#            r = range(self.nSys)
#            for i in r:
#                fc = cP.wellSeparated[i+2]
#                cP.plot(x=self.xPlot, y=self.addErr[:,i],
#                    marker=m,markersize=ms,markerfacecolor=mfc,markeredgecolor=mec,markeredgewidth=mew,
#                    linestyle=ls,linewidth=lw,c=fc,
#                    alpha = 0.7,label='System ' + str(i + 1), **kwargs)
#        else:
        fc = cP.wellSeparated[2]
        self.totErr[:,channel].plot(xtmp,
                alpha = 0.7, label='Channel ' + str(channel), **kwargs)

#        cP.xlabel(self.xPlot.getNameUnits())
#        cP.ylabel(self.addErr.getNameUnits())
#        plt.legend()


    def plotTotalErrorDistributions(self, channel=0, nBins=100, **kwargs):
        """ Plot the distributions of relative errors as an image for all data points in the line """
        self.getTotalError()
        self.setAlonglineAxis(self.plotAgainst)

        H = Histogram1D(values=np.log10(self.totErr[:,channel]),nBins=nBins)

        H.plot(**kwargs)

#        if self.nSys > 1:
#            c = tmp[system].counts.T
#            c = np.divide(c, np.max(c,0), casting='unsafe')
#            c.pcolor(x=self.xPlot, y=tmp[system].bins, **kwargs)
#            cP.title('Relative error posterior distributions for system '+str(system))
#        else:
#            c = tmp.counts.T
#            c = np.divide(c, np.max(c,0), casting='unsafe')
#            c.pcolor(x=self.xPlot, y=tmp.bins, **kwargs)
#            cP.title('Relative error posterior distributions')

    def histogram(self, nBins, depth1 = None, depth2 = None, invertPar = False, bestModel = False, **kwargs):
        """ Compute a histogram of the model, optionally show the histogram for given depth ranges instead """

        self.getMesh()

        if (depth1 is None):
            depth1 = self.mesh.z.cellEdges[0]
        if (depth2 is None):
            depth2 = self.mesh.z.cellEdges[-1]

        maxDepth = self.mesh.z.cellEdges[-1]

        # Ensure order in depth values
        if (depth1 > depth2):
            tmp = depth2
            depth2 = depth1
            depth1 = tmp

        # Don't need to check for depth being shallower than self.mesh.y.cellEdges[0] since the sortedsearch will return 0
        assert depth1 <= maxDepth, ValueError('Depth1 is greater than max depth {}'.format(maxDepth))
        assert depth2 <= maxDepth, ValueError('Depth2 is greater than max depth {}'.format(maxDepth))

        cell1 = self.mesh.z.cellIndex(depth1, clip=True)
        cell2 = self.mesh.z.cellIndex(depth2, clip=True)

        if (bestModel):
            self.getBestParameters()
            model = self.best
            title = 'Best model values between {:.3f} m and {:.3f} m depth'.format(depth1, depth2)
        else:
            self.getMeanParameters()
            model = self.mean
            title = 'Mean model values between {:.3f} m and {:.3f} m depth'.format(depth1, depth2)

        vals = model[:, cell1:cell2+1].deepcopy()

        log = kwargs.pop('log',False)

        if (invertPar):
            i = np.where(vals > 0.0)[0]
            vals[i] = 1.0 / vals[i]
            name = 'Resistivity'
            units = '$\Omega m$'
        else:
            name = 'Conductivity'
            units = '$Sm^{-1}$'

        if (log):
            vals, logLabel = cP._logSomething(vals,log)
            name = logLabel + name
        binEdges = StatArray(np.linspace(np.nanmin(vals),np.nanmax(vals),nBins+1), name, units)

        h = Histogram1D(bins = binEdges)
        h.update(vals)
        h.plot(**kwargs)
        cP.title(title)


    def plotXsection(self, invertPar = True, bestModel=False, percent = 67.0, useVariance=True, **kwargs):
        """ Plot a cross-section of the parameters """

        self.getMesh()

        if (bestModel):
            self.getBestParameters()
            tmp = self.best.T
        else:
            self.getMeanParameters()
            tmp = self.mean.T

        if (invertPar):
            tmp = 1.0 / tmp
            tmp.name = 'Resistivity'
            tmp.units = '$\Omega m$'
        else:
            tmp.name = 'Conductivity'
            tmp.units = '$Sm^{-1}$'

        if useVariance:
            self.getOpacity()
            kwargs['alpha'] = self.opacity

        self.mesh.pcolor(values = tmp, **kwargs)


    def plotFacies(self, mean, var, volFrac, percent=67.0, ylim=None):
        """ Plot a cross-section of the parameters """

        assert False, ValueError('Double check this')

        if (self.hitMap is None):
            self.hitMap = self.getAttribute('Hit Map')
        self.setAlonglineAxis(self.plotAgainst)
        self.getElevation()
        self.getMesh()
        zGrd = self.zGrid

        self.getOpacity()
        a = np.zeros([zGrd.size, self.nPoints + 1],order = 'F')  # Transparency amounts
        a[:, 1:] = self.opacity

        self._getX_pmesh(zGrd.size)
        self._getZ_pmesh(zGrd)

        c = np.zeros(self._xMesh.shape, order = 'F')  # Colour value of each cell
        c[:-1, 0] = np.nan
        c[:-1, -1] = np.nan
        c[-1, :] = np.nan

        self.assignFacies(mean, var, volFrac)
        c[:-1, 1:-1] = self.facies

        # Mask the Nans in the colorMap
        cm = ma.masked_invalid(c)

        # Get the "depth of investigation"
        self.getDOI(percent)

        ax = plt.gca()
        cP.pretty(ax)
        p = ax.pcolormesh(self._xMesh, self._zMesh, cm)
        cb = plt.colorbar(p, ax=ax)
        plt.plot(self.xPlot, self.elevation, color='k')
        if (self.z is None):
            self.z = np.asarray(self.getAttribute('z'))
        plt.plot(self.xPlot, self.z.reshape(self.z.size) + self.elevation, color='k')

        plt.plot(self.xPlot, self.elevation - self.doi, color='k', linestyle='-', alpha=0.7, linewidth=1)

        plt.savefig('myfig.png')
        for i, j in zip(p.get_facecolors(), a.flatten()):
            i[3] = j  # Set the alpha value of the RGBA tuple using m2
        fIO.deleteFile('myfig.png')

        if (not ylim is None):
            plt.ylim(ylim)

        cP.xlabel(self.xPlot.getNameUnits())
        cP.ylabel('Elevation (m)')
        cP.clabel(cb, 'Facies')


    def assignFacies(self, mean, var, volFrac):
        """ Assign facies to the parameter model given the pdfs of each facies
        mean:    :Means of the normal distributions for each facies
        var:     :Variance of the normal distributions for each facies
        volFrac: :Volume fraction of each facies
        """

        assert False, ValueError('Double check this')

        nFacies = len(mean)
        if (self.hitMap is None):
            self.hitMap = self.getAttribute('Hit Map')
        hitMap = self.hitMap[0]
        p = hitMap.x
        # Initialize the normalized probability distributions for the facies
        faciesPDF = np.zeros([nFacies, p.size])
        pTmp = np.log10(1.0 / p)
        for i in range(nFacies):
            tmpHist = Distribution('normal', mean[i], var[i])
            faciesPDF[i, :] = volFrac[i] * tmpHist.getPdf(pTmp)
            # Precompute the sum of the faciesPDF rows
        pdfSum = np.sum(faciesPDF, 0)
        # Compute the denominator
        denominator = 1.0 / \
            np.sum(np.repeat(pdfSum[np.newaxis, :], hitMap.y.size, axis=0), 1)
        # Initialize the facies Model
        self.facies = np.zeros([hitMap.y.size, self.nPoints], order = 'F')
        faciesWithDepth = np.zeros([hitMap.y.size, nFacies], order = 'F')
        for i in range(self.nPoints):
            for j in range(nFacies):
                fTmp = faciesPDF[j, :]
                faciesWithDepth[:, j] = np.sum(self.hitMap[i]._counts * np.repeat(fTmp[np.newaxis, :], hitMap.y.size, axis=0), 1) * denominator
            self.facies[:, i] = np.argmax(faciesWithDepth, 1)


    def plotDataPointResults(self, fid, sysPath=None):
        """ Plot the geobipy results for the given data point """
        R = self.getResults(fid=fid)
        R.initFigure(forcePlot=True)
        R.plot(forcePlot=True)


    def toVtk(self, fileName, format='binary'):
        """Write the parameter cross-section to an unstructured grid vtk file 
        
        Parameters
        ----------
        fileName : str
            Filename to save to.
        format : str, optional
            "ascii" or "binary" format. Ascii is readable, binary is not but results in smaller files.

        """
        self.getMesh()
        self.getBestParameters()
        self.getMeanParameters()
        self.getInterfaces()

        a = self.best.T
        b = self.mean.T
        c = self.interfaces.T

        d = StatArray(1.0 / a, "Best Conductivity", "$\fraq{S}{m}$")
        e = StatArray(1.0 / b, "Mean Conductivity", "$\fraq{S}{m}$")

        self.mesh.toVTK(fileName, format=format, cellData=[a, b, c, d, e])
        

    def getAttribute(self, attribute, iDs = None, index=None, **kwargs):
        """ Gets an attribute from the line results file """
        assert (not attribute is None), "Please specify an attribute: \n"+self.possibleAttributes()

        old = False
        if (old):
            keys = self._attrTokeyOld(attribute)
        else:
            keys = self._attrTokey(attribute)

        if (iDs is None):
            iDs = ['/']

        return hdfRead.readKeyFromFile(self.hdfFile, self.fName, iDs, keys, index=index, **kwargs)


    def _attrTokey(self,attributes):
        """ Takes an easy to remember user attribute and converts to the tag in the HDF file """
        if (isinstance(attributes, str)):
            attributes = [attributes]
        res = []
        nSys= None
        for attr in attributes:
            low = attr.lower()
            if (low == 'iteration #'):
                res.append('i')
            elif (low == '# of markov chains'):
                res.append('nmc')
            elif (low == 'burned in'):
                res.append('burnedin')
            elif (low == 'burn in #'):
                res.append('iburn')
            elif (low == 'data multiplier'):
                res.append('multiplier')
            elif (low == 'layer histogram'):
                res.append('khist')
            elif (low == 'elevation histogram'):
                res.append('dzhist')
            elif (low == 'layer depth histogram'):
                res.append('mzhist')
            elif (low == 'best data'):
                res.append('bestd')
            elif (low == 'x'):
                res.append('bestd/x')
            elif (low == 'y'):
                res.append('bestd/y')
            elif (low == 'z'):
                res.append('bestd/z')
            elif (low == 'elevation'):
                res.append('bestd/e')
            elif (low == 'observed data'):
                res.append('bestd/d')
            elif (low == 'predicted data'):
                res.append('bestd/p')
            elif (low == 'total error'):
                res.append('bestd/s')
            elif (low == '# of systems'):
                res.append('nsystems')
            elif (low == 'additive error'):
                res.append('bestd/addErr')
            elif (low == 'relative error'):
                res.append('bestd/relErr')
            elif (low == 'best model'):
                res.append('bestmodel')
            elif (low == 'meaninterp'):
                res.append('meaninterp')
            elif (low == 'bestinterp'):
                res.append('bestinterp')
            elif (low == 'opacityinterp'):
                res.append('opacityinterp')
            elif (low == '# layers'):
                res.append('bestmodel/nCells')
#            elif (low == 'current data'):
#                res.append('currentd')
            elif (low == 'hit map'):
                res.append('hitmap')
            elif (low == 'hitmap/y'):
                res.append('hitmap/y')
            elif (low == 'doi'):
                res.append('doi')
            elif (low == 'data misfit'):
                res.append('phids')
            elif (low == 'relative error histogram'):
                if (nSys is None): nSys = hdfRead.readKeyFromFile(self.hdfFile, self.fName, '/','nsystems')
                for i in range(nSys):
                    res.append('relerr' +str(i))
            elif (low == 'additive error histogram'):
                if (nSys is None): nSys = hdfRead.readKeyFromFile(self.hdfFile, self.fName, '/','nsystems')
                for i in range(nSys):
                    res.append('adderr' +str(i))
            elif (low == 'inversion time'):
                res.append('invtime')
            elif (low == 'saving time'):
                res.append('savetime')
            else:
                assert False, self.possibleAttributes(attr)
        return res


    def possibleAttributes(self, askedFor=""):
        print("====================================================\n"+
              "Incorrect attribute requested " + askedFor + "\n" +
              "====================================================\n"+
              "Possible Attribute options to read in \n" +
              "iteration # \n" +
              "# of markov chains \n" +
              "burned in\n" +
              "burn in # \n" +
              "data multiplier \n" +
              "layer histogram \n" +
              "elevation histogram \n" +
              "layer depth histogram \n" +
              "best data \n" +
              "x\n" +
              "y\n" +
              "z\n" +
              "elevation\n" +
              "observed data" +
              "predicted data" +
              "total error" +
              "# of systems\n" +
              "relative error\n" +
              "best model \n" +
              "# layers \n" +
              "current data \n" +
              "hit map \n" +
              "doi \n"+
              "data misfit \n" +
              "relative error histogram\n" +
              "additive error histogram\n" +
              "inversion time\n" +
              "saving time\n"+
              "====================================================\n")

    def createHdf(self, aFile, iDs, results):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """

        self.hdfFile = aFile

        nPoints = iDs.size
        self.iDs = np.sort(iDs)

        # Initialize and write the attributes that won't change
        aFile.create_dataset('ids',data=self.iDs)
        aFile.create_dataset('iplot', data=results.iPlot)
        aFile.create_dataset('plotme', data=results.plotMe)
        aFile.create_dataset('invertPar', data=results.invertPar)
        aFile.create_dataset('limits', data=results.limits)
        aFile.create_dataset('nmc', data=results.nMC)
        aFile.create_dataset('nsystems', data=results.nSystems)
        results.ratex.toHdf(aFile,'ratex')
#        aFile.create_dataset('ratex', [results.ratex.size], dtype=results.ratex.dtype)
#        aFile['ratex'][:] = results.ratex


        # Initialize the attributes that will be written later
        aFile.create_dataset('i', shape=[nPoints], dtype=results.i.dtype, fillvalue=np.nan)
        aFile.create_dataset('iburn', shape=[nPoints], dtype=results.iBurn.dtype, fillvalue=np.nan)
        aFile.create_dataset('burnedin', shape=[nPoints], dtype=type(results.burnedIn))
        aFile.create_dataset('doi',  shape=[nPoints], dtype=results.doi.dtype, fillvalue=np.nan)
        aFile.create_dataset('multiplier',  shape=[nPoints], dtype=results.multiplier.dtype, fillvalue=np.nan)
        aFile.create_dataset('invtime',  shape=[nPoints], dtype=float, fillvalue=np.nan)
        aFile.create_dataset('savetime',  shape=[nPoints], dtype=float, fillvalue=np.nan)

        results.meanInterp.createHdf(aFile,'meaninterp',nRepeats=nPoints, fillvalue=np.nan)
        results.bestInterp.createHdf(aFile,'bestinterp',nRepeats=nPoints, fillvalue=np.nan)
#        results.opacityInterp.createHdf(aFile,'opacityinterp',nRepeats=nPoints, fillvalue=np.nan)
#        aFile.create_dataset('meaninterp', [nPoints,nz], dtype=np.float64)
#        aFile.create_dataset('bestinterp', [nPoints,nz], dtype=np.float64)
#        aFile.create_dataset('opacityinterp', [nPoints,nz], dtype=np.float64)

        results.rate.createHdf(aFile,'rate',nRepeats=nPoints, fillvalue=np.nan)
#        aFile.create_dataset('rate', [nPoints,results.rate.size], dtype=results.rate.dtype)
        results.PhiDs.createHdf(aFile,'phids',nRepeats=nPoints, fillvalue=np.nan)
        #aFile.create_dataset('phids', [nPoints,results.PhiDs.size], dtype=results.PhiDs.dtype)

        # Create the layer histogram
        results.kHist.createHdf(aFile,'khist',nRepeats=nPoints, fillvalue=np.nan)

        # Create the Elevation histogram
        results.DzHist.createHdf(aFile,'dzhist',nRepeats=nPoints, fillvalue=np.nan)

        # Create the Interface histogram
        results.MzHist.createHdf(aFile,'mzhist',nRepeats=nPoints, fillvalue=np.nan)

        # Add the relative and additive errors
        for i in range(results.nSystems):
            results.relErr[i].createHdf(aFile,"relerr"+str(i),nRepeats=nPoints, fillvalue=np.nan)
            results.addErr[i].createHdf(aFile,"adderr"+str(i),nRepeats=nPoints, fillvalue=np.nan)

        # Add the Hitmap
        results.Hitmap.createHdf(aFile,'hitmap', nRepeats=nPoints, fillvalue=np.nan)

        # results.currentD.createHdf(aFile,'currentd', nRepeats=nPoints, fillvalue=np.nan)
        results.bestD.createHdf(aFile,'bestd', nRepeats=nPoints, fillvalue=np.nan)

        # Since the 1D models change size adaptively during the inversion, we need to pad the HDF creation to the maximum allowable number of layers.
        tmp = results.bestModel.pad(results.bestModel.maxLayers)

        tmp.createHdf(aFile,'bestmodel',nRepeats=nPoints, fillvalue=np.nan)

        if results.verbose:

            results.posteriorComponents.createHdf(aFile,'posteriorcomponents',nRepeats=nPoints, fillvalue=np.nan)

        # Add the best data components
#        aFile.create_dataset('bestdata.z', [nPoints], dtype=results.bestD.z.dtype)
#        aFile.create_dataset('bestdata.p', [nPoints,*results.bestD.p.shape], dtype=results.bestD.p.dtype)
#        aFile.create_dataset('bestdata.s', [nPoints,*results.bestD.s.shape], dtype=results.bestD.s.dtype)

        # Add the best model components
#        aFile.create_dataset('bestmodel.ncells', [nPoints], dtype=results.bestModel.nCells.dtype)
#        aFile.create_dataset('bestmodel.top', [nPoints], dtype=results.bestModel.top.dtype)
#        aFile.create_dataset('bestmodel.par', [nPoints,*results.bestModel.par.shape], dtype=results.bestModel.par.dtype)
#        aFile.create_dataset('bestmodel.depth', [nPoints,*results.bestModel.depth.shape], dtype=results.bestModel.depth.dtype)
#        aFile.create_dataset('bestmodel.thk', [nPoints,*results.bestModel.thk.shape], dtype=results.bestModel.thk.dtype)
#        aFile.create_dataset('bestmodel.chie', [nPoints,*results.bestModel.chie.shape], dtype=results.bestModel.chie.dtype)
#        aFile.create_dataset('bestmodel.chim', [nPoints,*results.bestModel.chim.shape], dtype=results.bestModel.chim.dtype)

#        self.currentD.createHdf(grp, 'currentd')
#        self.bestD.createHdf(grp, 'bestd')
#
#        tmp=self.bestModel.pad(self.bestModel.maxLayers)
#        tmp.createHdf(grp, 'bestmodel')

    def results2Hdf(self, results):
        """ Given a HDF file initialized as line results, write the contents of results to the appropriate arrays """

        assert results.ID in self.iDs, Exception("The HDF file does not have ID number {}. Available ids are between {} and {}".format(results.ID, np.min(self.iDs), np.max(self.iDs)))

        aFile = self.hdfFile

        # Get the point index
        i = self.iDs.searchsorted(results.ID)

        # Add the iteration number
        aFile['i'][i] = results.i

        # Add the burn in iteration
        aFile['iburn'][i] = results.iBurn

        # Add the burned in logical
        aFile['burnedin'][i] = results.burnedIn

        # Add the depth of investigation
        aFile['doi'][i] = results.doi

        # Add the multiplier
        aFile['multiplier'][i] = results.multiplier

        # Add the inversion time
        aFile['invtime'][i] = results.invTime

        # Add the savetime
#        aFile['savetime'][i] = results.saveTime

        # Interpolate the mean and best model to the discretized hitmap
        results.meanInterp[:] = results.Hitmap.axisMean()
        results.bestInterp[:] = results.bestModel.interpPar2Mesh(results.bestModel.par, results.Hitmap)
#        results.opacityInterp[:] = results.Hitmap.getOpacity()

        slic = np.s_[i, :]
        # Add the interpolated mean model
        results.meanInterp.writeHdf(aFile, 'meaninterp',  index=slic)
        # Add the interpolated best
        results.bestInterp.writeHdf(aFile, 'bestinterp',  index=slic)
        # Add the interpolated opacity
#        results.opacityInterp.writeHdf(aFile, 'opacityinterp',  index=slic)

        # Add the acceptance rate
        results.rate.writeHdf(aFile,'rate',index=slic)

        # Add the data misfit
        results.PhiDs.writeHdf(aFile,'phids',index=slic)

        # Add the layer histogram counts
        results.kHist.writeHdf(aFile,'khist',index=slic)

        # Add the elevation histogram counts
        results.DzHist.writeHdf(aFile,'dzhist',index=slic)

        # Add the interface histogram counts
        results.MzHist.writeHdf(aFile,'mzhist',index=slic)

        # Add the relative and additive errors
        for j in range(results.nSystems):
            results.relErr[j].writeHdf(aFile, "relerr" + str(j), index=slic)
            results.addErr[j].writeHdf(aFile, "adderr" + str(j), index=slic)

        # Add the hitmap
        results.Hitmap.writeHdf(aFile,'hitmap',  index=i)

        results.bestD.writeHdf(aFile,'bestd',  index=i)
        # results.currentD.writeHdf(aFile,'currentd',  index=i)

        results.bestModel.writeHdf(aFile,'bestmodel', index=i)

#        if results.verbose:
#            results.posteriorComponents.writeHdf(aFile, 'posteriorcomponents',  index=np.s_[i,:,:])





