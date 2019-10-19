""" @LineResults
Class to handle the HDF5 result files for a line of data.
 """
#from ..base import Error as Err
import os
import numpy as np
import numpy.ma as ma
import h5py
from ..classes.core.myObject import myObject
from ..classes.core import StatArray
from ..classes.statistics.Distribution import Distribution
from ..classes.statistics.Histogram1D import Histogram1D
from ..classes.statistics.Histogram2D import Histogram2D
from ..classes.statistics.Hitmap2D import Hitmap2D
from ..classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from ..classes.mesh.TopoRectilinearMesh2D import TopoRectilinearMesh2D
from ..classes.data.dataset.FdemData import FdemData
from ..classes.data.dataset.TdemData import TdemData
from ..classes.model.Model1D import Model1D
from ..base.HDF import hdfRead
from ..base import customPlots as cP
from ..base import customFunctions as cF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import split
from ..base import fileIO as fIO
from geobipy.src.inversion.Results import Results
import progressbar

try:
    from pyvtk import VtkData, UnstructuredGrid, CellData, Scalars
except:
    pass

class LineResults(myObject):
    """ Class to define results from EMinv1D_MCMC for a line of data """
    def __init__(self, fName=None, sysPath=None, hdfFile=None):
        """ Initialize the lineResults """
        if (fName is None): return

        self._additiveError = None
        self._best = None
        self._currentData = None
        self._bestData = None
        self._bestModel = None
        self._burnedIn = None
        self._confidenceRange = None
        self._doi = None
        self._elevation = None
        self._marginalProbability = None
        self._fiducials = None
        self._hitMap = None
        self._interfaces = None
        self._interfacePosterior = None
        self._lineHitmap = None
        self._mean = None
        self._modeParameter = None
        self._nLayers = None
        self._nPoints = None
        self._nSystems = None
        self._opacity = None
        self.range = None
        self._relativeError = None
        self.sysPath = sysPath
        self._totalError = None
        self._x = None
        self._y = None
        self._z = None
        self._zPosterior = None

        self._mesh = None

        self.fName = fName
        self.line = np.float64(os.path.splitext(split(fName)[1])[0])
        self.hdfFile = None
        if (hdfFile is None): # Open the file for serial read access
            self.open()
        else:
            self.hdfFile = hdfFile


    def open(self):
        """ Check whether the file is open """
        try:
            self.hdfFile.attrs
        except:
            self.hdfFile = h5py.File(self.fName, 'r+')


    def close(self):
        """ Check whether the file is open """
        if (self.hdfFile is None): return
        try:
            self.hdfFile.close()
        except:
            pass # Already closed

            
    def changeUnits(self, units='m'):
        """Change the units of the Coordinates
        
        Parameters
        ----------
        units : str
            The distance units to change to

        """
        if (units == 'km' and self.x.units != 'km'):
            self._x = self.x / 1000.0
            self._y = self.y / 1000.0
            
            self._x.units = 'km'
            self._y.units = 'km'
            

    def crossplotErrors(self, system=0, **kwargs):
        """ Create a crossplot of the relative errors against additive errors for the most probable data point, for each data point along the line """
        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','none')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.0)

        if (self.nSystems > 1):
            r = range(self.nSystems)
            for i in r:
                fc = cP.wellSeparated[i+2]
                cP.plot(x=self.relativeError[:,i], y=self.additiveError[:,i], c=fc,
                    alpha = 0.7,label='System ' + str(i + 1), **kwargs)

            plt.legend()

        else:
            fc = cP.wellSeparated[2]
            cP.plot(x=self.relativeError, y=self.additiveError, c=fc,
                    alpha = 0.7,label='System ' + str(1), **kwargs)

        cP.xlabel(self.relativeError.getNameUnits())
        cP.ylabel(self.additiveError.getNameUnits())


    @property
    def additiveError(self):
        """ Get the Additive error of the best data points """
        if (self._additiveError is None):
            self._additiveError = self.getAttribute('Additive Error')
        return self._additiveError

    
    @property
    def additiveErrorPosteriors(self):
        return self.data.addErr.posterior


    @property
    def bestData(self):
        """ Get the best data """

        if (self._bestData is None):
            attr = self._attrTokey('best data')
            dtype = self.hdfFile[attr[0]].attrs['repr']
            if "FdemDataPoint" in dtype:
                self._bestData = FdemData().fromHdf(self.hdfFile[attr[0]])
            elif "TdemDataPoint" in dtype:
                self._bestData = TdemData().fromHdf(self.hdfFile[attr[0]], sysPath = self.sysPath)
        return self._bestData
        

    @property
    def bestParameters(self):
        """ Get the best model of the parameters """
        if (self._best is None):
            self._best = StatArray.StatArray(self.getAttribute('bestinterp'), dtype=np.float64, name=self.parameterName, units=self.parameterUnits).T
        return self._best


    @property
    def confidenceRange(self):
        """ Get the model parameter opacity using the confidence intervals """
        if (self._confidenceRange is None):
            # Read in the opacity if present
            if "confidence_range" in self.hdfFile.keys():
                self._confidenceRange = StatArray.StatArray().fromHdf(self.hdfFile['confidence_range'])
            else:
                self.computeConfidenceRange()
                
                    
        return self._confidenceRange

    
    def computeConfidenceRange(self, percent=95.0, log=10):

        s = 'percent={}, log={}'.format(percent, log)

        self._confidenceRange = StatArray.StatArray(np.zeros(self.mesh.shape), '{} {}% Confidence Range'.format(cF._logLabel(log), percent), self.parameterUnits)

        loc = 'currentmodel/par/posterior'
        a = np.asarray(self.hdfFile[loc+'/arr/data'])
        try:
            b = np.asarray(self.hdfFile[loc+'/x/data'])
        except:
            b = np.asarray(self.hdfFile[loc+'/x/x/data'])

        try:
            c = np.asarray(self.hdfFile[loc+'/y/data'])
        except:
            c = np.asarray(self.hdfFile[loc+'/y/x/data'])

        h = Hitmap2D(xBinCentres = StatArray.StatArray(b[0, :]), yBinCentres = StatArray.StatArray(c[0, :]))
        h._counts[:, :] = a[0, :, :]
        self._confidenceRange[:, 0] = h.axisConfidenceRange(percent=percent, log=log)
        
        for i in progressbar.progressbar(range(1, self.nPoints)):
            h.x.xBinCentres = b[i, :]
            h._counts[:, :] = a[i, :, :]
            self._confidenceRange[:, i] = h.axisConfidenceRange(percent=percent, log=log)


        if 'confidence_range' in self.hdfFile.keys():
            del self.hdfFile['confidence_range']
                
        self._confidenceRange.toHdf(self.hdfFile, 'confidence_range')
        self.hdfFile['confidence_range'].attrs['options'] = s

    
    @property
    def data(self):
        """ Get the best data """

        if (self._currentData is None):
            attr = self._attrTokey('current data')
            dtype = self.hdfFile[attr[0]].attrs['repr']
            if "FdemDataPoint" in dtype:
                self._currentData = FdemData().fromHdf(self.hdfFile[attr[0]])
            elif "TdemDataPoint" in dtype:
                self._currentData = TdemData().fromHdf(self.hdfFile[attr[0]], sysPath = self.sysPath)
        return self._currentData


    @property
    def doi(self):

        if (self._doi is None):
            # Read in the opacity if present
            if "doi" in self.hdfFile.keys():
                self._doi = StatArray.StatArray().fromHdf(self.hdfFile['doi'])
            else:
                self.computeDOI()
        
        return self._doi


    def computeDOI(self, percent=67.0, window=1):
        """ Get the DOI of the line depending on a percentage variance cutoff for each data point """
        #if (not self.doi is None): return

        assert window > 0, ValueError("window must be >= 1")
        assert 0.0 < percent < 100.0, ValueError("Must have 0.0 < percent < 100.0")
        p = 0.01 * (100.0 - percent)

        doi = StatArray.StatArray(np.zeros(self.nPoints), 'Depth of investigation', self.height.units)

        loc = 'currentmodel/par/posterior'
        a = np.asarray(self.hdfFile[loc+'/arr/data'])
        try:
            b = np.asarray(self.hdfFile[loc+'/x/data'])
        except:
            b = np.asarray(self.hdfFile[loc+'/x/x/data'])

        try:
            c = np.asarray(self.hdfFile[loc+'/y/data'])
        except:
            c = np.asarray(self.hdfFile[loc+'/y/x/data'])

        h = Hitmap2D(xBinCentres = StatArray.StatArray(b[0, :]), yBinCentres = StatArray.StatArray(c[0, :]))
        h._counts[:, :] = a[0, :, :]
        doi[0] = h.getOpacityLevel(percent)
        
        for i in progressbar.progressbar(range(1, self.nPoints)):
            h.x.xBinCentres = b[i, :]
            h._counts[:, :] = a[i, :, :]
            doi[i] = h.getOpacityLevel(percent)

        self._doi = doi
        if window > 1:
            buffer = np.int(0.5 * window)
            tmp = doi.rolling(np.mean, window)
            self._doi[buffer:-buffer] = tmp
            self._doi[:buffer] = tmp[0]
            self._doi[-buffer:] = tmp[-1]
        
        if 'doi' in self.hdfFile.keys():
            del self.hdfFile['doi']
                
        self._doi.toHdf(self.hdfFile, 'doi')

    @property
    def elevation(self):
        """ Get the elevation of the data points """
        if (self._elevation is None):
            self._elevation = StatArray.StatArray(np.asarray(self.getAttribute('elevation')), 'Elevation', 'm')
        return self._elevation


    def extract1DModel(self, values, index=None, fiducial=None):
        """ Obtain the results for the given iD number """

        assert not (index is None and fiducial is None), Exception("Please specify either an integer index or a fiducial.")
        assert index is None or fiducial is None, Exception("Only specify either an integer index or a fiducial.")

        if not fiducial is None:
            assert fiducial in self.fiducials, ValueError("This fiducial {} is not available from this HDF5 file. The min max fids are {} to {}.".format(fiducial, self.fiducials.min(), self.fiducials.max()))
            # Get the point index
            i = self.fiducials.searchsorted(fiducial)
        else:
            i = index
            fiducial = self.fiducials[index]

        depth = self.mesh.z.cellEdges[:-1]
        parameter = values[:, i]

        return Model1D(self.mesh.z.nCells, depth=depth, parameters=parameter, hasHalfspace=False)

    
    @property
    def height(self):
        """Get the height of the observations. """
        if (self._z is None):
            self._z = self.getAttribute('z')
        return self._z


    @property
    def heightPosterior(self):
        if (self._zPosterior is None):
            self._zPosterior = self.getAttribute('height posterior')
            self._zPosterior.bins.name = 'Relative ' + self._zPosterior.bins.name
        return self._zPosterior


    @property
    def hitMap(self):
        """ Get the hitmaps for each data point """
        if (self._hitMap is None):
            self._hitMap = self.getAttribute('Hit Map', index=0)
        return self._hitMap


    @property
    def fiducials(self):
        """ Get the id numbers of the data points in the line results file """
        if (self._fiducials is None):
            try:
                self._fiducials = self.getAttribute('fiducials')
            except:
                self._fiducials = StatArray.StatArray(np.asarray(self.hdfFile.get('ids')), "fiducials")
        return self._fiducials      

    
    def fitMajorPeaks(self, intervals, **kwargs):
        """Fit distributions to the major peaks in each hitmap along the line.

        Parameters
        ----------
        intervals : array_like, optional
            Accumulate the histogram between these invervals before finding peaks

        """
        distributions = []

        hm = self.hitMap.deepcopy()
        counts = np.asarray(self.hdfFile['currentmodel/par/posterior/arr/data'])

        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.nPoints)):

            try:
                dpDistributions = hm.fitMajorPeaks(intervals, **kwargs)
                distributions.append(dpDistributions)
            except:
                pass

            hm._counts = counts[i, :, :]

        return distributions


    def identifyPeaks(self, depths, nBins = 250, width=4, limits=None):
        """Identifies peaks in the parameter posterior for each depth in depths.

        Parameters
        ----------
        depths : array_like
            Depth intervals to identify peaks between.

        Returns
        -------

        """

        from scipy.signal import find_peaks

        assert np.size(depths) > 2, ValueError("Depths must have size > 1.")

        tmp = self.lineHitmap.intervalStatistic(axis=0, intervals = depths, statistic='sum')

        depth = np.zeros(0)
        parameter = np.zeros(0)
                       
        # # Bar = progressbar.ProgressBar()
        # # for i in Bar(range(self.nPoints)):
        for i in range(tmp.y.nCells):
            peaks, _ = find_peaks(tmp.counts[i, :],  width=width)
            values = tmp.x.cellCentres[peaks]
            if not limits is None:
                values = values[(values > limits[0]) & (values < limits[1])]
            parameter = np.hstack([parameter, values])
            depth = np.hstack([depth, np.full(values.size, fill_value=0.5*(depths[i]+depths[i+1]))])

        return np.asarray([depth, parameter]).T


    @property
    def interfaces(self):
        """ Get the layer interfaces from the layer depth histograms """
        if (self._interfaces is None):
            maxCount = self.interfacePosterior.counts.max()
            self._interfaces = StatArray.StatArray(self.interfacePosterior.counts.T / np.float64(maxCount), "interfaces", "")

        return self._interfaces


    @property
    def interfacePosterior(self):
        if (self._interfacePosterior is None):
            self._interfacePosterior = self.getAttribute('layer depth posterior')
        return self._interfacePosterior


    @property
    def lineHitmap(self):
        """ Get the model parameter opacity using the confidence intervals """
        if (self._lineHitmap is None):
            # Read in the opacity if present
            if "line_hitmap" in self.hdfFile.keys():
                self._lineHitmap = Hitmap2D().fromHdf(self.hdfFile['line_hitmap'])
            else:
                self.computeLineHitmap()
                    
        return self._lineHitmap

    
    def computeLineHitmap(self, nBins=250, log=10):

        # First get the min max of the parameter hitmaps
        x0 = np.log10(self.minParameter)
        x1 = np.log10(self.maxParameter)

        counts = self.hdfFile['currentmodel/par/posterior/arr/data']

        xBins = StatArray.StatArray(np.logspace(x0, x1, nBins+1), self.parameterName, units = self.parameterUnits)

        self._lineHitmap = Histogram2D(xBins=xBins, yBins=self.mesh.z.cellEdges)

        parameters = RectilinearMesh1D().fromHdf(self.hdfFile['currentmodel/par/posterior/x'])
                       
        # Bar = progressbar.ProgressBar()
        # for i in Bar(range(self.nPoints)):
        for i in range(self.nPoints):
            p = RectilinearMesh1D(cellEdges=parameters.cellEdges[i, :])

            pj = self._lineHitmap.x.cellIndex(p.cellCentres, clip=True)

            cTmp = counts[i, :, :]

            self._lineHitmap._counts[:, pj] += cTmp

        if 'line_hitmap' in self.hdfFile.keys():
            del self.hdfFile['line_hitmap']
                
        self._lineHitmap.toHdf(self.hdfFile, 'line_hitmap')

    @property
    def maxParameter(self):
        """ Get the mean model of the parameters """
        tmp = np.asarray(self.hdfFile["currentmodel/par/posterior/x/x/data"][:, -1])
        return tmp.max()


    @property
    def meanParameters(self):
        if (self._mean is None):
            self._mean = StatArray.StatArray(self.getAttribute('meaninterp').T, name=self.parameterName, units=self.parameterUnits)

        return self._mean


    @property
    def mesh(self):
        """Get the 2D topo fitting rectilinear mesh. """
        
        if (self._mesh is None):
            if (self._hitMap is None):
                tmp = self.getAttribute('hitmap/y', index=0)
                try:
                    tmp = RectilinearMesh1D(cellCentres=tmp.cellCentres, edgesMin=0.0)
                except:
                    tmp = RectilinearMesh1D(cellCentres=tmp, edgesMin=0.0)
            else:
                tmp = self.hitMap.y
                try:
                    tmp = RectilinearMesh1D(cellCentres=tmp.cellCentres, edgesMin=0.0)
                except:
                    tmp = RectilinearMesh1D(cellCentres=tmp, edgesMin=0.0)

            self._mesh = TopoRectilinearMesh2D(xCentres=self.x, yCentres=self.y, zEdges=tmp.cellEdges, heightCentres=self.elevation)
        return self._mesh


    @property
    def minParameter(self):
        """ Get the mean model of the parameters """
        tmp = np.asarray(self.hdfFile["currentmodel/par/posterior/x/x/data"][:, 0])
        return tmp.min()


    @property
    def modeParameter(self):
        """ Get the model parameter opacity using the confidence intervals """
        if (self._modeParameter is None):
            # Read in the opacity if present
            if "mode_parameter" in self.hdfFile.keys():
                self._modeParameter = StatArray.StatArray().fromHdf(self.hdfFile['mode_parameter'])
            else:
                self.computeModeParameter()
                
                    
        return self._modeParameter


    def computeModeParameter(self):
       
        self._modeParameter = StatArray.StatArray(np.zeros(self.mesh.shape), self.parameterName, self.parameterUnits)

        loc = 'currentmodel/par/posterior'
        a = np.asarray(self.hdfFile[loc+'/arr/data'])
        try:
            b = np.asarray(self.hdfFile[loc+'/x/data'])
        except:
            b = np.asarray(self.hdfFile[loc+'/x/x/data'])

        try:
            c = np.asarray(self.hdfFile[loc+'/y/data'])
        except:
            c = np.asarray(self.hdfFile[loc+'/y/x/data'])

        h = Hitmap2D(xBinCentres = StatArray.StatArray(b[0, :]), yBinCentres = StatArray.StatArray(c[0, :]))
        h._counts[:, :] = a[0, :, :]
        self._modeParameter[:, 0] = h.axisMode()
        
        for i in progressbar.progressbar(range(1, self.nPoints)):
            h.x.xBinCentres = b[i, :]
            h._counts[:, :] = a[i, :, :]
            self._modeParameter[:, i] = h.axisMode()


        if 'mode_parameter' in self.hdfFile.keys():
            del self.hdfFile['mode_parameter']
                
        self._modeParameter.toHdf(self.hdfFile, 'mode_parameter')


    @property
    def nLayers(self):
        """ Get the number of layers in the best model for each data point """
        if (self._nLayers is None):
            self._nLayers = StatArray.StatArray(self.getAttribute('# Layers'), '# of Cells')
        return self._nLayers


    @property
    def nPoints(self):
        return self.fiducials.size
    

    @property
    def nSystems(self):
        """ Get the number of systems """
        if (self._nSystems is None):
            self._nSystems = self.getAttribute('# of systems')
        return self._nSystems


    @property
    def opacity(self):
        """ Get the model parameter opacity using the confidence intervals """
        if (self._opacity is None):
           self.computeOpacity()
                    
        return self._opacity

    
    def computeOpacity(self, percent=95.0, log=10, multiplier=0.5, useDOI=True):

        tmp = self.confidenceRange

        # if depthScaling:
        #     tmp = self.confidenceRange * np.repeat(self.mesh.z.cellCentres[:, np.newaxis], self.nPoints, axis=1)

        rnge = multiplier * (tmp.max() - tmp.min())

        self._opacity = StatArray.StatArray(tmp, "Opacity", "")
        
        self._opacity[self._opacity > rnge] = rnge

        self._opacity = 1.0 - (self._opacity / rnge)

        if useDOI:
            indices = self.mesh.z.cellIndex(self.doi)

            for i in range(self.nPoints):
                self._opacity[:indices[i], i] = 1.0


    @property
    def parameterName(self):

        return self.hitMap.x.cellCentres.name


    @property
    def parameterUnits(self):

        return self.hitMap.x.cellCentres.units    

    
    def percentageParameter(self, value, depth=None, depth2=None):

        # Get the depth grid
        if (not depth is None):
            assert depth <= self.mesh.z.cellEdges[-1], 'Depth is greater than max depth '+str(self.mesh.z.cellEdges[-1])
            j = self.mesh.z.cellIndex(depth)
            k = j+1
            if (not depth2 is None):
                assert depth2 <= self.mesh.z.cellEdges[-1], 'Depth2 is greater than max depth '+str(self.mesh.z.cellEdges[-1])
                assert depth <= depth2, 'Depth2 must be >= depth'
                k = self.mesh.z.cellIndex(depth2)

        percentage = StatArray.StatArray(np.empty(self.nPoints), name="Probability of {} > {:0.2f}".format(parameterName, value), units = parameterName.units)

        if depth:
            counts = self.hdfFile['currentmodel/par/posterior/arr/data'][:, j:k, :]
            # return StatArray.StatArray(np.sum(counts[:, :, pj:]) / np.sum(counts) * 100.0, name="Probability of {} > {:0.2f}".format(self.meanParameters.name, value), units = self.meanParameters.units)
        else:
            counts = self.hdfFile['currentmodel/par/posterior/arr/data']

        parameters = RectilinearMesh1D().fromHdf(self.hdfFile['currentmodel/par/posterior/x'])
               
        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.nPoints)):
            p = RectilinearMesh1D(cellEdges=parameters.cellEdges[i, :])
            pj = p.cellIndex(value)

            cTmp = counts[i, :, :]

            percentage[i] = np.sum(cTmp[:, pj:]) / cTmp.sum() * 100.0

        return percentage

        
    @property
    def relativeError(self):
        """ Get the Relative error of the best data points """
        if (self._relativeError is None):
            self._relativeError = self.getAttribute('Relative Error')
        return self._relativeError


    @property    
    def relativeErrorPosteriors(self):
        """ Get the Relative error of the best data points """
        return self.data.relErr.posterior


    def getResults(self, index=None, fiducial=None, reciprocateParameter=False):
        """ Obtain the results for the given iD number """

        assert not (index is None and fiducial is None), Exception("Please specify either an integer index or a fiducial.")
        assert index is None or fiducial is None, Exception("Only specify either an integer index or a fiducial.")

        if not fiducial is None:
            assert fiducial in self.fiducials, ValueError("This fiducial {} is not available from this HDF5 file. The min max fids are {} to {}.".format(fiducial, self.fiducials.min(), self.fiducials.max()))
            # Get the point index
            i = self._fiducials.searchsorted(fiducial)
        else:
            i = index
            fiducial = self._fiducials[index]

        hdfFile = self.hdfFile

        R = Results(reciprocateParameter=reciprocateParameter).fromHdf(hdfFile, index=i, systemFilePath=self.sysPath)

        return R
        


    @property
    def totalError(self):
        """ Get the total error of the best data points """
        if (self._totalError is None):
            self._totalError = self.getAttribute('Total Error')
        return self._totalError


    @property
    def x(self):
        """ Get the X co-ordinates (Easting) """
        if (self._x is None):
            self._x = self.getAttribute('x')
            if self._x.name in [None, '']:
                self._x.name = 'Easting'
            if self._x.units in [None, '']:
                self._x.units = 'm'
        return self._x


    @property
    def y(self):
        """ Get the Y co-ordinates (Easting) """
        if (self._y is None):
            self._y = self.getAttribute('y')

            if self._y.name in [None, '']:
                self._y.name = 'Northing'
            if self._y.units in [None, '']:
                self._y.units = 'm'
        return self._y


    def pcolorDataResidual(self, abs=False, **kwargs):
        """ Plot a channel of data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=False)

        values = self.bestData.deltaD.T

        if abs:
            values = values.abs()
        
        cP.pcolor(values, x=xtmp, y=StatArray.StatArray(np.arange(self.bestData.predictedData.shape[1]), name='Channel'), **kwargs)


    def pcolorObservedData(self, **kwargs):
        """ Plot a channel of data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=False)
        
        cP.pcolor(self.bestData.data.T, x=xtmp, y=StatArray.StatArray(np.arange(self.bestData.predictedData.shape[1]), name='Channel'), **kwargs)


    def pcolorPredictedData(self, **kwargs):
        """ Plot a channel of data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=False)
        
        cP.pcolor(self.bestData.predictedData.T, x=xtmp, y=StatArray.StatArray(np.arange(self.bestData.predictedData.shape[1]), name='Channel'), **kwargs)

    
    def plotPredictedData(self, channel=None, **kwargs):
        """ Plot a channel of the best predicted data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=True)

        if channel is None:
            channel = np.s_[:]

        cP.plot(xtmp, self.bestData.predictedData[:, channel], **kwargs)


    def plotDataElevation(self, **kwargs):
        """ Adds the data elevations to a plot """

        xAxis = kwargs.pop('xAxis', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color','k')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.5)

        xtmp = self.mesh.getXAxis(xAxis, centres=False)

        cP.plot(xtmp, self.height.edges() + self.elevation.edges(), **kwargs)

        if (labels):
            cP.xlabel(xtmp.getNameUnits())
            cP.ylabel('Elevation (m)')


    def plotDataResidual(self, channel=None, abs=False, **kwargs):
        """ Plot a channel of the observed data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=True)

        if channel is None:
            channel = np.s_[:]

        values = self.bestData.deltaD[:, channel]

        if abs:
            values = values.abs()

        cP.plot(xtmp, values, **kwargs)


    def plotDoi(self, percent=67.0, window=1, **kwargs):

        xAxis = kwargs.pop('xAxis', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color','k')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.5)

        if window == 1:
            xtmp = self.mesh.getXAxis(xAxis, centres=True)

            cP.plot(xtmp, self.elevation - self.doi, **kwargs)

        else:
            w2 = np.int(0.5 * window)
            w22 = -w2 if window % 2 == 0 else -w2-1
            
            xtmp = self.mesh.getXAxis(xAxis, centres=True)[w2-1:w22]

            cP.plot(xtmp, self.elevation[w2-1:w22] - self.doi, **kwargs)

        #if (labels):
        #    cP.xlabel(self.xPlot.getNameUnits())
        #    cP.ylabel('Elevation (m)')


    def plotElevation(self, **kwargs):

        xAxis = kwargs.pop('xAxis', 'x')
        labels = kwargs.pop('labels', True)
        kwargs['color'] = kwargs.pop('color','k')
        kwargs['linewidth'] = kwargs.pop('linewidth',0.5)

        self.mesh.plotHeight(xAxis=xAxis, **kwargs)

        # if (labels):
        #     cP.xlabel(xtmp.getNameUnits())
        #     cP.ylabel('Elevation (m)')


    def plotHeightPosteriors(self, **kwargs):
        """ Plot the horizontally stacked elevation histograms for each data point along the line """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis)

        post = self.heightPosterior


        c = post.counts.T
        c = np.divide(c, np.max(c,0), casting='unsafe')

        ax1 = plt.subplot(111)        
        c.pcolor(xtmp, y=post.bins, noColorbar=True, **kwargs)
        
        ax2 = ax1.twinx()

        c = cP.wellSeparated[0]
        post.relativeTo.plot(xtmp.internalEdges(), c=c, ax=ax2)
        ax2.set_ylabel(ax2.get_ylabel(), color=c)
        ax2.tick_params(axis='y', labelcolor=c)

        cP.title('Data height posterior distributions')


    def plotHighlightedObservationLocations(self, iDs, **kwargs):

        labels = kwargs.pop('labels', True)
        kwargs['marker'] = kwargs.pop('marker','*') # Downward pointing arrow
        kwargs['color'] = kwargs.pop('color',cP.wellSeparated[1])
        kwargs['linestyle'] = kwargs.pop('linestyle','none')
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth','0.1')
        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis)

        i = self.fiducials.searchsorted(iDs)

        tmp = self.height.reshape(self.height.size) + self.elevation

        plt.plot(xtmp[i], tmp[i], **kwargs)

        if (labels):
            cP.xlabel(xtmp.getNameUnits())
            cP.ylabel('Elevation (m)')


    def plotKlayers(self, **kwargs):
        """ Plot the number of layers in the best model for each data point """
        xAxis = kwargs.pop('xAxis', 'x')
        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth', 1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','none')

        xtmp = self.mesh.getXAxis(xAxis)
        self.nLayers.plot(xtmp, **kwargs)
        # cP.ylabel(self.nLayers.getNameUnits())
        cP.title('# of Layers in Best Model')


    def plotKlayersPosteriors(self, **kwargs):
        """ Plot the horizontally stacked elevation histograms for each data point along the line """

        post = self.getAttribute('layer posterior')

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis)

        c = post.counts.T
        c = np.divide(c, np.max(c,0), casting='unsafe')
        ax, pm, cb = c.pcolor(xtmp, y=post.binCentres[0, :], **kwargs)
        cP.title('# of Layers posterior distributions')


    def plotAdditiveError(self, **kwargs):
        """ Plot the relative errors of the data """
        xAxis = kwargs.pop('xAxis', 'x')
        m = kwargs.pop('marker','o')
        ms = kwargs.pop('markersize',5)
        mfc = kwargs.pop('markerfacecolor',None)
        mec = kwargs.pop('markeredgecolor','k')
        mew = kwargs.pop('markeredgewidth',1.0)
        ls = kwargs.pop('linestyle','-')
        lw = kwargs.pop('linewidth',1.0)

        xtmp = self.mesh.getXAxis(xAxis, centres=True)

        if (self.nSystems > 1):
            r = range(self.nSystems)
            for i in r:
                fc = cP.wellSeparated[i+2]
                cP.plot(xtmp, y=self.additiveError[:,i],
                    marker=m,markersize=ms,markerfacecolor=mfc,markeredgecolor=mec,markeredgewidth=mew,
                    linestyle=ls,linewidth=lw,c=fc,
                    alpha = 0.7,label='System ' + str(i + 1), **kwargs)
            plt.legend()
        else:
            fc = cP.wellSeparated[2]
            cP.plot(xtmp, y=self.additiveError,
                    marker=m,markersize=ms,markerfacecolor=mfc,markeredgecolor=mec,markeredgewidth=mew,
                    linestyle=ls,linewidth=lw,c=fc,
                    alpha = 0.7,label='System ' + str(1), **kwargs)

        # cP.xlabel(xtmp.getNameUnits())
        # cP.ylabel(self.additiveError.getNameUnits())
        


    def plotAdditiveErrorPosteriors(self, system=0, **kwargs):
        """ Plot the distributions of additive errors as an image for all data points in the line """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis)

        if self.nSystems > 1:
            post = self.additiveErrorPosteriors[system]
        else:
            post = self.additiveErrorPosteriors

        c = post.counts.T
        c = np.divide(c, np.max(c, 0), casting='unsafe')

        if post.isRelative:
            if np.all(post.relativeTo == post.relativeTo[0]):
                y = post.binCentres + post.relativeTo[0]
            else:
                stuff = 2
        else:
            y = post.binCentres[0, :]
        c.pcolor(xtmp, y=y, **kwargs)
        cP.title('Additive error posterior distributions for system {}'.format(system))


    def plotError2DJointProbabilityDistribution(self, datapoint, system=0, **kwargs):
        """ For a given datapoint, obtains the posterior distributions of relative and additive error and creates the 2D joint probability distribution """

        # Read in the histogram of relative error for the data point
        rel = self.getAttribute('Relative error histogram', index=datapoint)
        # Read in the histogram of additive error for the data point
        add = self.getAttribute('Additive error histogram', index=datapoint)

        joint = Histogram2D()
        joint.create2DjointProbabilityDistribution(rel[system],add[system])

        joint.pcolor(**kwargs)


    def plotInterfaces(self, cut=0.0, **kwargs):
        """ Plot a cross section of the layer depth histograms. Truncation is optional. """

        kwargs['noColorbar'] = kwargs.pop('noColorbar', True)

        self.plotXsection(values=self.interfaces, **kwargs)
        


    def plotObservedData(self, channel=None, **kwargs):
        """ Plot a channel of the observed data as points """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis, centres=True)

        if channel is None:
            channel = np.s_[:]

        cP.plot(xtmp, self.bestData.data[:, channel], **kwargs)


    def plotOpacity(self, **kwargs):
        """ Plot the opacity """
        self.plotXsection(values = self.opacity, **kwargs)


    def plotRelativeErrorPosteriors(self, system=0, **kwargs):
        """ Plot the distributions of relative errors as an image for all data points in the line """

        xAxis = kwargs.pop('xAxis', 'x')

        xtmp = self.mesh.getXAxis(xAxis)

        if self.nSystems > 1:
            post = self.relativeErrorPosteriors[system]
        else:
            post = self.relativeErrorPosteriors

        c = post.counts.T
        c = np.divide(c, np.max(c,0), casting='unsafe')

        if post.isRelative:
            if np.all(post.relativeTo == post.relativeTo[0]):
                y = post.binCentres + post.relativeTo[0]
            else:
                stuff = 2
        else:
            y = post.binCentres[0, :]
        c.pcolor(xtmp, y=y, **kwargs)

        cP.title('Relative error posterior distributions for system {}'.format(system))


    def plotRelativeError(self, **kwargs):
        """ Plot the relative errors of the data """

        xAxis = kwargs.pop('xAxis', 'x')
        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','-')
        kwargs['linewidth'] = kwargs.pop('linewidth',1.0)

        xtmp = self.mesh.getXAxis(xAxis)

        if (self.nSystems > 1):
            r = range(self.nSystems)
            for i in r:
                kwargs['c'] = cP.wellSeparated[i+2]
                self.relativeError[:, i].plot(xtmp,
                    alpha = 0.7, label='System {}'.format(i + 1), **kwargs)
            plt.legend()
        else:
            kwargs['c'] = cP.wellSeparated[2]
            self.relativeError.plot(xtmp,
                    alpha = 0.7, label='System {}'.format(1), **kwargs)


    def plotTotalError(self, channel, **kwargs):
        """ Plot the relative errors of the data """


        xAxis = kwargs.pop('xAxis', 'x')

        kwargs['marker'] = kwargs.pop('marker','o')
        kwargs['markersize'] = kwargs.pop('markersize',5)
        kwargs['markerfacecolor'] = kwargs.pop('markerfacecolor',None)
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor','k')
        kwargs['markeredgewidth'] = kwargs.pop('markeredgewidth',1.0)
        kwargs['linestyle'] = kwargs.pop('linestyle','-')
        kwargs['linewidth'] = kwargs.pop('linewidth',1.0)

        xtmp = self.mesh.getXAxis(xAxis)

#        if (self.nSystems > 1):
#            r = range(self.nSystems)
#            for i in r:
#                fc = cP.wellSeparated[i+2]
#                cP.plot(x=self.xPlot, y=self.additiveError[:,i],
#                    marker=m,markersize=ms,markerfacecolor=mfc,markeredgecolor=mec,markeredgewidth=mew,
#                    linestyle=ls,linewidth=lw,c=fc,
#                    alpha = 0.7,label='System ' + str(i + 1), **kwargs)
#        else:
        fc = cP.wellSeparated[2]
        self.totalError[:,channel].plot(xtmp,
                alpha = 0.7, label='Channel ' + str(channel), **kwargs)

#        cP.xlabel(self.xPlot.getNameUnits())
#        cP.ylabel(self.additiveError.getNameUnits())
#        plt.legend()


    def plotTotalErrorDistributions(self, channel=0, nBins=100, **kwargs):
        """ Plot the distributions of relative errors as an image for all data points in the line """
        self.setAlonglineAxis(self.plotAgainst)

        H = Histogram1D(values=np.log10(self.totalError[:,channel]),nBins=nBins)

        H.plot(**kwargs)

#        if self.nSystems > 1:
#            c = tmp[system].counts.T
#            c = np.divide(c, np.max(c,0), casting='unsafe')
#            c.pcolor(x=self.xPlot, y=tmp[system].bins, **kwargs)
#            cP.title('Relative error posterior distributions for system '+str(system))
#        else:
#            c = tmp.counts.T
#            c = np.divide(c, np.max(c,0), casting='unsafe')
#            c.pcolor(x=self.xPlot, y=tmp.bins, **kwargs)
#            cP.title('Relative error posterior distributions')

    def histogram(self, nBins, depth1 = None, depth2 = None, reciprocateParameter = False, bestModel = False, **kwargs):
        """ Compute a histogram of the model, optionally show the histogram for given depth ranges instead """

        if (depth1 is None):
            depth1 = np.maximum(self.mesh.z.cellEdges[0], 0.0)
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
            model = self.bestParameters
            title = 'Best model values between {:.3f} m and {:.3f} m depth'.format(depth1, depth2)
        else:
            model = self.meanParameters
            title = 'Mean model values between {:.3f} m and {:.3f} m depth'.format(depth1, depth2)

        vals = model[:, cell1:cell2+1].deepcopy()

        log = kwargs.pop('log',False)

        if (reciprocateParameter):
            i = np.where(vals > 0.0)[0]
            vals[i] = 1.0 / vals[i]
            name = 'Resistivity'
            units = '$\Omega m$'
        else:
            name = 'Conductivity'
            units = '$Sm^{-1}$'

        if (log):
            vals, logLabel = cF._log(vals,log)
            name = logLabel + name
        binEdges = StatArray.StatArray(np.linspace(np.nanmin(vals), np.nanmax(vals), nBins+1), name, units)

        h = Histogram1D(bins = binEdges)
        h.update(vals)
        h.plot(**kwargs)
        cP.title(title)

    
    def parameterHistogram(self, nBins, depth = None, depth2 = None, log=None):
        """ Compute a histogram of all the parameter values, optionally show the histogram for given depth ranges instead """

        # Get the depth grid
        if (not depth is None):
            assert depth <= self.mesh.z.cellEdges[-1], 'Depth is greater than max depth '+str(self.mesh.z.cellEdges[-1])
            j = self.mesh.z.cellIndex(depth)
            k = j+1
            if (not depth2 is None):
                assert depth2 <= self.mesh.z.cellEdges[-1], 'Depth2 is greater than max depth '+str(self.mesh.z.cellEdges[-1])
                assert depth <= depth2, 'Depth2 must be >= depth'
                k = self.mesh.z.cellIndex(depth2)

        # First get the min max of the parameter hitmaps
        x0 = np.log10(self.minParameter)
        x1 = np.log10(self.maxParameter)

        if depth:
            counts = self.hdfFile['currentmodel/par/posterior/arr/data'][:, j:k, :]
            # return StatArray.StatArray(np.sum(counts[:, :, pj:]) / np.sum(counts) * 100.0, name="Probability of {} > {:0.2f}".format(self.meanParameters.name, value), units = self.meanParameters.units)
        else:
            counts = self.hdfFile['currentmodel/par/posterior/arr/data']

        parameters = RectilinearMesh1D().fromHdf(self.hdfFile['currentmodel/par/posterior/x'])

        bins = StatArray.StatArray(np.logspace(x0, x1, nBins), self.parameterName, units = self.parameterUnits)

        out = Histogram1D(bins=bins, log=log)

                       
        # Bar = progressbar.ProgressBar()
        # for i in Bar(range(self.nPoints)):
        for i in range(self.nPoints):
            p = RectilinearMesh1D(cellEdges=parameters.cellEdges[i, :])

            pj = out.cellIndex(p.cellCentres, clip=True)

            cTmp = counts[i, :, :]

            out.counts[pj] += np.sum(cTmp, axis=0)

        return out


    def plotBestModel(self, reciprocateParameter = False, useVariance=True, **kwargs):

        values = self.bestParameters
        if (reciprocateParameter):
            values = 1.0 / values
            values.name = 'Resistivity'
            values.units = '$\Omega m$'

        return self.plotXsection(values = values, useVariance = useVariance, **kwargs)


    def plotMeanModel(self, reciprocateParameter = False, useVariance=True, **kwargs):

        values = self.meanParameters
        if (reciprocateParameter):
            values = 1.0 / values
            values.name = 'Resistivity'
            values.units = '$\Omega m$'

        return self.plotXsection(values = values, useVariance = useVariance, **kwargs)


    def plotModeModel(self, reciprocateParameter = False, useVariance=True, **kwargs):

        values = self.modeParameter
        if (reciprocateParameter):
            values = 1.0 / values
            values.name = 'Resistivity'
            values.units = '$\Omega m$'

        return self.plotXsection(values = values, useVariance = useVariance, **kwargs)


    def plotXsection(self, values, useVariance=False, **kwargs):
        """ Plot a cross-section of the parameters """

        if useVariance:
            kwargs['alpha'] = self.opacity
    
        ax, pm, cb = self.mesh.pcolor(values = values, **kwargs)
        return ax, pm, cb


    # def plotFacies(self, mean, var, volFrac, percent=67.0, ylim=None):
    #     """ Plot a cross-section of the parameters """

    #     assert False, ValueError('Double check this')

    #     self.setAlonglineAxis(self.plotAgainst)
    #     zGrd = self.zGrid

    #     a = np.zeros([zGrd.size, self.nPoints + 1],order = 'F')  # Transparency amounts
    #     a[:, 1:] = self.opacity

    #     self._getX_pmesh(zGrd.size)
    #     self._getZ_pmesh(zGrd)

    #     c = np.zeros(self._xMesh.shape, order = 'F')  # Colour value of each cell
    #     c[:-1, 0] = np.nan
    #     c[:-1, -1] = np.nan
    #     c[-1, :] = np.nan

    #     self.assignFacies(mean, var, volFrac)
    #     c[:-1, 1:-1] = self.facies

    #     # Mask the Nans in the colorMap
    #     cm = ma.masked_invalid(c)

    #     # Get the "depth of investigation"
    #     self.getDOI(percent)

    #     ax = plt.gca()
    #     cP.pretty(ax)
    #     p = ax.pcolormesh(self._xMesh, self._zMesh, cm)
    #     cb = plt.colorbar(p, ax=ax)
    #     plt.plot(self.xPlot, self.elevation, color='k')
    #     if (self.z is None):
    #         self.z = np.asarray(self.getAttribute('z'))
    #     plt.plot(self.xPlot, self.z.reshape(self.z.size) + self.elevation, color='k')

    #     plt.plot(self.xPlot, self.elevation - self.doi, color='k', linestyle='-', alpha=0.7, linewidth=1)

    #     plt.savefig('myfig.png')
    #     for i, j in zip(p.get_facecolors(), a.flatten()):
    #         i[3] = j  # Set the alpha value of the RGBA tuple using m2
    #     fIO.deleteFile('myfig.png')

    #     if (not ylim is None):
    #         plt.ylim(ylim)

    #     cP.xlabel(self.xPlot.getNameUnits())
    #     cP.ylabel('Elevation (m)')
    #     cP.clabel(cb, 'Facies')


    def marginalProbability(self, i=None):

        if self._marginalProbability is None:

            assert 'marginal_probability' in self.hdfFile.keys() or 'facies_probability' in self.hdfFile.keys(), Exception("Marginal probabilities need computing, use LineResults.computeMarginalProbability()")

            if 'marginal_probability' in self.hdfFile.keys():
                self._marginalProbability = StatArray.StatArray().fromHdf(self.hdfFile['marginal_probability'])
            elif 'facies_probability' in self.hdfFile.keys():
                self._marginalProbability = StatArray.StatArray().fromHdf(self.hdfFile['facies_probability'])

        if i is None:
            return self._marginalProbability

        else:
            nClasses = self._marginalProbability.shape[0]
            return StatArray.StatArray(self._marginalProbability[i, :, :], name = "$\frac{P_{" + str(i+1) + "}}{\sum_{i=1}^{" + str(nClasses) + "}P_{i}}$")

    
    def computeMarginalProbability(self, fractions, distributions, **kwargs):

        print("Computing marginal probabilities. This can take a while. \nOnce you have done this, and you no longer need to change the input parameters, simply use LineResults.marginalProbability to access.")

        nFacies = np.size(distributions)

        self._marginalProbability = StatArray.StatArray(np.empty([nFacies, self.hitMap.y.nCells, self.nPoints]), name='Marginal probability')

        counts = self.hdfFile['currentmodel/par/posterior/arr/data']
        parameters = RectilinearMesh1D().fromHdf(self.hdfFile['currentmodel/par/posterior/x'])

        hm = self.getAttribute('hit map', index=0)

        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.nPoints)):
            hm._counts = counts[i, :, :]
            hm._x = RectilinearMesh1D(cellEdges=parameters.cellEdges[i, :])
            self._marginalProbability[:, :, i] = hm.marginalProbability(fractions, distributions, axis=0, **kwargs)
        
        if 'marginal_probability' in self.hdfFile.keys():
            del self.hdfFile['marginal_probability']
        if 'facies_probability' in self.hdfFile.keys():
            del self.hdfFile['facies_probability']
        
        self._marginalProbability.toHdf(self.hdfFile, 'marginal_probability')

        return self.marginalProbability


    @property
    def highestMarginal(self):

        out = StatArray.StatArray(self.mesh.shape, "Most probable class")
        mp = self.marginalProbability()
        for i in range(self.nPoints):
            out[:, i] = np.argmax(mp[:, :, i], axis=0)
    
        return out


    @property
    def probability_of_highest_marginal(self):

        out = StatArray.StatArray(self.mesh.shape, "Probability")
        
        hm = self.highestMarginal
        classes = np.unique(hm)

        mp = self.marginalProbability()

        for i, c in enumerate(classes):
            iWhere = np.where(hm == c)
            out[iWhere[0], iWhere[1]] = mp[i, iWhere[0], iWhere[1]]
    
        return out


    def plotDataPointResults(self, fiducial):
        """ Plot the geobipy results for the given data point """
        R = self.getResults(fiducial=fiducial)
        R.initFigure(forcePlot=True)
        R.plot(forcePlot=True)

    
    def plotSummary(self, data, fiducial, **kwargs):

        R = self.getResults(fiducial=fiducial)

        cWidth = 3
        
        nCols = 15 + (3 * R.nSystems) + 1

        gs = gridspec.GridSpec(18, nCols)
        gs.update(wspace=20.0, hspace=20.0)
        ax = [None]*(7+(2*R.nSystems))

        ax[0] = plt.subplot(gs[:3, :nCols - 10 - (R.nSystems * 3)]) # Data misfit vs iteration
        R._plotMisfitVsIteration(markersize=1, marker='.',)


        ax[1] = plt.subplot(gs[3:6, :nCols - 10 - (R.nSystems * 3)]) # Histogram of # of layers
        R._plotNumberOfLayersPosterior()


        ax[2] = plt.subplot(gs[6:12, :nCols - 13]) # Site Map
        data.scatter2D(c='k', s=1)
        line = data.getLine(line=self.line)
        line.scatter2D(c='cyan')
        cP.plot(R.bestDataPoint.x, R.bestDataPoint.y, color='r', marker='o')


        ax[3] = plt.subplot(gs[6:12, nCols - 13:nCols - 10]) # Data Point
        R._plotObservedPredictedData()
        plt.title('')


        ax[4] = plt.subplot(gs[:12, nCols - 10: nCols - 4]) # Hitmap
        R._plotHitmapPosterior()


        ax[5] = plt.subplot(gs[:12, nCols - 4: nCols-1]) # Interface histogram
        R._plotLayerDepthPosterior()


        ax[6] = plt.subplot(gs[12:, :]) # cross section
        self.plotXsection(**kwargs)


        for i in range(R.nSystems):

            j0 = nCols - 10 - (R.nSystems* 3) + (i * 3)
            j1 = j0 + 3

            ax[7+(2*i)] = plt.subplot(gs[:3, j0:j1])
            R._plotRelativeErrorPosterior(system=i)
            cP.title('System ' + str(i + 1))

            # Update the histogram of additive data errors
            ax[7+(2*i)-1] = plt.subplot(gs[3:6, j0:j1])
            # ax= plt.subplot(self.gs[3:6, 2 * self.nSystemstems + j])
            R._plotAdditiveErrorPosterior(system=i)



    def toVtk(self, fileName, format='binary'):
        """Write the parameter cross-section to an unstructured grid vtk file 
        
        Parameters
        ----------
        fileName : str
            Filename to save to.
        format : str, optional
            "ascii" or "binary" format. Ascii is readable, binary is not but results in smaller files.

        """
        a = self.bestParameters.T
        b = self.meanParameters.T
        c = self.interfaces().T

        d = StatArray.StatArray(1.0 / a, "Best Conductivity", "$\fraq{S}{m}$")
        e = StatArray.StatArray(1.0 / b, "Mean Conductivity", "$\fraq{S}{m}$")

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


    def _attrTokey(self, attributes):
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
            elif (low == 'height posterior'):
                res.append('currentdatapoint/z/posterior')
            elif (low == 'fiducials'):
                res.append('fiducials')
            elif (low == 'layer posterior'):
                res.append('currentmodel/nCells/posterior')
            elif (low == 'layer depth posterior'):
                res.append('currentmodel/depth/posterior')
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
            elif (low == 'current data'):
                res.append('currentdatapoint')
            elif (low == 'hit map'):
                res.append('currentmodel/par/posterior')
            elif (low == 'hitmap/y'):
                res.append('currentmodel/par/posterior/y')
            elif (low == 'doi'):
                res.append('doi')
            elif (low == 'data misfit'):
                res.append('phids')
            elif (low == 'relative error posterior'):
                if (nSys is None): nSys = hdfRead.readKeyFromFile(self.hdfFile, self.fName, '/','nsystems')
                for i in range(nSys):
                    res.append('currentdatapoint/relErr/posterior' +str(i))
            elif (low == 'additive error posterior'):
                if (nSys is None): nSys = hdfRead.readKeyFromFile(self.hdfFile, self.fName, '/','nsystems')
                for i in range(nSys):
                    res.append('currentdatapoint/addErr/posterior' +str(i))
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
              "layer posterior \n" +
              "height posterior \n" +
              "layer depth posterior \n" +
              "best data \n" +
              "fiducials" +
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
              "relative error posterior\n" +
              "additive error posterior\n" +
              "inversion time\n" +
              "saving time\n"+
              "====================================================\n")


    def createHdf(self, hdfFile, fiducials, results):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """

        self.hdfFile = hdfFile

        nPoints = fiducials.size
        self._fiducials = StatArray.StatArray(np.sort(fiducials), "fiducials")

        # Initialize and write the attributes that won't change
        # hdfFile.create_dataset('ids',data=self.fiducials)
        self.fiducials.createHdf(hdfFile, 'fiducials')
        self.fiducials.writeHdf(hdfFile, 'fiducials')
        hdfFile.create_dataset('iplot', data=results.iPlot)
        hdfFile.create_dataset('plotme', data=results.plotMe)
        hdfFile.create_dataset('reciprocateParameter', data=results.reciprocateParameter)

        if not results.limits is None:
            hdfFile.create_dataset('limits', data=results.limits)
        hdfFile.create_dataset('nmc', data=results.nMC)
        hdfFile.create_dataset('nsystems', data=results.nSystems)
        results.ratex.toHdf(hdfFile,'ratex')
#        hdfFile.create_dataset('ratex', [results.ratex.size], dtype=results.ratex.dtype)
#        hdfFile['ratex'][:] = results.ratex


        # Initialize the attributes that will be written later
        hdfFile.create_dataset('i', shape=[nPoints], dtype=results.i.dtype, fillvalue=np.nan)
        hdfFile.create_dataset('iburn', shape=[nPoints], dtype=results.iBurn.dtype, fillvalue=np.nan)
        hdfFile.create_dataset('burnedin', shape=[nPoints], dtype=type(results.burnedIn))
        hdfFile.create_dataset('doi',  shape=[nPoints], dtype=float, fillvalue=np.nan)
        hdfFile.create_dataset('multiplier',  shape=[nPoints], dtype=results.multiplier.dtype, fillvalue=np.nan)
        hdfFile.create_dataset('invtime',  shape=[nPoints], dtype=float, fillvalue=np.nan)
        hdfFile.create_dataset('savetime',  shape=[nPoints], dtype=float, fillvalue=np.nan)

        results.meanInterp.createHdf(hdfFile,'meaninterp', nRepeats=nPoints, fillvalue=np.nan)
        results.bestInterp.createHdf(hdfFile,'bestinterp', nRepeats=nPoints, fillvalue=np.nan)
        results.opacityInterp.createHdf(hdfFile,'opacityinterp',nRepeats=nPoints, fillvalue=np.nan)
#        hdfFile.create_dataset('opacityinterp', [nPoints,nz], dtype=np.float64)
        
        results.rate.createHdf(hdfFile,'rate',nRepeats=nPoints, fillvalue=np.nan)
#        hdfFile.create_dataset('rate', [nPoints,results.rate.size], dtype=results.rate.dtype)
        results.PhiDs.createHdf(hdfFile,'phids',nRepeats=nPoints, fillvalue=np.nan)
        #hdfFile.create_dataset('phids', [nPoints,results.PhiDs.size], dtype=results.PhiDs.dtype)

        # Create the layer histogram
        # results.kHist.createHdf(hdfFile,'khist',nRepeats=nPoints, fillvalue=np.nan)

        # Create the Elevation histogram
        # results.DzHist.createHdf(hdfFile,'dzhist',nRepeats=nPoints, fillvalue=np.nan)

        # Create the Interface histogram
        # results.MzHist.createHdf(hdfFile,'mzhist',nRepeats=nPoints, fillvalue=np.nan)

        # Add the relative and additive errors
        # for i in range(results.nSystems):
            # results.relErr[i].createHdf(hdfFile,"relerr"+str(i),nRepeats=nPoints, fillvalue=np.nan)
            # results.addErr[i].createHdf(hdfFile,"adderr"+str(i),nRepeats=nPoints, fillvalue=np.nan)

        # Add the Hitmap
        # results.Hitmap.createHdf(hdfFile,'hitmap', nRepeats=nPoints, fillvalue=np.nan)

        results.currentDataPoint.createHdf(hdfFile,'currentdatapoint', nRepeats=nPoints, fillvalue=np.nan)
        results.bestDataPoint.createHdf(hdfFile,'bestd', withPosterior=False, nRepeats=nPoints, fillvalue=np.nan)

        # Since the 1D models change size adaptively during the inversion, we need to pad the HDF creation to the maximum allowable number of layers.
        
        tmp = results.currentModel.pad(results.currentModel.maxLayers)

        tmp.createHdf(hdfFile, 'currentmodel', nRepeats=nPoints, fillvalue=np.nan)

        tmp = results.bestModel.pad(results.bestModel.maxLayers)
        tmp.createHdf(hdfFile, 'bestmodel', withPosterior=False, nRepeats=nPoints, fillvalue=np.nan)

        if results.verbose:
            results.posteriorComponents.createHdf(hdfFile,'posteriorcomponents',nRepeats=nPoints, fillvalue=np.nan)

        # Add the best data components
#        hdfFile.create_dataset('bestdata.z', [nPoints], dtype=results.bestD.z.dtype)
#        hdfFile.create_dataset('bestdata.p', [nPoints,*results.bestD.p.shape], dtype=results.bestD.p.dtype)
#        hdfFile.create_dataset('bestdata.s', [nPoints,*results.bestD.s.shape], dtype=results.bestD.s.dtype)

        # Add the best model components
#        hdfFile.create_dataset('bestmodel.ncells', [nPoints], dtype=results.bestModel.nCells.dtype)
#        hdfFile.create_dataset('bestmodel.top', [nPoints], dtype=results.bestModel.top.dtype)
#        hdfFile.create_dataset('bestmodel.par', [nPoints,*results.bestModel.par.shape], dtype=results.bestModel.par.dtype)
#        hdfFile.create_dataset('bestmodel.depth', [nPoints,*results.bestModel.depth.shape], dtype=results.bestModel.depth.dtype)
#        hdfFile.create_dataset('bestmodel.thk', [nPoints,*results.bestModel.thk.shape], dtype=results.bestModel.thk.dtype)
#        hdfFile.create_dataset('bestmodel.chie', [nPoints,*results.bestModel.chie.shape], dtype=results.bestModel.chie.dtype)
#        hdfFile.create_dataset('bestmodel.chim', [nPoints,*results.bestModel.chim.shape], dtype=results.bestModel.chim.dtype)

#        self.currentD.createHdf(grp, 'currentd')
#        self.bestD.createHdf(grp, 'bestd')
#
#        tmp=self.bestModel.pad(self.bestModel.maxLayers)
#        tmp.createHdf(grp, 'bestmodel')

    def results2Hdf(self, results):
        """ Given a HDF file initialized as line results, write the contents of results to the appropriate arrays """

        assert results.fiducial in self.fiducials, Exception("The HDF file does not have ID number {}. Available ids are between {} and {}".format(results.fiducial, np.min(self.fiducials), np.max(self.fiducials)))

        hdfFile = self.hdfFile

        # Get the point index
        i = self.fiducials.searchsorted(results.fiducial)

        # Add the iteration number
        hdfFile['i'][i] = results.i

        # Add the burn in iteration
        hdfFile['iburn'][i] = results.iBurn

        # Add the burned in logical
        hdfFile['burnedin'][i] = results.burnedIn

        # Add the depth of investigation
        #hdfFile['doi'][i] = results.doi()

        # Add the multiplier
        hdfFile['multiplier'][i] = results.multiplier

        # Add the inversion time
        hdfFile['invtime'][i] = results.invTime

        # Add the savetime
#        hdfFile['savetime'][i] = results.saveTime

        # Interpolate the mean and best model to the discretized hitmap
        hm = results.currentModel.par.posterior
        results.meanInterp[:] = hm.axisMean()
        results.bestInterp[:] = results.bestModel.interpPar2Mesh(results.bestModel.par, hm)
        # results.opacityInterp[:] = results.Hitmap.confidenceRange(percent=95.0, log='e')

        slic = np.s_[i, :]
        # Add the interpolated mean model
        results.meanInterp.writeHdf(hdfFile, 'meaninterp',  index=slic)
        # Add the interpolated best
        results.bestInterp.writeHdf(hdfFile, 'bestinterp',  index=slic)
        # Add the interpolated opacity
        # results.opacityInterp.writeHdf(hdfFile, 'opacityinterp',  index=slic)

        # Add the acceptance rate
        results.rate.writeHdf(hdfFile, 'rate', index=slic)
        

        # Add the data misfit
        results.PhiDs.writeHdf(hdfFile,'phids',index=slic)

        # Add the layer histogram counts
        # results.kHist.writeHdf(hdfFile,'khist',index=slic)

        # # Add the elevation histogram counts
        # results.DzHist.writeHdf(hdfFile,'dzhist',index=slic)

        # Add the interface histogram counts
        # results.MzHist.writeHdf(hdfFile,'mzhist',index=slic)

        # # Add the relative and additive errors
        # for j in range(results.nSystems):
        #     results.relErr[j].writeHdf(hdfFile, "relerr" + str(j), index=slic)
        #     results.addErr[j].writeHdf(hdfFile, "adderr" + str(j), index=slic)

        # Add the hitmap
        # results.Hitmap.writeHdf(hdfFile,'hitmap',  index=i)

        results.currentDataPoint.writeHdf(hdfFile,'currentdatapoint',  index=i)

        results.bestDataPoint.writeHdf(hdfFile,'bestd', withPosterior=False, index=i)

        results.currentModel.writeHdf(hdfFile,'currentmodel', index=i)

        results.bestModel.writeHdf(hdfFile,'bestmodel', withPosterior=False, index=i)

#        if results.verbose:
#            results.posteriorComponents.writeHdf(hdfFile, 'posteriorcomponents',  index=np.s_[i,:,:])
