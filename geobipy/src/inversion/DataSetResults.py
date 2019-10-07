""" @DataSetResults
Class to handle the HDF5 result files for a full data set.
 """
from ..base import Error as Err
import matplotlib.pyplot as plt
import numpy as np
import h5py
#import numpy.ma as ma
from ..classes.core.myObject import myObject
from ..classes.core import StatArray
from ..base.fileIO import fileExists

from ..classes.statistics.Histogram1D import Histogram1D
from ..classes.statistics.Hitmap2D import Hitmap2D
from ..classes.pointcloud.PointCloud3D import PointCloud3D
from ..base import interpolation as interpolation
from .LineResults import LineResults
#from ..classes.statistics.Distribution import Distribution
from ..base.HDF import hdfRead
from ..base import customPlots as cP
from os.path import join
from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
#from ..base import fileIO as fIO
#from os.path import split
#from pyvtk import VtkData, UnstructuredGrid, CellData, Scalars

from os import listdir
import progressbar

class DataSetResults(myObject):
    """ Class to define results from Inv_MCMC for a full data set """

    def __init__(self, directory, files = None, sysPath = None):
        """ Initialize the lineResults
        directory = directory containing folders for each line of data results
        """
        self.directory = directory
        self.fileList = None
        self._nPoints = None
        self.cumNpoints = None
        self.nLines = None
        self.bounds = None
        if (files is None):
            self.getFileList(directory)
        else:
            self.fileList = files
            self.nLines = len(files)

        self.lines=[]
        for i in range(self.nLines):
            fName = self.fileList[i]
            fileExists(fName)
            self.lines.append(LineResults(fName, sysPath=sysPath))
        self._lineIndices = None

        self._pointcloud = None
        self._additiveError = None
        self._relativeError = None
        self.kdtree = None
        self._meanParameters = None
        self._bestParameters = None
        self.doi = None
        self.doi2D = None
        self.mean3D = None
        self.best3D = None
        self._facies = None
        self._interfaces = None


    def open(self):
        """ Check whether the file is open """
        for line in self.lines:
            line.open()


    def close(self):
        """ Check whether the file is open """
        for line in self.lines:
            line.close()


    def getFileList(self, directory):
        """ Get the list of line result files for the dataset """
        self.fileList = []
        for file in [f for f in listdir(directory) if f.endswith('.h5')]:
            fName = join(directory,file)
            fileExists(fName)
            self.fileList.append(fName)
            
        self.fileList = sorted(self.fileList)
        
        assert len(self.fileList) > 0, 'Could not find .h5 files in current directory'
        self.nLines = len(self.fileList)


    @property
    def additiveError(self):

        if self._additiveError is None:
            self._additiveError = StatArray.StatArray([self.nSystems, self.nPoints], name=self.lines[0].additiveError.name, units=self.lines[0].additiveError.units, order = 'F')

            print('Reading additive error')
            Bar=progressbar.ProgressBar()
            for i in Bar(range(self.nLines)):
                self._additiveError[:, self.lineIndices[i]] = self.lines[i].additiveError.T
                self.lines[i]._additiveError = None # Free memory

        return self._additiveError


    @property
    def bestParameters(self):

        if (self._bestParameters is None):

            self._bestParameters = StatArray.StatArray([self.zGrid.nCells, self.nPoints], name=self.lines[0].bestParameters.name, units=self.lines[0].bestParameters.units, order = 'F')

            print('Reading best parameters')
            Bar=progressbar.ProgressBar()
            for i in Bar(range(self.nLines)):
                self._bestParameters[:, self.lineIndices[i]] = self.lines[i].bestParameters.T
                self.lines[i]._bestParameters = None # Free memory

        return self._bestParameters


    @property
    def hitmapCounts(self):
        if (self._counts is None):
            mesh = self.lines[0].mesh
            self._counts = np.empty([])


    @property
    def elevation(self):
        return self.pointcloud.elevation

    
    @property
    def facies(self):
        assert not self._facies is None, Exception("Facies must be set using self.setFaciesProbabilities()")
        return self._facies

    
    @property
    def height(self):
        return self.pointcloud.z


    def identifyPeaks(self, depths, nBins = 250, width=4, limits=None):
        """Identifies peaks in the parameter posterior for each depth in depths.

        Parameters
        ----------
        depths : array_like
            Depth intervals to identify peaks between.

        Returns
        -------

        """

        print(limits)

        out = self.lines[0].identifyPeaks(depths, nBins, width, limits)
        for line in self.lines[1:]:
            out = np.vstack([out, line.identifyPeaks(depths, nBins, width, limits)])        

        return out

    
    @property
    def interfaces(self):
        if (self._interfaces is None):

            self._interfaces = StatArray.StatArray([self.zGrid.nCells, self.nPoints], name=self.lines[0].interfaces.name, units=self.lines[0].interfaces.units)

            print('Reading interfaces')
            Bar=progressbar.ProgressBar()
            for i in Bar(range(self.nLines)):
                self._interfaces[:, self.lineIndices[i]] = self.lines[i].interfaces.T
                self.lines[i]._interfaces = None # Free memory

        return self._interfaces


    @property
    def lineIndices(self):
        
        if self._lineIndices is None:
            self._lineIndices = []
            i0 = 0
            for i in range(self.nLines):
                i1 = i0 + self.lines[i].nPoints
                self._lineIndices.append(np.s_[i0:i1])
                i1 = i0
        return self._lineIndices


    @property
    def meanParameters(self):

        if (self._meanParameters is None):

            self._meanParameters = StatArray.StatArray([self.zGrid.nCells, self.nPoints], name=self.lines[0].meanParameters.name, units=self.lines[0].meanParameters.units, order = 'F')

            print('Reading mean parameters')
            Bar=progressbar.ProgressBar()
            for i in Bar(range(self.nLines)):
                self._meanParameters[:, self.lineIndices[i]] = self.lines[i].meanParameters
                self.lines[i]._meanParameters = None # Free memory

        return self._meanParameters

        
    @property
    def nPoints(self):
        """ Get the total number of data points """
        if (self._nPoints is None):
            tmp = np.asarray([this.nPoints for this in self.lines])
            self._cumNpoints = np.cumsum(tmp)
            self._nPoints = np.sum(tmp)
        return self._nPoints


    @property
    def nSystems(self):
        """ Get the number of systems """
        return self.lines[0].nSystems

    
    def parameterHistogram(self, nBins, depth = None, depth2 = None, log=None):
        """ Compute a histogram of all the parameter values, optionally show the histogram for given depth ranges instead """

        out = self.lines[0].parameterHistogram(nBins=nBins, depth=depth, depth2=depth2, log=log)

        for line in self.lines[1:]:
            tmp = line.parameterHistogram(nBins=nBins, depth=depth, depth2=depth2, log=log)
            out._counts += tmp.counts

        return out
        

    @property
    def pointcloud(self):

        if self._pointcloud is None:
            x = StatArray.StatArray(self.nPoints, name=self.lines[0].x.name, units=self.lines[0].x.units)
            y = StatArray.StatArray(self.nPoints, name=self.lines[0].y.name, units=self.lines[0].y.units)
            z = StatArray.StatArray(self.nPoints, name=self.lines[0].height.name, units=self.lines[0].height.units)
            e = StatArray.StatArray(self.nPoints, name=self.lines[0].elevation.name, units=self.lines[0].elevation.units)
            # Loop over the lines in the data set and get the attributes
            print('Reading co-ordinates')
            Bar = progressbar.ProgressBar()
            for i in Bar(range(self.nLines)):
                indices = self.lineIndices[i]
                x[indices] = self.lines[i].x
                y[indices] = self.lines[i].y
                z[indices] = self.lines[i].height
                e[indices] = self.lines[i].elevation

                self.lines[i]._x = None
                self.lines[i]._y = None
                self.lines[i]._height = None
                self.lines[i]._elevation = None

            self._pointcloud = PointCloud3D(self.nPoints, x, y, z, e)
        return self._pointcloud


    @property
    def relativeError(self):

        if self._relativeError is None:
            self._relativeError = StatArray.StatArray([self.nSystems, self.nPoints], name=self.lines[0].relativeError.name, units=self.lines[0].relativeError.units, order = 'F')

            print('Reading relative error')
            Bar=progressbar.ProgressBar()
            for i in Bar(range(self.nLines)):
                self._relativeError[:, self.lineIndices[i]] = self.lines[i].relativeError.T
                self.lines[i]._relativeError = None # Free memory

        return self._relativeError


    @property
    def x(self):
        return self.pointcloud.x
            

    @property
    def y(self):
        return self.pointcloud.y

    
    def dataPointResults(self, fiducial, reciprocateParameters=False):
        """Get the inversion results for the given fiducial. 
        
        Parameters
        ----------
        fiducial : float
            Unique fiducial of the data point.
            
        Returns
        -------
        out : geobipy.Results
            The inversion results for the data point.
            
        """

        index = self.fiducialIndex(fiducial)

        return self.lines[index[0]].getResults(index=index[1], reciprocateParameter=reciprocateParameters)
       

    def getLineNumber(self, i):
        """ Get the line number for the given data point index """
        return self.lines[self.getLineIndex(i)].line


    def getLineIndex(self, i):
        """ Get the line file name for the given data point index """
        assert i >= 0, 'Datapoint index must be >= 0'
        if i > self.nPoints-1: raise IndexError('index {} is out of bounds for data point index with size {}'.format(i, self.nPoints))
        return self._cumNpoints.searchsorted(i)


    def fiducial(self, i):
        """ Get the ID of the given data point """
        iLine = self.getLineIndex(i)
        if (iLine > 0):
            i -= self.cumNpoints[iLine-1]
        return self.lines[iLine].iDs[i]

    
    def fiducialIndex(self, fiducial):
        """Get the line number and index for the specified fiducial.
        
        Parameters
        ----------
        fiducial : float
            The unique fiducial for the data point

        Returns
        -------
        out : ints
            Line number and index of the data point in that line.

        """
        
        if np.size(fiducial) == 1:
            for iLine, line in enumerate(self.lines):
                if fiducial in line.fiducials:
                    return np.asarray([iLine, line.fiducials.searchsorted(fiducial)])

        indices = []
        for iLine, line in enumerate(self.lines):
            for fid in fiducial:
                if fid in line.fiducials:
                    indices.append([iLine, line.fiducials.searchsorted(fid)])
        return np.asarray(indices)


    def histogram(self, nBins, depth1 = None, depth2 = None, reciprocateParameter = False, bestModel = False, withDoi=False, percent=67.0, force = False, **kwargs):
        """ Compute a histogram of the model, optionally show the histogram for given depth ranges instead """

        if (depth1 is None):
            depth1 = self.zGrid.cellCentres[0]
        if (depth2 is None):
            depth2 = self.zGrid.cellCentres[-1]

        # Ensure order in depth values
        if (depth1 > depth2):
            tmp=depth2
            depth2 = depth1
            depth1 = tmp

        # Don't need to check for depth being shallower than zGrid[0] since the sortedsearch with return 0
        if (depth1 > self.zGrid.cellEdges[-1]): Err.Emsg('mapDepthSlice: Depth is greater than max depth - '+str(self.zGrid.cellEdges[-1]))
        if (depth2 > self.zGrid.cellEdges[-1]): Err.Emsg('mapDepthSlice: Depth2 is greater than max depth - '+str(self.zGrid.cellEdges[-1]))

        if (bestModel):
            model = self.bestParameters
        else:
            model = self.meanParameters

        if withDoi:
            depth1 = np.minimum(self.doi, depth1)
            depth2 = np.minimum(self.doi, depth2)
            z = np.repeat(self.zGrid[:,np.newaxis],self.nPoints,axis=1)
            vals = model[(z > depth1)&(z < depth2)]
        else:
            cell1 = self.zGrid.cellCentres.searchsorted(depth1)
            cell2 = self.zGrid.cellCentres.searchsorted(depth2)

            vals = model[cell1:cell2+1, :]

        log = kwargs.pop('log',False)

        if (reciprocateParameter):
            vals = 1.0/vals
            name = 'Resistivity'
            units = '$\Omega m$'
        else:
            name = 'Conductivity'
            units = '$Sm^{-1}$'

        if (log):
            vals,logLabel=cP._log(vals,log)
            name = logLabel + name
        vals = StatArray.StatArray(vals, name, units)

        h = Histogram1D(np.linspace(vals.min(),vals.max(),nBins))
        h.update(vals)
        h.plot(**kwargs)
        return h


    @property
    def zGrid(self):
        """ Gets the discretization in depth """
        return self.lines[0].mesh.z


    # def getAttribute(self,  mean=False, best=False, opacity=False, doi=False, relErr=False, addErr=False, percent=67.0, force=False):
    #     """ Get a subsurface property """

    #     assert (not all([not mean, not best, not opacity, not doi, not relErr, not addErr])), 'Please choose at least one attribute' + help(self.getAttrubute)

    #     # Turn off attributes that are already loaded
    #     if (mean):
    #         mean=self.mean is None
    #     if (best):
    #         best=self.best is None
    #     if (relErr):
    #         relErr=self.relErr is None
    #     # Getting the doi is cheap, so always ask for it even if opacity is requested
    #     doi = opacity or doi
    #     if (doi):
    #         doi=self.doi is None

    #     if (all([not mean, not best, not opacity, not doi, not relErr, not addErr])):
    #         return

    #     # Get the number of systems
    #     if (relErr or addErr):
    #         self.getNsys()

    #     if (mean or best or doi):
    #         # Get the number of cells
    #         nz = self.zGrid.nCells

        
    #     if (doi):
    #         self.opacity=np.zeros([nz,self.nPoints], order = 'F')
    #         self.doi = StatArray.StatArray(np.zeros(self.nPoints),'Depth of Investigation','m')
    #     if (relErr):
    #         self.lines[0].getRelativeError()
    #         if (self.nSys > 1):
    #             self.relErr = StatArray.StatArray([self.nPoints, self.nSys],name=self.lines[0].relErr.name,units=self.lines[0].relErr.units, order = 'F')
    #         else:
    #             self.relErr = StatArray.StatArray(self.nPoints,name=self.lines[0].relErr.name,units=self.lines[0].relErr.units, order = 'F')
            
    #     # Loop over the lines in the data set and get the attributes
    #     print('Reading attributes from dataset results')
    #     Bar=progressbar.ProgressBar()
    #     for i in Bar(range(self.nLines)):

    #         # Perform line getters
    #         if (mean):
    #             self.mean = StatArray.StatArray([nz,self.nPoints], name=self.lines[0].meanParameters.name, units=self.lines[0].meanParameters.units, order = 'F')
    #             self.mean[:, self.lineIndices[i]] = self.lines[i].meanParameters.T
    #             self.lines[i].mean = None # Free memory
    #         if (best):
    #             self.best[:, self.lineIndices[i]] = self.lines[i].bestParameters.T
    #             self.lines[i].best = None # Free memory
    #         if (doi):
    #             # Get the DOI for this line
    #             self.lines[i].getDOI(percent)
    #             self.opacity[:, self.lineIndices[i]] = self.lines[i].opacity.T
    #             self.doi[self.lineIndices[i]] = self.lines[i].doi
    #             self.lines[i].opacity = None # Free memory
    #             self.lines[i].doi = None # Free memory
    #         # Deallocate line attributes to save space
    #         self.lines[i]._hitMap = None

    #     if (xy):
    #         self.points.getBounds() # Get the bounding box
            

    def getMean3D(self, dx, dy, mask = False, clip = False, force=False, method='ct'):
        """ Interpolate each depth slice to create a 3D volume """
        if (not self.mean3D is None and not force): return
           
        # Test for an existing file, created with the same parameters.
        # Read it and return if it exists.
        file = 'mean3D.h5'
        if fileExists(file):
            variables = hdfRead.read_all(file)
            if (dx == variables['dx'] and dy == variables['dy'] and mask == variables['mask'] and clip == variables['clip'] and method == variables['method']):
                self.mean3D = variables['mean3d']
                return
            
           
        method = method.lower()
        if method == 'ct':
            self.__getMean3D_CloughTocher(dx=dx, dy=dy, mask=mask, clip=clip, force=force)
        elif method == 'mc':
            self.__getMean3D_minimumCurvature(dx=dx, dy=dy, mask=mask, clip=clip, force=force)
        else:
            assert False, ValueError("method must be either 'ct' or 'mc' ")
            
        with h5py.File('mean3D.h5','w') as f:
            f.create_dataset(name = 'dx', data = dx)
            f.create_dataset(name = 'dy', data = dy)
            f.create_dataset(name = 'mask', data = mask)
            f.create_dataset(name = 'clip', data = clip)
            f.create_dataset(name = 'method', data = method)
            self.mean3D.toHdf(f,'mean3d')
            
        
    
    def __getMean3D_minimumCurvature(self, dx, dy, mask=None, clip=False, force=False):
               
        
        x = self.pointcloud.x.deepcopy()
        y = self.pointcloud.y.deepcopy()

        values = self.meanParameters[0, :]
        x1, y1, vals = interpolation.minimumCurvature(x, y, values, self.pointcloud.bounds, dx=dx, dy=dy, mask=mask, clip=clip, iterations=2000, tension=0.25, accuracy=0.01)

        # Initialize 3D volume
        mean3D = StatArray.StatArray(np.zeros([self.zGrid.nCells, y1.size, x1.size], order = 'F'),name = 'Conductivity', units = '$Sm^{-1}$')
        mean3D[0, :, :] = vals
        
        # Interpolate for each depth
        print('Interpolating using minimum curvature')
        Bar=progressbar.ProgressBar()
        for i in Bar(range(1, self.zGrid.nCells)):
            # Get the model values for the current depth
            values = self.meanParameters[i, :]
            dum1, dum2, vals = interpolation.minimumCurvature(x, y, values, self.pointcloud.bounds, dx=dx, dy=dy, mask=mask, clip=clip, iterations=2000, tension=0.25, accuracy=0.01)
            # Add values to the 3D array
            mean3D[i, :, :] = vals
                  
        self.mean3D = mean3D 

    
    def __getMean3D_CloughTocher(self, dx, dy, mask=None, clip=False, force=False):
        
        # Get the discretization
        if (dx is None):
            tmp = self.pointcloud.bounds[1]-self.pointcloud.bounds[0]
            dx = 0.01 * tmp
        assert dx > 0.0, "dx must be positive!"
        
        # Get the discretization
        if (dy is None):
            tmp = self.pointcloud.bounds[3]-self.pointcloud.bounds[2]
            dy = 0.01 * tmp
        assert dy > 0.0, "dy must be positive!"
        
        tmp = np.column_stack((self.pointcloud.x, self.points.y))

        # Get the points to interpolate to
        x,y,intPoints = interpolation.getGridLocations2D(self.pointcloud.bounds, dx, dy)

        # Create a distance mask
        if mask:
            self.pointcloud.setKdTree(nDims=2) # Set the KdTree on the data points
            g = np.meshgrid(x,y)
            xi = _ndim_coords_from_arrays(tuple(g), ndim=tmp.shape[1])
            dists, indexes = self.points.kdtree.query(xi)
            iMask = np.where(dists > mask)

        # Get the value bounds
        minV = np.nanmin(self.mean)
        maxV = np.nanmax(self.mean)

        # Initialize 3D volume
        mean3D = StatArray.StatArray(np.zeros([self.zGrid.size, y.nCells, x.nCells], order = 'F'),name = 'Conductivity', units = '$Sm^{-1}$')

        # Triangulate the data locations
        dTri = Delaunay(tmp)

        # Interpolate for each depth
        print('Interpolating using clough tocher')
        Bar=progressbar.ProgressBar()
        for i in Bar(range(self.zGrid.size)):
            # Get the model values for the current depth
            vals1D = self.mean[i,:]
            # Create the interpolant
            f=CloughTocher2DInterpolator(dTri,vals1D)
            # Interpolate to the grid
            vals = f(intPoints)
            # Reshape to a 2D array
            vals = vals.reshape(y.size,x.size)

            # clip values to the observed values
            if (clip):
                vals.clip(minV, maxV)

            # Mask based on distance
            if (mask):
                vals[iMask] = np.nan

            # Add values to the 3D array
            mean3D[i,:,:] = vals
        self.mean3D = mean3D #.reshape(self.zGrid.size*y.size*x.size)

    
    def map(self, dx, dy, values, system=0, mask = None, clip = True, **kwargs):
        """ Create a map of a parameter """
        nPlots = values.ndim

        for i in range(nPlots):
            plt.subplot(nPlots, 1, i+1)
            self.pointcloud.mapPlot(dx = dx, dy = dy, mask = mask, clip = clip, c = np.atleast_2d(values)[i, :], **kwargs)


    def mapFacies(self, dx, dy, depth,  **kwargs):

        cell1 = self.zGrid.cellIndex(depth)

        nFacies = self.facies.shape[1]

        for i in range(nFacies):
            plt.subplot(nFacies, 1, i+1)
            self.pointcloud.mapPlot(dx = dx, dy = dy, c = self.facies[:, i, cell1], **kwargs)


    def setFaciesProbabilities(self, fractions, distributions, reciprocateParameter=False, log=None):
        """ Assign facies to the parameter model given the pdfs of each facies
        mean:    :Means of the normal distributions for each facies
        var:     :Variance of the normal distributions for each facies
        volFrac: :Volume fraction of each facies
        """

        self._facies = np.empty([self.nPoints, np.size(distributions), self.zGrid.nCells])

        g = self.lines[0].hdfFile['currentmodel/par/posterior']

        i0 = 0
        Bar = progressbar.ProgressBar()
        for i in Bar(range(self.lines[0].nPoints)):
            self._facies[i, :, :] = Hitmap2D().fromHdf(g, index=i).marginalProbability(fractions, distributions, reciprocate=reciprocateParameter, log=log, axis=0)


    def percentageParameter(self, value, depth, depth2=None):  

        percentage = StatArray.StatArray(np.empty(self.nPoints), name="Probability of {} > {:0.2f}".format(self.meanParameters.name, value), units = self.meanParameters.units)

        print(self.lineIndices)
        print('Calculating percentages')
        Bar=progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            percentage[self.lineIndices[i]] = self.lines[i].percentageParameter(value, depth, depth2)
        
        return percentage
            

    def getDepthSlice(self, depth, depth2=None, reciprocateParameter=False, bestModel=False, force=False):

        # Get the depth grid
        assert depth <= self.zGrid.cellEdges[-1], 'Depth is greater than max depth '+str(self.zGrid.cellEdges[-1])
        if (not depth2 is None):
            assert depth2 <= self.zGrid.cellEdges[-1], 'Depth2 is greater than max depth '+str(self.zGrid.cellEdges[-1])
            assert depth <= depth2, 'Depth2 must be >= depth'

        if (bestModel):
            model = self.bestParameters
        else:
            model = self.meanParameters

        model[model == 0.0] = 1.0

        cell1 = self.zGrid.cellIndex(depth)

        if (depth2 is None):
            vals1D = model[cell1, :]
        else:
            cell2 = self.zGrid.cellIndex(depth2)
            vals1D = np.mean(model[cell1:cell2+1,:], axis = 0)

        if (reciprocateParameter):
            vals1D = StatArray.StatArray(1.0/vals1D, name = 'Resistivity', units = '$\Omega m$')
        else:
            vals1D = StatArray.StatArray(vals1D, name = 'Conductivity', units = '$Sm^{-1}$')
        return vals1D


    def mapDepthSlice(self, dx, dy, depth, depth2 = None, reciprocateParameter = False, bestModel = False, mask = None, clip = True, **kwargs):
        """ Create a depth slice through the recovered model """

        vals1D = self.getDepthSlice(depth=depth, depth2=depth2, reciprocateParameter=reciprocateParameter, bestModel=bestModel)
        
        ax = self.map(dx, dy, vals1D, mask = mask, clip = clip, **kwargs)
        
        return ax


    def mapElevation(self,dx, dy, mask = None, clip = True, extrapolate=None, force=False, **kwargs):
        """ Create a map of a parameter """
        self.pointcloud.mapPlot(dx = dx, dy = dy, mask = mask, clip = clip, extrapolate=extrapolate, c = self.elevation, **kwargs)
        # cP.xlabel('Easting (m)')
        # cP.ylabel('Northing (m)')

    # def mapDoi(self, dx, dy, mask = None, clip = True, extrapolate=None, force=False, **kwargs):
    #     """ Create a map of a parameter """
    #     self.getAttribute(xy=True, doi=True, force=force)
    #     self.points.mapPlot(dx = dx, dy = dy, mask = mask, clip = clip, extrapolate=extrapolate, c = self.doi, **kwargs)
    #     cP.xlabel('Easting (m)')
    #     cP.ylabel('Northing (m)')


    def plotDepthSlice(self, depth, depth2 = None, reciprocateParameter = False, bestModel = False, mask = None, clip = True, force=False, *args, **kwargs):
        """ Create a depth slice through the recovered model """

        vals1D = self.getDepthSlice(depth=depth, depth2=depth2, reciprocateParameter=reciprocateParameter, bestModel=bestModel, force=force)
        ax = self.pointcloud.scatter2D(c = vals1D, *args, **kwargs)
        return ax


    def scatter2D(self, values, **kwargs):

        if (not 'edgecolor' in kwargs):
            kwargs['edgecolor']='k'
        if (not 's' in kwargs):
            kwargs['s']=10.0

        nPlots = values.ndim

        for i in range(nPlots):
            plt.subplot(nPlots, 1, i+1)
            self.pointcloud.scatter2D(c = np.atleast_2d(values)[i, :], **kwargs)

    
    def plotAdditiveError(self, **kwargs):
        """ Plot the observation locations """
        self.scatter2D(self.additiveError, **kwargs)

    
    def plotBestFacies(self, depth):

        cell1 = self.zGrid.cellIndex(depth)

        nFacies = self.facies.shape[1]
        bestFacies = StatArray.StatArray(np.argmax(self.facies[:, :, cell1], axis=1), 'Best Facies')

        for i in range(nFacies):
            j = np.where(bestFacies == i)[0]
            plt.scatter(self.pointcloud.x[j], self.pointcloud.y[j], label='Facies {}'.format(i), c=cP.wellSeparated[i])
        plt.legend()


    def plotInterfaceProbability(self, depth, lowerThreshold=0.0, **kwargs):

        cell1 = self.zGrid.cellIndex(depth)

        slce = self.interfaces[cell1, :]
        if lowerThreshold > 0.0:
            slce = self.interfaces[cell1, :].deepcopy()
            slce[slce < lowerThreshold] = np.nan

        self.pointcloud.scatter2D(c = slce, **kwargs)


    def plotElevation(self, **kwargs):
        """ Plot the observation locations """
        self.scatter2D(self.elevation, **kwargs)

    
    def plotFacies(self, depth, **kwargs):

        cell1 = self.zGrid.cellIndex(depth)

        nFacies = self.facies.shape[1]

        for i in range(nFacies):
            plt.subplot(nFacies, 1, i+1)
            self.pointcloud.scatter2D(c = self.facies[:, i, cell1], **kwargs)


    def plotRelativeError(self, **kwargs):
        """ Plot the observation locations """
        self.scatter2D(self.relativeError, **kwargs)


    def plotXplot(self, bestModel=True, withDoi=True, reciprocateParameter=True, log10=True, **kwargs):
        """ Plot the cross plot of a model against depth """

        tmp = self.getParVsZ(bestModel=bestModel, withDoi=withDoi, reciprocateParameter=reciprocateParameter, log10=log10)
        # Repeat the depths for plotting
        cP.plot(tmp[:,0], tmp[:,1], **kwargs)
        if (bestModel):
            cP.xlabel(self.best.getNameUnits())
        else:
            cP.xlabel(self.mean.getNameUnits())
        cP.ylabel(self.zGrid.getNameUnits())
        return tmp

    
    def plotXsection(self, line, xAxis='easting', reciprocateParameter = False, bestModel=False, percent = 67.0, useVariance=True, **kwargs):
        """ Plot a x section for the ith line """
        self.lines[line].setAlonglineAxis(xAxis)
        self.lines[line].plotXsection(reciprocateParameter=reciprocateParameter, bestModel=bestModel, percent=percent, useVariance=useVariance, **kwargs)


    def getParVsZ(self, bestModel=False, withDoi=True, reciprocateParameter=True, log10=True, clipNan=True):
        """ Get the depth and parameters, optionally within the doi """
        # Get the depths
        z = np.tile(self.zGrid,self.nPoints)

        if (bestModel):
            self.getAttribute(best=True, doi=withDoi)
            model = np.zeros(self.best.shape)
            model[:,:] = self.best
        else:
            self.getAttribute(mean=True, doi=withDoi)
            model = np.zeros(self.best.shape)
            model[:,:] = self.mean

        if (withDoi):
            zTmp = np.repeat(self.zGrid[:,np.newaxis],self.nPoints,axis=1)
            model[zTmp > self.doi] = np.nan

        model = model.reshape(model.size, order='F')

        if reciprocateParameter:
            model = 1.0/model.reshape(model.size, order='F')

        if log10:
            model = np.log10(model)

        res = StatArray.StatArray(np.column_stack((model,z)))

        if (clipNan):
            res = res[np.logical_not(np.isnan(res[:,0]))]

        return res

    def kMeans(self, nClusters, precomputedParVsZ=None, standardize=False, log10Depth=False, plot=False, bestModel=True, withDoi=True, reciprocateParameter=True, log10=True, clipNan=True, **kwargs):
        """  """
        if (precomputedParVsZ is None):
            ParVsZ = self.getParVsZ(bestModel=bestModel, withDoi=withDoi, reciprocateParameter=reciprocateParameter, log10=log10, clipNan=clipNan)
        else:
            ParVsZ = precomputedParVsZ

        assert isinstance(ParVsZ, StatArray.StatArray), "precomputedParVsZ must be an StatArray"

        if (log10Depth):
            ParVsZ[:,1] = np.log10(ParVsZ[:,1])

        return ParVsZ.kMeans(nClusters, standardize=standardize, nIterations=10, plot=plot, **kwargs)


    def GMM(self, ParVsZ, clusterID, trainPercent=90.0, plot=True):
        """ Classify the subsurface parameters """
        assert isinstance(ParVsZ, StatArray.StatArray), "ParVsZ must be an StatArray"
        ParVsZ.GMM(clusterID, trainPercent=trainPercent, covType=['spherical','tied','diag','full'], plot=plot)



    def toVTK(self, fName, dx, dy, mask=False, clip=False, force=False, method='ct'):
        """ Convert a 3D volume of interpolated values to vtk for visualization in Paraview """

        self.getMean3D(dx=dx, dy=dy, mask=mask, clip=clip, force=force, method=method)
        self.pointcloud.getBounds()

        x, y, intPoints = interpolation.getGridLocations2D(self.pointcloud.bounds, dx, dy)
        z = self.zGrid


        from pyvtk import VtkData, UnstructuredGrid, PointData, CellData, Scalars

        # Get the 3D dimensions
        mx = x.size
        my = y.size
        mz = z.nCells
        
        nPoints = mx * my * mz
        nCells = (mx-1)*(my-1)*(mz-1)

        # Interpolate the elevation to the grid nodes
        if (method == 'ct'):
            tx,ty, vals = self.pointcloud.interpCloughTocher(self.elevation, dx = dx,dy=dy, mask = mask, clip = clip, extrapolate='nearest')
        elif (method == 'mc'):
            tx,ty, vals = self.pointcloud.interpMinimumCurvature(self.elevation, dx = dx, dy=dy, mask = mask, clip = clip)
            
        vals = vals[:my,:mx]
        vals = vals.reshape(mx*my)

        # Set up the nodes and voxel indices
        points = np.zeros([nPoints,3], order='F')
        points[:,0] = np.tile(x, my*mz)
        points[:,1] = np.tile(y.repeat(mx), mz)
        points[:,2] = np.tile(vals, mz) - z.cellCentres.repeat(mx*my)

        # Create the cell indices into the points
        p = np.arange(nPoints).reshape((mz, my, mx))
        voxels = np.zeros([nCells, 8], dtype=np.int)
        iCell = 0
        for k in range(mz-1):
            k1 = k + 1
            for j in range(my-1):
                j1 = j + 1
                for i in range(mx-1):
                    i1 = i + 1
                    voxels[iCell,:] = [p[k1,j,i],p[k1,j,i1],p[k1,j1,i1],p[k1,j1,i], p[k,j,i],p[k,j,i1],p[k,j1,i1],p[k,j1,i]]
                    iCell += 1

        # Create the various point data
        pointID = Scalars(np.arange(nPoints), name='Point iD')
        pointElev = Scalars(points[:,2], name='Point Elevation (m)')

        tmp = self.mean3D.reshape(np.size(self.mean3D))
        tmp[tmp == 0.0] = np.nan

        print(np.nanmin(tmp), np.nanmax(tmp))
        tmp1 = 1.0 / tmp

        print(np.nanmin(tmp), np.nanmax(tmp))
        pointRes = Scalars(tmp1, name = 'log10(Resistivity) (Ohm m)')
        tmp1 = np.log10(tmp)

        pointCon = Scalars(tmp1, name = 'log10(Conductivity) (S/m)')

        print(nPoints, tmp.size)
        
        PData = PointData(pointID, pointElev, pointRes)#, pointCon)
        CData = CellData(Scalars(np.arange(nCells),name='Cell iD'))
        vtk = VtkData(
              UnstructuredGrid(points,
                               hexahedron=voxels),
#                               ),
              PData,
              CData,
              'Some Name'
              )

        vtk.tofile(fName, 'binary')










































if __name__ == "__main__":

    d="/Users/nfoks/Projects/bMinsley/Tdem/SLV/ScatterV"
    ds = DataSetResults(d)
