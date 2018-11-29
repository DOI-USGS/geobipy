""" @DataSetResults
Class to handle the HDF5 result files for a full data set.
 """
from ..base import Error as Err
import numpy as np
import h5py
#import numpy.ma as ma
from ..classes.core.myObject import myObject
from ..classes.core.StatArray import StatArray
from ..base.fileIO import fileExists

from ..classes.statistics.Histogram1D import Histogram1D
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

    def __init__(self, directory, files = None):
        """ Initialize the lineResults
        directory = directory containing folders for each line of data results
        """
        self.directory = directory
        self.fileList = None
        self.nPoints = None
        self.cumNpoints = None
        self.nSys = None
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
            self.lines.append(LineResults(fName))

        self.points = None
        self.elevation= None
        self.addErr = None
        self.relErr = None
        self.kdtree = None
        self.mean = None
        self.best = None
        self.doi = None
        self.doi2D = None
        self.mean3D = None
        self.best3D = None
        self.zGrid = None


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
        

    def getNpoints(self, force=False):
        """ Get the total number of data points """
        if (not self.nPoints is None and not force): return
        tmp=np.asarray([this.nPoints for this in self.lines])
        self.cumNpoints = np.cumsum(tmp)
        self.nPoints = np.sum(tmp)

    def getLineNumber(self, i):
        """ Get the line number for the given data point index """
        return self.lines[self.getLineIndex(i)].line

    def getLineIndex(self, i):
        """ Get the line file name for the given data point index """
        self.getNpoints()
        assert i >= 0, 'Datapoint index must be >= 0'
        if i > self.nPoints-1: raise IndexError('index {} is out of bounds for data point index with size {}'.format(i,self.nPoints))
        return self.cumNpoints.searchsorted(i)

    def getID(self, i):
        """ Get the ID of the given data point """
        iLine = self.getLineIndex(i)
        if (iLine > 0):
            i -= self.cumNpoints[iLine-1]
        return self.lines[iLine].iDs[i]


    def getNsys(self, force=False):
        """ Get the number of systems """
        if (not self.nSys is None and not force): return
        self.nSys = self.lines[0].getAttribute('# of systems')


    def histogram(self,nBins, depth1 = None, depth2 = None, invertPar = True, bestModel = False, withDoi=False, percent=67.0, force = False, **kwargs):
        """ Compute a histogram of the model, optionally show the histogram for given depth ranges instead """
        # Get the depth grid
        self.getZGrid(force=force)

        if (depth1 is None):
            depth1 = self.zGrid[0]
        if (depth2 is None):
            depth2 = self.zGrid[-1]

        # Ensure order in depth values
        if (depth1 > depth2):
            tmp=depth2
            depth2 = depth1
            depth1 = tmp

        # Don't need to check for depth being shallower than zGrid[0] since the sortedsearch with return 0
        if (depth1 > self.zGrid[-1]): Err.Emsg('mapDepthSlice: Depth is greater than max depth - '+str(self.zGrid[-1]))
        if (depth2 > self.zGrid[-1]): Err.Emsg('mapDepthSlice: Depth2 is greater than max depth - '+str(self.zGrid[-1]))

        if (bestModel):
            if (withDoi):
                self.getAttribute(best=True, doi=True, percent=percent)
            else:
                self.getAttribute(best=True, force=force)
            model = self.best
        else:
            if (withDoi):
                self.getAttribute(mean=True, doi=True, percent=percent)
            else:
                self.getAttribute(mean=True, force=force)
            model = self.mean

        if withDoi:
            depth1 = np.minimum(self.doi, depth1)
            depth2 = np.minimum(self.doi, depth2)
            z = np.repeat(self.zGrid[:,np.newaxis],self.nPoints,axis=1)
            vals = model[(z > depth1)&(z < depth2)]
        else:
            cell1 = self.zGrid.searchsorted(depth1)
            cell2 = self.zGrid.searchsorted(depth2)
            vals = model[cell1:cell2+1,:]

        log = kwargs.pop('log',False)

        if (invertPar):
            vals = 1.0/vals
            name = 'Resistivity'
            units = '$\Omega m$'
        else:
            name = 'Conductivity'
            units = '$Sm^{-1}$'

        if (log):
            vals,logLabel=cP._logSomething(vals,log)
            name = logLabel+name
        vals = StatArray(vals, name, units)

        h = Histogram1D(np.linspace(vals.min(),vals.max(),nBins))
        h.update(vals)
        h.plot(**kwargs)
        return h


    def getZGrid(self, force=False):
        """ Gets the discretization in depth """
        if (not self.zGrid is None and not force): return
        # Get the mean model for the first point to get the axes
        self.lines[0].getZgrid()
        self.zGrid = self.lines[0].zGrid


    def getAttribute(self, xy=False, elevation=False, mean=False, best=False, opacity=False, doi=False, relErr=False, addErr=False, percent=67.0, force=False):
        """ Get a subsurface property """

        assert (not all([not xy, not elevation, not mean, not best, not opacity, not doi, not relErr, not addErr])), 'Please choose at least one attribute' + help(self.getAttrubute)

        # Turn off attributes that are already loaded
        if (xy):
            xy=self.points is None
        if (elevation):
            elevation=self.elevation is None
        if (mean):
            mean=self.mean is None
        if (best):
            best=self.best is None
        if (relErr):
            relErr=self.relErr is None
        if (addErr):
            addErr=self.addErr is None
        # Getting the doi is cheap, so always ask for it even if opacity is requested
        doi = opacity or doi
        if (doi):
            doi=self.doi is None

        if (all([not xy, not elevation, not mean, not best, not opacity, not doi, not relErr, not addErr])):
            return

        # Get the number of data points
        self.getNpoints(force=force)

        # Get the number of systems
        if (relErr or addErr):
            self.getNsys()

        if (mean or best or doi):
            # Get the depth grid
            self.getZGrid(force=force)
            # Get the number of cells
            nz = self.zGrid.size

        # Initialize attributes
        if (xy):
            self.points = PointCloud3D(self.nPoints)
        if (elevation):
            self.elevation = StatArray(self.nPoints,name='Elevation',units='m')
        if (mean):
            self.lines[0].getMeanParameters()
            self.mean = StatArray([nz,self.nPoints], name=self.lines[0].mean.name, units=self.lines[0].mean.units, order = 'F')
        if (best):
            self.lines[0].getBestParameters()
            self.best = StatArray([nz,self.nPoints], name=self.lines[0].best.name, units=self.lines[0].best.units, order = 'F')
        if (doi):
            self.opacity=np.zeros([nz,self.nPoints], order = 'F')
            self.doi = StatArray(np.zeros(self.nPoints),'Depth of Investigation','m')
        if (relErr):
            self.lines[0].getRelativeError()
            if (self.nSys > 1):
                self.relErr = StatArray([self.nPoints, self.nSys],name=self.lines[0].relErr.name,units=self.lines[0].relErr.units, order = 'F')
            else:
                self.relErr = StatArray(self.nPoints,name=self.lines[0].relErr.name,units=self.lines[0].relErr.units, order = 'F')
        if (addErr):
            self.lines[0].getAdditiveError()
            if (self.nSys > 1):
                self.addErr = StatArray([self.nPoints, self.nSys],name=self.lines[0].addErr.name,units=self.lines[0].addErr.units, order = 'F')
            else:
                self.addErr = StatArray(self.nPoints,name=self.lines[0].addErr.name,units=self.lines[0].addErr.units, order = 'F')

        # Loop over the lines in the data set and get the attributes
        i0 = 0
        print('Reading attributes from dataset results')
        Bar=progressbar.ProgressBar()
        for i in Bar(range(self.nLines)):
            i1 = i0 + self.lines[i].nPoints

            # Perform line getters
            if (xy):
                self.lines[i].getX()
                self.lines[i].getY()
                self.points.x[i0:i1] = self.lines[i].x
                self.points.y[i0:i1] = self.lines[i].y
                self.lines[i].x=None
                self.lines[i].y=None
            if (elevation):
                self.lines[i].getElevation()
                self.elevation[i0:i1] = self.lines[i].elevation
                self.lines[i].elevation = None
            if (mean):
                self.lines[i].getMeanParameters()
                self.mean[:,i0:i1] = self.lines[i].mean.T
                self.lines[i].mean = None # Free memory
            if (best):
                self.lines[i].getBestParameters()
                self.best[:,i0:i1] = self.lines[i].best.T
                self.lines[i].best = None # Free memory
            if (doi):
                # Get the DOI for this line
                self.lines[i].getDOI(percent)
                self.opacity[:,i0:i1] = self.lines[i].opacity.T
                self.doi[i0:i1] = self.lines[i].doi
                self.lines[i].opacity = None # Free memory
                self.lines[i].doi = None # Free memory
            if (relErr):
                self.lines[i].getRelativeError()
                if (self.nSys > 1):
                    self.relErr[i0:i1,:] = self.lines[i].relErr
                else:
                    self.relErr[i0:i1] = self.lines[i].relErr
                self.lines[i].relErr = None # Free memory
            if (addErr):
                self.lines[i].getAdditiveError()
                if (self.nSys > 1):
                    self.addErr[i0:i1,:] = self.lines[i].addErr
                else:
                    self.addErr[i0:i1] = self.lines[i].addErr
                self.lines[i].addErr = None # Free memory

            # Deallocate line attributes to save space
            self.lines[i].hitMap = None

            i0 = i1
        if (xy):
            self.points.getBounds() # Get the bounding box

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
            
           
        self.getAttribute(xy=True, mean=True, force=force)
        
        method = method.lower()
        if method == 'ct':
            self.__getMean3D_CloughTocher(dx=dx, dy=dy, mask=mask, clip=clip, force=force)
        elif method == 'mc':
            self.__getMean3D_minimumCurvature(dx=dx, dy=dy, mask=mask, clip=clip, force=force)
        else:
            assert False, ValueError("method must be either 'ct' or 'mc' ")
            
        print('TEST: ',np.min(self.mean3D), np.max(self.mean3D))
            
        with h5py.File('mean3D.h5','w') as f:
            f.create_dataset(name = 'dx', data = dx)
            f.create_dataset(name = 'dy', data = dy)
            f.create_dataset(name = 'mask', data = mask)
            f.create_dataset(name = 'clip', data = clip)
            f.create_dataset(name = 'method', data = method)
            self.mean3D.toHdf(f,'mean3d')
            
        
    
    def __getMean3D_minimumCurvature(self, dx, dy, mask=None, clip=False, force=False):
               
        
        # Get the points to interpolate to
        x,y,intPoints = interpolation.getGridLocations2D(self.points.bounds, dx, dy)
        
        # Initialize 3D volume
        mean3D = StatArray(np.zeros([self.zGrid.size, y.size, x.size], order = 'F'),name = 'Conductivity', units = '$Sm^{-1}$')
        
        # Interpolate for each depth
        print('Interpolating using minimum curvature')
        Bar=progressbar.ProgressBar()
        for i in Bar(range(self.zGrid.size)):
            # Get the model values for the current depth
            values = self.mean[i,:]
            x,y,vals = interpolation.minimumCurvature(self.points.x, self.points.y, values, self.points.bounds, dx=dx, dy=dy, mask=mask, clip=clip, iterations=2000, tension=0.25, accuracy=0.01)
            # Add values to the 3D array
            mean3D[i,:,:] = vals
                  
        self.mean3D = mean3D 
    
    def __getMean3D_CloughTocher(self, dx, dy, mask=None, clip=False, force=False):
        
        # Get the discretization
        if (dx is None):
            tmp = self.points.bounds[1]-self.points.bounds[0]
            dx = 0.01 * tmp
        assert dx > 0.0, "dx must be positive!"
        
        # Get the discretization
        if (dy is None):
            tmp = self.points.bounds[3]-self.points.bounds[2]
            dy = 0.01 * tmp
        assert dy > 0.0, "dy must be positive!"
        
        tmp = np.column_stack((self.points.x, self.points.y))

        # Get the points to interpolate to
        x,y,intPoints = interpolation.getGridLocations2D(self.points.bounds, dx, dy)

        # Create a distance mask
        if mask:
            self.points.setKdTree(nDims=2) # Set the KdTree on the data points
            g = np.meshgrid(x,y)
            xi = _ndim_coords_from_arrays(tuple(g), ndim=tmp.shape[1])
            dists, indexes = self.points.kdtree.query(xi)
            iMask = np.where(dists > mask)

        # Get the value bounds
        minV = np.nanmin(self.mean)
        maxV = np.nanmax(self.mean)

        # Initialize 3D volume
        mean3D = StatArray(np.zeros([self.zGrid.size, y.size, x.size], order = 'F'),name = 'Conductivity', units = '$Sm^{-1}$')

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


    def mapAdditiveError(self,dx, dy, system=0, mask = None, clip = True, force=False, **kwargs):
        """ Create a map of a parameter """
        self.getAttribute(xy=True, addErr=True, force=force)        
        if (self.nSys > 1):
            self.points.mapPlot(dx = dx, dy = dy, mask = mask, clip = clip, c = self.addErr[:,system], **kwargs)
        else:
            self.points.mapPlot(dx = dx, dy = dy, mask = mask, clip = clip, c = self.addErr, **kwargs)
        cP.xlabel('Easting (m)')
        cP.ylabel('Northing (m)')


    def mapRelativeError(self,dx, dy, system=0, mask = None, clip = True, force=False, **kwargs):
        """ Create a map of a parameter """
        self.getAttribute(xy=True, relErr=True, force=force)
        if (self.nSys > 1):
            self.points.mapPlot(dx = dx, dy = dy, mask = mask, clip = clip, c = self.relErr[:,system], **kwargs)
        else:
            self.points.mapPlot(dx = dx, dy = dy, mask = mask, clip = clip, c = self.relErr, **kwargs)
        cP.xlabel('Easting (m)')
        cP.ylabel('Northing (m)')


    def getDepthSlice(self, depth, depth2=None, invertPar=True, bestModel=False, force=False):
        # Get the depth grid
        self.getZGrid(force=force)

        assert depth <= self.zGrid[-1], 'Depth is greater than max depth '+str(self.zGrid[-1])
        if (not depth2 is None):
            assert depth2 <= self.zGrid[-1], 'Depth2 is greater than max depth '+str(self.zGrid[-1])
            assert depth <= depth2, 'Depth2 must be >= depth'

        if (bestModel):
            self.getAttribute(xy=True, best=True, force=force)
            model = self.best
        else:
            self.getAttribute(xy=True, mean=True, force=force)
            model = self.mean
            
        model[model == 0.0] = 1.0
        print(model.shape)
        print('Model: ',model.min(), model.max())

        cell1 = self.zGrid.searchsorted(depth)

        if (not depth2 is None):
            cell2 = self.zGrid.searchsorted(depth2)
            vals1D = np.mean(model[cell1:cell2+1,:],axis = 0)
        else:
            vals1D = model[cell1,:]
            
        if (invertPar):
            vals1D = StatArray(1.0/vals1D,name = 'Resistivity', units = '$\Omega m$')
        else:
            vals1D = StatArray(vals1D,name = 'Conductivity', units = '$Sm^{-1}$')
        return vals1D


    def mapDepthSlice(self, dx, dy, depth, depth2 = None, invertPar = True, bestModel = False, mask = None, clip = True, force=False, **kwargs):
        """ Create a depth slice through the recovered model """

        vals1D = self.getDepthSlice(depth=depth, depth2=depth2, invertPar=invertPar, bestModel=bestModel, force=force)
        
        print(vals1D.min(), vals1D.max())
        
        ax = self.points.mapPlot(dx = dx, dy = dy, mask = mask, clip = clip, c = vals1D, **kwargs)
        cP.xlabel('Easting (m)')
        cP.ylabel('Northing (m)')
        
        return ax


    def mapElevation(self,dx, dy, mask = None, clip = True, extrapolate=None, force=False, **kwargs):
        """ Create a map of a parameter """
        self.getAttribute(xy=True, elevation=True, force=force)
        self.points.mapPlot(dx = dx, dy = dy, mask = mask, clip = clip, extrapolate=extrapolate, c = self.elevation, **kwargs)
        cP.xlabel('Easting (m)')
        cP.ylabel('Northing (m)')

    def mapDoi(self, dx, dy, mask = None, clip = True, extrapolate=None, force=False, **kwargs):
        """ Create a map of a parameter """
        self.getAttribute(xy=True, doi=True, force=force)
        self.points.mapPlot(dx = dx, dy = dy, mask = mask, clip = clip, extrapolate=extrapolate, c = self.doi, **kwargs)
        cP.xlabel('Easting (m)')
        cP.ylabel('Northing (m)')


    def plotDepthSlice(self, depth, depth2 = None, invertPar = True, bestModel = False, mask = None, clip = True, force=False, *args, **kwargs):
        """ Create a depth slice through the recovered model """

        vals1D = self.getDepthSlice(depth=depth, depth2=depth2, invertPar=invertPar, bestModel=bestModel, force=force)

        ax = self.points.scatter2D(c = vals1D, *args, **kwargs)
        cP.xlabel('Easting (m)')
        cP.ylabel('Northing (m)')
        
        return ax

    def plotElevation(self, force=False,**kwargs):
        """ Plot the observation locations """
        self.getAttribute(xy=True, elevation=True, force=force)

        if (not 'edgecolor' in kwargs):
            kwargs['edgecolor']='k'
        if (not 's' in kwargs):
            kwargs['s']=10.0

        self.points.scatter2D(c = self.elevation, **kwargs)
        cP.xlabel('Easting (m)')
        cP.ylabel('Northing (m)')


    def plotXplot(self, bestModel=True, withDoi=True, invertPar=True, log10=True, **kwargs):
        """ Plot the cross plot of a model against depth """

        tmp = self.getParVsZ(bestModel=bestModel, withDoi=withDoi, invertPar=invertPar, log10=log10)
        # Repeat the depths for plotting
        cP.plot(tmp[:,0], tmp[:,1], **kwargs)
        if (bestModel):
            cP.xlabel(self.best.getNameUnits())
        else:
            cP.xlabel(self.mean.getNameUnits())
        cP.ylabel(self.zGrid.getNameUnits())
        return tmp
    
    def plotXsection(self, line, xAxis='easting', invertPar = True, bestModel=False, percent = 67.0, useVariance=True, **kwargs):
        """ Plot a x section for the ith line """
        self.lines[line].setAlonglineAxis(xAxis)
        self.lines[line].plotXsection(invertPar=invertPar, bestModel=bestModel, percent=percent, useVariance=useVariance, **kwargs)

    def getParVsZ(self, bestModel=False, withDoi=True, invertPar=True, log10=True, clipNan=True):
        """ Get the depth and parameters, optionally within the doi """
        self.getNpoints()
        # Get the depths
        self.getZGrid()
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

        if invertPar:
            model = 1.0/model.reshape(model.size, order='F')

        if log10:
            model = np.log10(model)

        res = StatArray(np.column_stack((model,z)))

        if (clipNan):
            res = res[np.logical_not(np.isnan(res[:,0]))]

        return res

    def kMeans(self, nClusters, precomputedParVsZ=None, standardize=False, log10Depth=False, plot=False, bestModel=True, withDoi=True, invertPar=True, log10=True, clipNan=True, **kwargs):
        """  """
        if (precomputedParVsZ is None):
            ParVsZ = self.getParVsZ(bestModel=bestModel, withDoi=withDoi, invertPar=invertPar, log10=log10, clipNan=clipNan)
        else:
            ParVsZ = precomputedParVsZ

        assert isinstance(ParVsZ, StatArray), "precomputedParVsZ must be an StatArray"

        if (log10Depth):
            ParVsZ[:,1] = np.log10(ParVsZ[:,1])

        return ParVsZ.kMeans(nClusters, standardize=standardize, nIterations=10, plot=plot, **kwargs)


    def GMM(self, ParVsZ, clusterID, trainPercent=90.0, plot=True):
        """ Classify the subsurface parameters """
        assert isinstance(ParVsZ, StatArray), "ParVsZ must be an StatArray"
        ParVsZ.GMM(clusterID, trainPercent=trainPercent, covType=['spherical','tied','diag','full'], plot=plot)



    def toVTK(self, fName, dx, dy, mask=False, clip=False, force=False, method='ct'):
        """ Convert a 3D volume of interpolated values to vtk for visualization in Paraview """

        print('toVTK')

        self.getAttribute(xy=True, elevation=True, force=force)
        
        self.getMean3D(dx=dx, dy=dy, mask=mask, clip=clip, force=force, method=method)
        self.getZGrid()
        self.points.getBounds()

        x,y,intPoints = interpolation.getGridLocations2D(self.points.bounds, dx, dy)
        z=self.zGrid


        from pyvtk import VtkData, UnstructuredGrid, PointData, CellData, Scalars

        # Get the 3D dimensions
        mx = x.size
        my = y.size
        mz = z.size
        
        nPoints = mx*my*mz
        nCells = (mx-1)*(my-1)*(mz-1)

        # Interpolate the elevation to the grid nodes
        if (method == 'ct'):
            tx,ty, vals = self.points.interpCloughTocher(self.elevation, dx = dx,dy=dy, mask = mask, clip = clip, extrapolate='nearest')
        elif (method == 'mc'):
            tx,ty, vals = self.points.interpMinimumCurvature(self.elevation, dx = dx, dy=dy, mask = mask, clip = clip)
            
        vals = vals[:my,:mx]
        vals = vals.reshape(mx*my)

        # Set up the nodes and voxel indices
        points = np.zeros([nPoints,3], order='F')
        points[:,0] = np.tile(x,my*mz)
        points[:,1] = np.tile(y.repeat(mx),mz)
        points[:,2] = np.tile(vals,mz)-z.repeat(mx*my)

        # Create the cell indices into the points
        p = np.arange(nPoints).reshape((mz,my,mx))
        voxels = np.zeros([nCells,8],dtype=np.int)
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
        pointID = Scalars(np.arange(nPoints),name='Point iD')
        pointElev = Scalars(points[:,2],name='Point Elevation (m)')

        tmp=self.mean3D.reshape(np.size(self.mean3D))
        tmp1 = np.log10(1.0/tmp)
        pointRes = Scalars(tmp1, name = 'log10(Resistivity) (Ohm m)')
        tmp1 = np.log10(tmp)

        pointCon = Scalars(tmp1, name = 'log10(Conductivity) (S/m)')
        
        PData = PointData(pointID, pointElev, pointRes, pointCon)
        CData = CellData(Scalars(np.arange(nCells),name='Cell iD'))
        vtk = VtkData(
              UnstructuredGrid(points,
                               hexahedron=voxels),
#                               ),
              PData,
              CData,
              'Some Name'
              )

        vtk.tofile(fName, 'ascii')










































if __name__ == "__main__":

    d="/Users/nfoks/Projects/bMinsley/Tdem/SLV/ScatterV"
    ds = DataSetResults(d)
