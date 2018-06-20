""" @PointCloud3D
Module describing a 3D point cloud
"""
from ...classes.core.myObject import myObject
import numpy as np
from ...classes.core.StatArray import StatArray
from ...base import fileIO as fIO
from ...base import interpolation
from ...base import customFunctions as cf
from ...base import customPlots as cP
from .Point import Point
from ...base import MPI
from scipy.spatial import cKDTree


class PointCloud3D(myObject):
    """ 3D Point Cloud with x,y,z co-ordinates """

    def __init__(self, N=0, x=None, y=None, z=None, units="m"):
        """ Initialize the class """




        # Number of points in the cloud
        self.N = N
        # StatArray of the x co-ordinates
        self.x = StatArray(self.N, "Easting", units)
        # StatArray of the y co-ordinates
        self.y = StatArray(self.N, "Northing", units)
        # StatArray of the z co-ordinates
        self.z = StatArray(self.N, "Height", units)
        # KdTree
        self.kdtree = None
        # Bounding Box
        self.bounds = None

    def __getitem__(self, i):
        """ Define item getter """
        n = np.size(i)
        tmp = PointCloud3D(n, self.x.units)
        tmp.x[:] = self.x[i]
        tmp.y[:] = self.y[i]
        tmp.z[:] = self.z[i]
        return tmp

    def getBounds(self):
        """ Gets the bounding box of the data set """
        self.bounds = np.zeros(4)
        self.bounds[:]=(np.min(self.x),np.max(self.x),np.min(self.y),np.max(self.y))

    def getPoint(self, i):
        """ Get a point from the 3D Point Cloud
        Returns a point P """
        P = Point(self.x[i], self.y[i], self.z[i])
        return P

    def summary(self, out=False):
        """ Display a summary of the 3D Point Cloud """
        msg = "3D Point Cloud: \n"
        msg += "Number of Points: :" + str(self.N) + '\n'
        msg += self.x.summary(True)
        msg += self.y.summary(True)
        msg += self.z.summary(True)
        if (out):
            return msg
        print(msg)

    def maketest(self, nPoints):
        """ Creates a small test of random points """
        PointCloud3D.__init__(self, nPoints)
        self.x[:] = (2.0 * np.random.rand(nPoints)) - 1.0
        self.y[:] = (2.0 * np.random.rand(nPoints)) - 1.0
        self.z[:] = cf.cosSin1(self.x, self.y, 1.0, 1.0)

    def read(self, fname, nHeaders=0, cols=range(3)):
        """ Reads x y z co-ordinates from an ascii file
        if cols is not given, the first three columns are read.
        Otherwise specify with cols=[2,3,4] """
        assert np.size(cols) == 3, 'size(cols) must equal 3'
        nLines = fIO.getNlines(fname, nHeaders)
        # Initialize the Data
        self.__init__(nLines)
        # Read each line assign the values to the class
        tmp = fIO.getHeaderNames(fname, cols)
        with open(fname) as f:
            self.x.name = tmp[0]
            self.y.name = tmp[1]
            self.z.name = tmp[2]
            fIO.skipLines(f, nHeaders)  # Skip header lines
            for j, line in enumerate(f):  # For each line in the file
                values = fIO.getRealNumbersfromLine(line, cols)  # grab the requested entries
                # Assign values into object
                self.x[j] = values[0]
                self.y[j] = values[1]
                self.z[j] = values[2]

    def scatter2D(self, **kwargs):
        """ Plots the x y locations of the point cloud with the height for colour"""
        if (not 'linewidth' in kwargs):
            kwargs['linewidth'] = 0.1
        if (not 'c' in kwargs):
            kwargs['c'] = self.z
        ax = cP.scatter2D(self.x, y=self.y, **kwargs)
        return ax


    def setKdTree(self, nDims=3):
        """ Creates the kdtree mapping the point locations """
        if (nDims == 2):
            tmp = np.column_stack((self.x, self.y))
        elif (nDims == 3):
            tmp = np.column_stack((self.x, self.y, self.z))
        self.kdtree = cKDTree(tmp)


    def nearest(self, x, k=1, eps=0, p=2, radius=np.inf):

        assert (not self.kdtree is None), TypeError('kdtree has not been set, use self.setKdTree()')
        return self.kdtree.query(x, k, eps, p, distance_upper_bound=radius)


    def mapPlot(self, dx=None, dy=None, extrapolate=None, i=None, **kwargs):
        """ Create a map of a parameter """

        cTmp = kwargs.pop('c',self.z)
        
        mask = kwargs.pop('mask',False)
        
        clip = kwargs.pop('clip',True)
        
        method = kwargs.pop('method', "ct")
        method = method.lower()
              
        
        if method == 'ct':
            x,y,vals = self.interpCloughTocher(cTmp, dx=dx, dy=dy, mask=mask, clip=clip, extrapolate=extrapolate, i=i)
        elif method == 'mc':
            x,y,vals = self.interpMinimumCurvature(cTmp, dx, dy, mask=mask, clip=clip, i=i)
        else:
            assert False, ValueError("method must be either 'ct' or 'mc' ")  
                        
        ax = cP.pcolor(vals,x,y, **kwargs)
        cP.xlabel(cf.getNameUnits(self.x))
        cP.ylabel(cf.getNameUnits(self.y))
        return ax


#    def interpRBF(self, values, nDims, dx = None, function = None, epsilon = None, smooth = None, norm = None, **kwargs):
#        """ Interpolate values to a grid using radial basis functions """
#        # Get the bounding box
#        self.getBounds()
#        print(self.bounds)
#        # Get the discretization
#        if (dx is None):
#            tmp = np.min((self.bounds[1]-self.bounds[0], self.bounds[3]-self.bounds[2]))
#            dx = 0.01 * tmp
#        assert dx > 0.0, ValueError("Interpolation cell size must be positive!")
#
#        if (nDims == 2):
#            z = np.ones(self.N)
#        elif (nDims == 3):
#            z = self.z
#
#        x,y,vals = interpolation.RBF(dx, self.bounds, self.x, self.y, z, values)
#        return x, y, vals

    def interpCloughTocher(self, values, dx = None, dy=None, mask = False, clip = None, extrapolate=None, i=None):
        """ Interpolate values at the points to a grid """

        # Get the bounding box
        self.getBounds()
        
        # Get the discretization
        if (dx is None):
            tmp = self.bounds[1]-self.bounds[0]
            dx = 0.01 * tmp
        assert dx > 0.0, ValueError("dx must be positive!")
        
        # Get the discretization
        if (dy is None):
            tmp = self.bounds[3]-self.bounds[2]
            dy = 0.01 * tmp
        assert dy > 0.0, ValueError("dy must be positive!")

        kdtree = None
        if (not i is None):
            xTmp = self.x[i]
            yTmp = self.y[i]
            vTmp = values[i]
            tmp = np.column_stack((xTmp, yTmp))
            if (mask or extrapolate):
                kdtree = cKDTree(tmp)
        else:
            tmp = np.column_stack((self.x, self.y))
            vTmp = values
            if (mask or extrapolate):
                self.setKdTree(nDims = 2)
                kdtree = self.kdtree

        xc,yc,vals = interpolation.CT(dx, dy, self.bounds, tmp, vTmp , mask = mask, kdtree = kdtree, clip = clip, extrapolate=extrapolate)

        x = np.linspace(self.bounds[0], self.bounds[1], xc.size+1)
        y = np.linspace(self.bounds[2], self.bounds[3], yc.size+1)

        return x, y, vals
    
    
    def interpMinimumCurvature(self, values, dx=None, dy=None, mask=False, clip=None, i=None):
    
        # Get the bounding box
        self.getBounds()
        
        # Get the discretization
        if (dx is None):
            tmp = self.bounds[1]-self.bounds[0]
            dx = 0.01 * tmp
        assert dx > 0.0, ValueError("dx must be positive!")
        
        # Get the discretization
        if (dy is None):
            tmp = self.bounds[3]-self.bounds[2]
            dy = 0.01 * tmp
        assert dy > 0.0, ValueError("dy must be positive!")
                
        x,y,vals = interpolation.minimumCurvature(self.x, self.y, values, self.bounds, dx, dy, mask=mask, iterations=2000, tension=0.25, accuracy=0.01)
        return x, y, vals        


    def Bcast(self, world):
        """ Braodcase a PointCloud3D using MPI """
        N = MPI.Bcast(self.N, world)
        this = PointCloud3D(N)
        this.x = self.x.Bcast(world)
        this.y = self.y.Bcast(world)
        this.z = self.z.Bcast(world)
        return this

    def Scatterv(self, myStart, myChunk, world):
        """ ScatterV a PointCloud3D using MPI """
        #print(world.rank," PointCloud3D.Scatterv")
        N = myChunk[world.rank]
        #print(world.rank," mySize "+str(N))
        this = PointCloud3D(N)
        #print(world.rank,' PointCloud3D initialized')
        this.x = self.x.Scatterv(myStart, myChunk, world)
        this.y = self.y.Scatterv(myStart, myChunk, world)
        this.z = self.z.Scatterv(myStart, myChunk, world)
        return this

if __name__ == "__main__":
    P = PointCloud3D()
    P.read('EMdata.csv', 1, [3, 2, 4])
