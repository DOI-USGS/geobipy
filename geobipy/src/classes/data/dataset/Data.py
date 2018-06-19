""" @Data_Class
Module describing a Data Set where values are associated with an xyz co-ordinate
"""
import numpy as np
from ....classes.core.StatArray import StatArray
from ....base import fileIO as fIO
from ....base import customFunctions as cf
from ....base import customPlots as cP
from ...pointcloud.PointCloud3D import PointCloud3D
from ....classes.core.myObject import myObject
from ....base import MPI as myMPI
import matplotlib.pyplot as plt


class DataSet(myObject):

    def __init__(self, nPoints=1, nChannels=1, units=None):
        """ Initialize the DataSet """
        # StatArray of data
        self.D = StatArray([nPoints, nChannels], "Data", units, order='F')
        # StatArray of Standard Deviations
        self.Std = StatArray(np.ones([nPoints, nChannels]), "Standard Deviation", units, order='F')
        self.nPoints = nPoints
        self.nChannels = nChannels

    def Bcast(self, world):
        """ Broadcast the DataSet using MPI """
        nPoints = myMPI.Bcast(self.nPoints, world)
        nChannels = myMPI.Bcast(self.nChannels, world)
        this = DataSet(nPoints, nChannels)
        this.D = self.D.Bcast(world)
        this.Std = self.Std.Bcast(world)
        return this

    def Scatterv(self, myStart, myChunk, world):
        """ Scatterv the DataSet using MPI """
        nPoints = myChunk[world.rank]
        nChannels = myMPI.Bcast(self.nChannels, world)
        this = DataSet(nPoints, nChannels)
        this.D = self.D.Scatterv(myStart, myChunk, world)
        this.Std = self.Std.Scatterv(myStart, myChunk, world)
        return this


class Data(PointCloud3D):
    """ Class defining a set of Data """

    def __init__(self, nPoints=1, nChannels=1, units=None):
        """ Initialize the Data class """
        # Number of Channels
        self.nChannels = np.int64(nChannels)
        PointCloud3D.__init__(self, nPoints)
        # StatArray of data
        self.D = StatArray([self.N, self.nChannels], "Data", units)
        # StatArray of Standard Deviations
        self.Std = StatArray([self.N, self.nChannels], "Standard Deviation", units)
        self.names = ['Channel '+str(i) for i in range(nChannels)]

    def maketest(self, nPoints, nChannels):
        """ Create a test example """
        Data.__init__(self, nPoints, nChannels)   # Initialize the Data array
        # Use the PointCloud3D example creator
        PointCloud3D.maketest(self, nPoints)
        a = 1.0
        b = 2.0
        # Create different Rosenbrock functions as the test data
        for i in range(nChannels):
            tmp = cf.rosenbrock(self.x, self.y, a, b)
            # Put the tmp array into the data column
            self.D[:, i] = tmp[:]
            b *= 2.0

    def read(self, fname, cols, nHeaders=0, nChannels=0):
        """ Read the specified columns from an ascii file
        cols[0,1,2,...] should be the indices of the x,y,z co-ordinates """
        nCols = len(cols)
        #if any([cols < 0]): err.Emsg("Please specify the columns to read the first three indices should be xyz")
        # Get the number of points
        nLines = fIO.getNlines(fname, nHeaders)
        # Get the number of Data if none was specified
        if (nChannels == 0):
            nChannels = nCols - 3
        # Initialize the Data
        Data.__init__(self, nLines, nChannels)
        # Get the names of the headers
        names = fIO.getHeaderNames(fname, cols)
        print(names)
        self.x.name=names[0]
        self.y.name=names[1]
        self.z.name=names[2]
        self.names=names[3:]
        # Read each line assign the values to the class
        with open(fname) as f:
            fIO.skipLines(f, nHeaders)  # Skip header lines
            for j, line in enumerate(f):  # For each line in the file
                values = fIO.getRealNumbersfromLine(line, cols)  # grab the requested entries
                # Assign values into object
                self.x[j] = values[0]
                self.y[j] = values[1]
                self.z[j] = values[2]
                self.D[j, ] = values[3:]

    def getChannel(self, channel):
        """ Gets the data in the specified channel """
        assert channel >= 0 and channel < self.nChannels, 'Requested channel must be less than '+str(self.nChannels)

        tmp = StatArray(self.D[:, channel], self.names[channel])

        return tmp

    def __getitem__(self, i):
        """ Define item getter for Data """
        tmp = Data(np.size(i), self.nChannels, self.D.units)
        tmp.x[:] = self.x[i]
        tmp.y[:] = self.y[i]
        tmp.z[:] = self.z[i]
        tmp.D[:, :] = self.D[i, :]
        tmp.Std[:, :] = self.Std[i, :]
        tmp.names[:] = self.names[i]
        return tmp

    def getLine(self, line):
        """ Get the data from the given line number """
        i = np.where(self.line == line)[0]
        return self[i]

    def summary(self):
        """ Display a summary of the Data """
        PointCloud3D.summary(self)
        print("Data:          :")
        print("# of Channels: :" + str(self.nChannels))
        print("# of Total Data:" + str(self.N * self.nChannels))
        print("Channel Names:  ", self.names)
        print('')

    def plot(self, channels=None, *args, **kwargs):
        """ Plots the specifed columns as a line plot, if cols is not given, all the columns are plotted """
        x = StatArray(np.arange(0, len(self.D), 1), name="index")

        ax = plt.gca()
        if channels is None:
            nCols = np.size(self.D, 1)
            for i in range(nCols):
                cP.plot(x, self.D[:,i],label=self.names[i],*args, **kwargs)
                #plt.plot(x,self.D[:,i],label=self.names[i],*args, **kwargs)
        else:
            for j, i in enumerate(channels):
                cP.plot(x,self.D[:,j],label=self.names[i],*args, **kwargs)
    
        plt.title("Data")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        leg=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
        leg.set_title(self.D.getNameUnits())

        plt.xlabel(cf.getNameUnits(x))



    def mapChannel(self, channel, *args, **kwargs):
        """ Create a map of the specified data channel """

        assert channel >= 0 and channel < self.nChannels, ValueError('Requested channel must be less than '+str(self.nChannels))

        kwargs['c'] = self.getChannel(channel)

        self.mapPlot(*args, **kwargs)

        cP.title(self.names[channel])

    def Bcast(self, world):
        """ Broadcast a Data object using MPI """
        pc3d = None
        pc3d = PointCloud3D.Bcast(self, world)
#      print(world.rank,' After PointCloud3D.Scatterv')
        nChannels = myMPI.Bcast(self.nChannels, world)
        this = Data(pc3d.N, nChannels)
        this.x = pc3d.x
        this.y = pc3d.y
        this.z = pc3d.z
        this.D = self.D.Bcast(world)
        this.Std = self.Std.Bcast(world)
        return this

    def Scatterv(self, myStart, myChunk, world):
        """ Scatterv a Data object using MPI """
#      print(world.rank,' Data.Scatterv')
        pc3d = None
        pc3d = PointCloud3D.Scatterv(self, myStart, myChunk, world)
#      print(world.rank,' After PointCloud3D.Scatterv')
        nChannels = myMPI.Bcast(self.nChannels, world)
        this = Data(pc3d.N, nChannels)
        this.x = pc3d.x
        this.y = pc3d.y
        this.z = pc3d.z
        this.D = self.D.Scatterv(myStart, myChunk, world)
        this.Std = self.Std.Scatterv(myStart, myChunk, world)
        return this
