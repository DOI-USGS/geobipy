""" @TopoRectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified.  The upper surface of the mesh can be draped.
"""
from copy import deepcopy
from ...classes.core import StatArray
from .RectilinearMesh1D import RectilinearMesh1D
from .RectilinearMesh2D import RectilinearMesh2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from ...base import customPlots as cP
from ...base import fileIO as fio

try:
    from pyvtk import VtkData, CellData, Scalars, PolyData
except:
    pass


class TopoRectilinearMesh2D(RectilinearMesh2D):
    """Class defining a 2D rectilinear mesh with cell centres and edges and an upper surface that is draped along
    an elevation profile.

    Contains a simple 2D mesh with cell edges, widths, and centre locations.
    In the instantiation, there are three cartesian co-ordinates, x, y, z, specified for the cell centres or edges.
    This allows you to, for example, create a vertical 2D mesh that is not parallel to either the
    x or y axis, like a collection line of data.
    The mesh lies in the x-z plane.  The y axis can be used to compute the distance sqrt(x^2 + y^2),
    which can then be used for plotting.
    A height profile can also be given that specifies the height of the upper most cell.
    The cell locations in z are then added to the height profile.

    TopoRectilinearMesh2D(xCentres, xEdges, yCentres, yEdges, zCentres, zEdges, heightCentres, heightEdges)

    Parameters
    ----------
    xCentres : geobipy.StatArray, optional
        The locations of the centre of each cell in the "x" direction. Only xCentres or xEdges can be given, not both.
    xEdges : geobipy.StatArray, optional
        The locations of the edges of each cell, including the outermost edges, in the "x" direction. Only xCentres or xEdges can be given, not both.
    yCentres : geobipy.StatArray, optional
        The locations of the centre of each cell in the "y" direction. Only yCentres or yEdges can be given, not both.
    yEdges : geobipy.StatArray, optional
        The locations of the edges of each cell, including the outermost edges, in the "y" direction. Only yCentres or yEdges can be given, not both.
    zCentres : geobipy.StatArray, optional
        The locations of the centre of each cell in the "z" direction. Only zCentres or zEdges can be given, not both.
    zEdges : geobipy.StatArray, optional
        The locations of the edges of each cell, including the outermost edges, in the "z" direction. Only zCentres or zEdges can be given, not both.
    heightCentres : geobipy.StatArray, optional
        The height of each point at the x, y locations. Only heightCentres or heightEdges can be given, not both.
    heightEdges : geobipy.StatArray, optional
        The height of each point at the x, y locations of the edges of each cell, including the outermost edges. Only heightCentres or heightEdges can be given, not both.

    Returns
    -------
    out : TopoRectilinearMesh2D
        The 2D mesh.

    """


    def __init__(self, xCentres=None, xEdges=None, yCentres=None, yEdges=None, zCentres=None, zEdges=None, heightCentres=None, heightEdges=None):
        """ Initialize a 2D Rectilinear Mesh"""

        if (all(x is None for x in [xCentres, xEdges, yCentres, yEdges, zCentres, zEdges, heightCentres, heightEdges])):
            return

        super().__init__(xCentres, xEdges, yCentres, yEdges, zCentres, zEdges)

        # mesh of the z axis values
        self._height = RectilinearMesh1D(cellCentres=heightCentres, cellEdges=heightEdges)

        assert self._height.nCells == self._x.nCells, Exception("heights must have enough values for {} cells or {} edges.".format(self.x.nCells, self.x.nEdges))

        self._xMesh = self.xMesh()
        self._zMesh = self.zMesh()


    def __getitem__(self, slic):
        """Slice into the mesh. """

        assert np.shape(slic) == (2, ), ValueError("slic must be over two dimensions.")

        if self.xyz:
            return TopoRectilinearMesh2D(xEdges=self._x[slic[1]], yEdges=self._y[slic[1]], zEdges=self._z[slic[0]], heightEdges=self._height[slic[1]])
        else:
            return TopoRectilinearMesh2D(xEdges=self._x[slic[1]], yEdges=self._y[slic[0]], heightEdges=self._height[slic[1]])


    @property
    def height(self):
        return self._height


    def xMesh(self, xAxis='x'):
        """Creates an array suitable for plt.pcolormesh for the abscissa.

        Parameters
        ----------
        xAxis : str
            If xAxis is 'x', the horizontal xAxis uses self.x
            If xAxis is 'y', the horizontal xAxis uses self.y
            If xAxis is 'r', the horizontal xAxis uses cumulative distance along the line.

        """

        assert xAxis in ['x', 'y', 'r'], Exception("xAxis must be either 'x', 'y' or 'r'")
        if xAxis == 'x':
            xMesh = np.repeat(self.x.cellEdges[np.newaxis, :], self.z.nCells+1, 0)
        elif xAxis == 'y':
            assert self.xyz, Exception("To plot against 'y' the mesh must be instantiated with three co-ordinates")
            xMesh = np.repeat(self.y.cellEdges[np.newaxis, :], self.z.nCells+1, 0)
        elif xAxis == 'r':
            assert self.xyz, Exception("To plot against 'r' the mesh must be instantiated with three co-ordinates")
            dx = np.diff(self.x.cellEdges)
            dy = np.diff(self.y.cellEdges)
            distance = StatArray.StatArray(np.zeros(self.x.nEdges), 'Distance', self.x.cellCentres.units)
            distance[1:] = np.cumsum(np.sqrt(dx**2.0 + dy**2.0))
            xMesh = np.repeat(distance[np.newaxis, :], self.z.nCells+1, 0)

        return xMesh


    def zMesh(self, zAxis='absolute'):
        """Creates an array suitable for plt.pcolormesh for the ordinate """
        assert zAxis.lower() in ['relative', 'absolute'], Exception("zAxis must be either 'relative' or 'absolute'")
        if zAxis.lower() == 'relative':
            return np.repeat(self.z.cellEdges[:, np.newaxis], self.x.nCells+1, 1)
        elif zAxis.lower() == 'absolute':
            return self.height.cellEdges - np.repeat(self.z.cellEdges[:, np.newaxis], self.x.nCells+1, 1)


    def pcolor(self, values, xAxis='x', zAxis='absolute', **kwargs):
        """Create a pseudocolour plot of a 2D array using the mesh.

        Parameters
        ----------
        values : array_like or StatArray
            A 2D array of colour values.
        xAxis : str
            If xAxis is 'x', the horizontal xAxis uses self.x
            If xAxis is 'y', the horizontal xAxis uses self.y
            If xAxis is 'r', the horizontal xAxis uses cumulative distance along the line
        zAxis : str
            If zAxis is 'absolute' the vertical axis is the height plus z.
            If zAxis is 'relative' the vertical axis is z.

        Other Parameters
        ----------------
        alpha : scalar or array_like, optional
            If alpha is scalar, behaves like standard matplotlib alpha and opacity is applied to entire plot
            If array_like, each pixel is given an individual alpha value.
        log : 'e' or float, optional
            Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
            Values in c that are <= 0 are masked.
        equalize : bool, optional
            Equalize the histogram of the colourmap so that all colours have an equal amount.
        nbins : int, optional
            Number of bins to use for histogram equalization.
        xscale : str, optional
            Scale the x axis? e.g. xscale = 'linear' or 'log'
        yscale : str, optional
            Scale the y axis? e.g. yscale = 'linear' or 'log'.
        flipX : bool, optional
            Flip the X axis
        flipY : bool, optional
            Flip the Y axis
        grid : bool, optional
            Plot the grid
        noColorbar : bool, optional
            Turn off the colour bar, useful if multiple customPlots plotting routines are used on the same figure.
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.

        Returns
        -------
        ax
            matplotlib .Axes

        See Also
        --------
        matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

        """
        # assert isinstance(values, StatArray), TypeError("values must be a StatArray")
        assert np.all(values.shape == self.shape), ValueError("values must have shape {} but have shape {}".format(self.shape, values.shape))

        xm = self.xMesh(xAxis=xAxis)
        zm = self.zMesh(zAxis=zAxis)

        if zAxis.lower() == 'relative':
            kwargs['flipY'] = kwargs.pop('flipY', True)
        ax, pm, cb = cP.pcolormesh(xm, zm, values, **kwargs)
        cP.xlabel(xm.getNameUnits())
        cP.ylabel(zm.getNameUnits())

        return ax, pm, cb


    def plotGrid(self, xAxis='x', **kwargs):
        """Plot the grid lines of the mesh. """

        xscale = kwargs.pop('xscale', 'linear')
        yscale = kwargs.pop('yscale', 'linear')
        flipX = kwargs.pop('flipX', False)
        flipY = kwargs.pop('flipY', False)
        c = kwargs.pop('color', 'k')

        xtmp = super().getXAxis(xAxis)

        ax = plt.gca()
        cP.pretty(ax)
        ax.vlines(x = xtmp, ymin=self._zMesh[0, :], ymax=self._zMesh[-1, :], **kwargs)
        segs = np.zeros([self.z.nEdges, self.x.nEdges, 2])
        segs[:, :, 0] = np.repeat(xtmp[np.newaxis, :], self.z.nEdges, 0)
        segs[:, :, 1] = self.height.cellEdges - np.repeat(self.z.cellEdges[:, np.newaxis], self.x.nEdges, 1)

        ls = LineCollection(segs, color='k', linestyle='solid', **kwargs)
        ax.add_collection(ls)

        dz = 0.02 * np.abs(xtmp.max() - xtmp.min())
        ax.set_xlim(xtmp.min() - dz, xtmp.max() + dz)
        dz = 0.02 * np.abs(self._zMesh.max() - self._zMesh.min())
        ax.set_ylim(self._zMesh.min() - dz, self._zMesh.max() + dz)


        plt.xscale(xscale)
        plt.yscale(yscale)
        cP.xlabel(xtmp.getNameUnits())
        cP.ylabel(self.y._cellCentres.getNameUnits())

        if flipX:
            ax.set_xlim(ax.get_xlim()[::-1])

        if flipY:
            ax.set_ylim(ax.get_ylim()[::-1])


    def plotHeight(self, xAxis='x', centres=False, **kwargs):
        """Plot the height of the mesh as a line. """

        kwargs['c'] = kwargs.pop('color', 'k')
        kwargs['linewidth'] = kwargs.pop('linewidth', 1.0)

        xtmp = super().getXAxis(xAxis, centres=centres)

        if centres:
            self.height.cellCentres.plot(xtmp, **kwargs)
        else:
            self.height.cellEdges.plot(xtmp, **kwargs)


    def vtkStructure(self):
        """Generates a vtk mesh structure that can be used in a vtk file.

        Returns
        -------
        out : pyvtk.VtkData
            Vtk data structure

        """
        # Generate the quad node locations in x
        x = self.x.cellEdges
        y = self.y.cellEdges
        z = self.z.cellEdges

        nCells = self.nCells

        z = self.z.cellEdges
        nNodes = self.x.nEdges * self.z.nEdges

        # Constuct the node locations for the vtk file
        nodes = np.empty([nNodes, 3])

        nodes[:, 0] = self.xMesh('x').reshape(self.nNodes)
        nodes[:, 1] = self.xMesh('y').reshape(self.nNodes)
        nodes[:, 2] = self.zMesh('absolute').reshape(self.nNodes)

        tmp = np.int32([0, 1, self.x.nEdges+1, self.x.nEdges])
        a = np.ones(self.x.nCells, dtype=np.int32)
        a[0] = 2
        index = (np.repeat(tmp[:, np.newaxis], nCells, 1) + np.cumsum(np.tile(a, self.z.nCells))-2).T

        return VtkData(PolyData(points=nodes, polygons=index))