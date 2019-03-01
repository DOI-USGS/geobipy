""" @RectilinearMesh1D_Class
Module describing a 1D Rectilinear Mesh class
"""
from ...classes.core.myObject import myObject
from ...classes.core.StatArray import StatArray
from copy import deepcopy
import numpy as np
from ...base import customFunctions as cf
from ...base import customPlots as cp
from ...base.customFunctions import safeEval


class RectilinearMesh1D(myObject):
    """Class defining a 1D rectilinear mesh with cell centres and edges.

    Contains a simple 1D mesh with cell edges, widths, and centre locations.

    RectilinearMesh1D(cellCentres, cellEdges, edgesMin, edgesMax)

    Parameters
    ----------
    cellCentres : geobipy.StatArray, optional
        The locations of the centre of each cell. Only cellCentres or cellEdges can be given.
    cellEdges : geobipy.StatArray, optional
        The locations of the edges of each cell, including the outermost edges. Only cellCentres or cellEdges can be given.
    edgesMin : float, optional
        Only used if instantiated with cellCentres. 
        Normally the 'left' edge is calucated by centres[0] - 0.5 * (centres[1] - centres[0]).
        Instead, force the leftmost edge to be edgesMin.
    edgesMax : float, optional
        Only used if instantiated with cellCentres. 
        Normally the 'right' edge is calucated by centres[-1] - 0.5 * (centres[-1] - centres[-2]).
        Instead, force the rightmost edge to be edgesMax.

    Returns
    -------
    out : RectilinearMesh1D
        The 1D mesh.

    Raises
    ------
    Exception
        If both cellCentres and cellEdges are given.
    TypeError
        cellCentres must be a geobipy.StatArray.
    TypeError
        cellEdges must be a geobipy.StatArray.

    """


    def __init__(self, cellCentres=None, cellEdges=None, edgesMin=None, edgesMax=None):
        """ Initialize a 1D Rectilinear Mesh"""
        self._cellCentres = None
        self._cellEdges = None
        
        if (cellCentres is None and cellEdges is None):
            return
        
        assert (not(not cellCentres is None and not cellEdges is None)), Exception('Cannot instantiate with both centres and edges values')

        if not cellCentres is None:
            if isinstance(cellCentres, RectilinearMesh1D):
                self._cellCentres = cellCentres._cellCentres.deepcopy()
                self._cellEdges = cellCentres._cellEdges.deepcopy()
            else:
                assert isinstance(cellCentres, StatArray), TypeError("cellCentres must be a geobipy.StatArray")
                cellCentres = np.squeeze(cellCentres)
                ## StatArray of the x axis values
                self._cellCentres = cellCentres.deepcopy()
                self._cellEdges = self._cellCentres.edges(min=edgesMin, max=edgesMax)

        if not cellEdges is None:
            if isinstance(cellEdges, RectilinearMesh1D):
                self._cellCentres = cellEdges._cellCentres.deepcopy()
                self._cellEdges = cellEdges._cellEdges.deepcopy()
            else:
                assert isinstance(cellEdges, StatArray), TypeError("cellEdges must be a geobipy.StatArray")
                cellEdges = np.squeeze(cellEdges)
                self._cellEdges = cellEdges.deepcopy()
                self._cellCentres = cellEdges[:-1] + 0.5 * np.abs(np.diff(cellEdges))

        # Set some extra variables for speed
        # Is the discretization regular
        self.isRegular = self._cellCentres.isRegular
        # Get the increment
        self.dx = self._cellCentres[1] - self._cellCentres[0]


    def deepcopy(self):
        """Create a deepcopy

        Returns
        -------
        out : RectilinearMesh1D
            Deepcopy of RectilinearMesh1D

        """
        return deepcopy(self)


    def __deepcopy__(self, memo):
        out = RectilinearMesh1D()
        out._cellCentres = self._cellCentres.deepcopy()
        out._cellEdges = self._cellEdges.deepcopy()
        out.isRegular = self.isRegular
        out.dx = self.dx
        
        return out


    @property
    def cellCentres(self):
        return self._cellCentres


    @property
    def cellEdges(self):
        return self._cellEdges

    @property
    def displayLimits(self):
        dx = 0.02 * self.range
        return (self._cellEdges[0] - dx, self._cellEdges[-1] + dx)

    @property
    def internalCellEdges(self):
        return self._cellEdges[1:-1]

    @property
    def cellWidths(self):
        return np.abs(np.diff(self._cellEdges))

    @property
    def nCells(self):
        return self._cellCentres.size

    @property
    def nEdges(self):
        return self._cellEdges.size

    @property
    def name(self):
        return self._cellCentres.getName

    @property
    def range(self):
        """Get the difference between end edges."""
        return np.abs(self._cellEdges[-1] - self._cellEdges[0])

    @property
    def units(self):
        return self._cellCentres.getUnits


    def cellIndex(self, values, clip=False):
        """ Get the index to the cell that each value in values falls in.

        Parameters
        ----------
        values : array_like
            The values to find the cell indices for
        clip : bool
            A negative index which would normally wrap will clip to 0 and self.bins.size instead.
    
        Returns
        -------
        out : array_like
            The cell indices

        """

        if self.isRegular():
            iBin = np.int64((values - self._cellCentres[0]) / self.dx)
            if not clip:
                return np.squeeze(iBin[(iBin >= 0) & (iBin < self.nCells)])
            iBin = np.maximum(iBin,0)
            iBin = np.minimum(iBin, self.nCells - 1)
        else:
            iBin = self._cellEdges.searchsorted(values, side='right')
            iBin -= 1
            if not clip:
                return np.squeeze(iBin[(iBin >= 0) & (iBin < self.nCells)])
            iBin = np.maximum(iBin,0)
            iBin = np.minimum(iBin,self.nCells - 1)

        return np.squeeze(iBin)


    def pcolor(self, values, **kwargs):
        """Create a pseudocolour plot.

        Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

        Parameters
        ----------
        values : array_like
            The value of each cell.

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

        See Also
        --------
        geobipy.customPlots.pcolor : For non matplotlib keywords.
        matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

        """

        assert isinstance(values, StatArray), TypeError("arr must be a StatArray")
        assert values.size == self.nCells, ValueError("arr must have size nCell {}".format(self.nCells))

        ax = values.pcolor(y = self.cellEdges, **kwargs)
        
        return ax


    def plotGrid(self, **kwargs):
        """ Plot the grid lines of the mesh.
        
        See Also
        --------
        geobipy.StatArray.pcolor : For additional plotting arguments

        """

        x = StatArray(np.full(self.nCells, np.nan))
        x.pcolor(y = self.cellEdges, grid=True, noColorbar=True, **kwargs)


    def hdfName(self):
        """ Reprodicibility procedure """
        return('RectilinearMesh1D()')


    def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self._cellCentres.createHdf(grp, 'x', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        self._cellCentres.writeHdf(parent, myName+'/x',  index=index)


    def toHdf(self, h5obj, myName):
        """ Write the StatArray to an HDF object
        h5obj: :An HDF File or Group Object.
        """
        # Create a new group inside h5obj
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self._cellCentres.toHdf(grp, 'x')


    def fromHdf(self, grp, index=None):
        """ Reads in the object froma HDF file """
        item = grp.get('x')
        obj = eval(safeEval(item.attrs.get('repr')))
        x = obj.fromHdf(item, index=index)
        res = RectilinearMesh1D(x)
        return res


    def summary(self):
        """ Print a summary of self """
        self._cellCentres.summary()

