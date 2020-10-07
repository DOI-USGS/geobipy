""" @RectilinearMesh1D_Class
Module describing a 1D Rectilinear Mesh class
"""
from ...classes.core.myObject import myObject
from ...classes.core import StatArray
from copy import deepcopy
import numpy as np
from ...base import customFunctions as cF
from ...base import customPlots as cp
from scipy.sparse import diags

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
    log : 'e' or float, optional
        Entries are given in linear space, but internally cells are logged.
        Plotting is in log space.
    relativeTo : float, optional
        If a float is given, updates will be relative to this value.

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

    def __init__(self, cellCentres=None, cellEdges=None, edgesMin=None, edgesMax=None, log=None, relativeTo=0.0):
        """ Initialize a 1D Rectilinear Mesh"""
        self._cellCentres = None
        self._cellEdges = None

        self.log = log
        self.relativeTo = relativeTo

        if (cellCentres is None and cellEdges is None):
            self._cellEdges = StatArray.StatArray(0)
            self._cellCentres = StatArray.StatArray(0)
            return

        assert (not(not cellCentres is None and not cellEdges is None)), Exception('Cannot instantiate with both centres and edges values')

        if not cellCentres is None:
            self.cellCentres = cellCentres

        if not cellEdges is None:
            self.cellEdges = cellEdges

        # Is the discretization regular
        self.isRegular = self._cellCentres.isRegular
        # Get the increment
        self.dx = self._cellEdges[1] - self._cellEdges[0]


    def deepcopy(self):
        """Create a deepcopy

        Returns
        -------
        out : RectilinearMesh1D
            Deepcopy of RectilinearMesh1D

        """
        return deepcopy(self)


    def __deepcopy__(self, memo):
        out = type(self)()
        out._cellCentres = self._cellCentres.deepcopy()
        out._cellEdges = self._cellEdges.deepcopy()
        out.isRegular = self.isRegular
        out.dx = self.dx
        out._relativeTo = self._relativeTo

        return out


    def __getitem__(self, slic):
        """Slice into the class. """

        assert np.shape(slic) == (), ValueError("slic must have one dimension.")

        s2stop = None
        if isinstance(slic, slice):
            if not slic.stop is None:
                s2stop = slic.stop + 1 if slic.stop > 0 else slic.stop
            s2 = slice(slic.start, s2stop, slic.step)
        else:
            s2 = slice(slic, slic + 2, 1)

        tmp = self._cellEdges[s2]
        assert tmp.size > 1, ValueError("slic must contain at least one cell.")
        return type(self)(cellEdges=tmp)


    def __add__(self, other):
        return RectilinearMesh1D(cellEdges=self.cellCentres + other)

    def __sub__(self, other):

        return RectilinearMesh1D(cellEdges=self.cellCentres - other)

    def __mul__(self, other):

        return RectilinearMesh1D(cellEdges=self.cellCentres * other)

    def __truediv__(self, other):

        return RectilinearMesh1D(cellEdges=self.cellCentres / other)


    @property
    def cellCentres(self):
        return self._cellCentres + self.relativeTo


    @cellCentres.setter
    def cellCentres(self, values):
        if isinstance(values, RectilinearMesh1D):
            self._cellCentres = values._cellCentres.deepcopy()
            self._cellEdges = values._cellEdges.deepcopy()
        else:
            if not isinstance(values, StatArray.StatArray):
                values = StatArray.StatArray(values)

            values, _ = cF._log(values, log=self.log)
            values -= self.relativeTo

            values.name = cF._logLabel(self.log) + values.getName()

            # assert np.ndim(values) == 1, ValueError("cellCentres must be 1D")
            ## StatArray of the x axis values
            self._cellCentres = values.deepcopy()
            self._cellEdges = self._cellCentres.edges()


    @property
    def cellEdges(self):
        return self._cellEdges + self.relativeTo


    @cellEdges.setter
    def cellEdges(self, values):
        if isinstance(values, RectilinearMesh1D):
            self._cellCentres = values._cellCentres.deepcopy()
            self._cellEdges = values._cellEdges.deepcopy()
        else:
            if not isinstance(values, StatArray.StatArray):
                values = StatArray.StatArray(values)

            values, _ = cF._log(values, log=self.log)
            values -= self.relativeTo

            values.name = cF._logLabel(self.log) + values.getName()
            # assert np.ndim(values) == 1, ValueError("cellEdges must be 1D")
            self._cellEdges = values.deepcopy()
            self._cellCentres = values.internalEdges()


    @property
    def cellWidths(self):
        return np.abs(np.diff(self._cellEdges))

    @property
    def centreTocentre(self):
        return np.diff(self.cellCentres)

    @property
    def displayLimits(self):
        dx = 0.02 * self.range
        return (self._cellEdges[0] - dx, self._cellEdges[-1] + dx)

    @property
    def internalCellEdges(self):
        return self._cellEdges[1:-1]

    @property
    def nCells(self):
        return 0 if self._cellCentres is None else self._cellCentres.size

    @property
    def nEdges(self):
        return 0 if self._cellEdges is None else self._cellEdges.size

    @property
    def nNodes(self):
        return self.nEdges

    @property
    def name(self):
        return self._cellCentres.getName()

    @property
    def range(self):
        """Get the difference between end edges."""
        return np.abs(self._cellEdges[-1] - self._cellEdges[0])

    @property
    def relativeTo(self):
        return self._relativeTo

    @relativeTo.setter
    def relativeTo(self, value):
        if np.all(value > 0.0):
            value, _ = cF._log(value, self.log)
        self._relativeTo = StatArray.StatArray(value)


    @property
    def units(self):
        return self._cellCentres.getUnits()


    def cellIndex(self, values, clip=False, trim=False):
        """ Get the index to the cell that each value in values falls in.

        Parameters
        ----------
        values : array_like
            The values to find the cell indices for
        clip : bool
            A negative index which would normally wrap will clip to 0 and self.bins.size instead.
        trim : bool
            Do not include out of axis indices. Negates clip, since they wont be included in the output.

        Returns
        -------
        out : array_like
            The cell indices

        """

        values, dum = cF._log(np.atleast_1d(values).flatten(), self.log)

        values = values - self.relativeTo

        # Get the bin indices for all values
        if self.isRegular():
            iBin = np.int64((values - self._cellCentres[0]) / self.dx)
        else:
            iBin = self._cellEdges.searchsorted(values, side='right') - 1

        iBin = np.atleast_1d(iBin)

        # Remove indices that are out of bounds
        if trim:
            iBin = iBin[(values >= self._cellEdges[0]) & (values < self._cellEdges[-1])]
        else:
            # Force out of bounds to be in bounds if we are clipping
            if clip:
                iBin = np.maximum(iBin, 0)
                iBin = np.minimum(iBin, self.nCells - 1)
            # Make sure values outside the lower edge are -1
            else:
                iBin[values < self._cellEdges[0]] = -1
                iBin[values >= self._cellEdges[-1]] = self.nCells

        return np.squeeze(iBin)


    def gradientMatrix(self):
        tmp = 1.0/np.sqrt(self.centreTocentre)
        return diags([tmp, -tmp], [0, 1], shape=(self.nCells-1, self.nCells))



    def inBounds(self, values):
        """Return whether values are inside the cell edges

        Parameters
        ----------
        values : array_like
            Check if these are inside left <= values < right.

        Returns
        -------
        out : bools
            Are the values inside.

        """
        return (values >= self._cellEdges[0]) & (values < self._cellEdges[-1])


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

        assert isinstance(values, StatArray.StatArray), TypeError("arr must be a StatArray")
        assert values.size == self.nCells, ValueError("arr must have size nCell {}".format(self.nCells))

        kwargs['y'] = kwargs.pop('y', self.cellEdges)

        ax = values.pcolor(**kwargs)

        return ax


    def plotGrid(self, **kwargs):
        """ Plot the grid lines of the mesh.

        See Also
        --------
        geobipy.StatArray.pcolor : For additional plotting arguments

        """

        x = StatArray.StatArray(np.full(self.nCells, np.nan))
        x.pcolor(y = self.cellEdges, grid=True, noColorbar=True, **kwargs)


    def hdfName(self):
        """ Reprodicibility procedure """
        return('RectilinearMesh1D()')


    def createHdf(self, parent, myName, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self._cellCentres.createHdf(grp, 'x', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        self._cellCentres.writeHdf(parent, myName+'/x',  withPosterior=withPosterior, index=index)


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
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        x = obj.fromHdf(item, index=index)
        res = RectilinearMesh1D(x)
        return res


    def summary(self, out=False):
        """ Print a summary of self """
        msg = ("Cell Centres \n"
               "{}"
               "Cell Edges"
               "{}").format(
                   self._cellCentres.summary(True),
                   self._cellEdges.summary(True)
               )

        return msg if out else print(msg)


    def Bcast(self, world, root=0):

        if world.rank == root:
            edges = self.cellEdges
        else:
            edges = StatArray.StatArray(0)

        edges = self.cellEdges.Bcast(world, root=root)

        return RectilinearMesh1D(cellEdges=edges)



