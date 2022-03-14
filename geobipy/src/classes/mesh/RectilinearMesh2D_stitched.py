""" @RectilinearMesh2D_Class
Module describing a 2D Rectilinear Mesh class with x and y axes specified
"""
from copy import copy, deepcopy
from .Mesh import Mesh
from ...classes.core import StatArray
from .RectilinearMesh2D import RectilinearMesh2D
import numpy as np
import matplotlib.cm as mplcm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from scipy.stats import binned_statistic
from ...base import plotting as cP
from ...base import utilities
# from ...base import geometry
from scipy.sparse import (kron, diags)
from scipy import interpolate

try:
    from pyvtk import VtkData, CellData, Scalars, PolyData
except:
    pass

class RectilinearMesh2D_stitched(RectilinearMesh2D):
    """Class defining stitched 1D rectilinear meshes.
    """

    def __init__(self, max_cells, x=None, relativeTo=None, nCells=None, **kwargs):
        """ Initialize a 2D Rectilinear Mesh"""

        self._max_cells = np.int32(max_cells)
        self.x = kwargs if x is None else x
        self.y_edges = None
        self.y_log = kwargs.get('ylog')
        self.relativeTo = relativeTo

        self.nCells = nCells

    @property
    def max_cells(self):
        return self._max_cells

    @property
    def nCells(self):
        return self._nCells

    @nCells.setter
    def nCells(self, values):
        if values is None:
            values = self.x.nCells
        else:
            assert np.size(values) == self.x.nCells, ValueError("Size of nCells must be {}".format(self.x.nCells))

        self._nCells = StatArray.StatArray(values, 'Number of cells', dtype=np.int32)

    @property
    def shape(self):
        return (self.x.nCells.item(), self.max_cells)

    @property
    def summary(self):
        """ Display a summary of the 3D Point Cloud """
        msg = ("2D Stitched Rectilinear Mesh: \n"
              "nCells: {} \nx\n{}").format(self.nCells.summary, self.x.summary)
        # if not self.relativeTo is None:
        #     msg += self.relativeTo.summary
        return msg

    # @property
    # def y(self):
    #     return self.y_edges

    @property
    def y_edges(self):
        return self._y_edges

    @y_edges.setter
    def y_edges(self, values):

        if values is None:
            values = (self.x.nCells, self.max_cells)
        else:
            
            assert np.ndim(values) == 2, ValueError("y_edges must have 2 dimensions")
            assert np.shape(values)[0] == self.x.nCells, ValueError("First dimension of y_edges must have size {}".format(self.x.nCells))

            values, dum = utilities._log(values, log=self.y_log)

        self._y_edges = StatArray.StatArray(values)

    def __getitem__(self, slic):
        """Allow slicing of the histogram."""
        from .RectilinearMesh1D import RectilinearMesh1D
        assert np.shape(slic) == (1,), ValueError("slic must be over 2 dimensions.")

        if isinstance(slic, (int, np.integer)):
            relativeTo = self.relativeTo[slic] if not self.relativeTo is None else None
            return RectilinearMesh1D(edges=self._z[slic, :self.nCells[slic]+2], relativeTo=relativeTo)

        else:
            slic0 = slic
            if isinstance(slic.stop, (int, np.integer)):
                # If a slice, add one to the end for bins.
                slic0 = slice(slic.start, slic.stop+1, slic.step)
            
            relativeTo = self.relativeTo[slic] if not self.relativeTo is None else None
            if self.xyz:
                out = type(self)(xEdges=self._x.edges[slic0], yEdges=self._y.edges[slic0], zEdges=self._z_edges[slic0, :], relativeTo=relativeTo)
            else:
                out = type(self)(xEdges=self._x.edges[slic0], yEdges=self._z_edges[slic0, :], relativeTo=relativeTo)
            out.nCells = self.nCells[slic]
            return out

    def pcolor(self, values, **kwargs):

        assert np.all(np.shape(values) == self.shape), ValueError("values must have shape {}".format(self.shape))

        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        cP.pretty(ax)

        xscale = kwargs.pop('xscale', 'linear')
        if self.x.log is not None:
            xscale = 'log'
        yscale = kwargs.pop('yscale', 'linear')
        if self.y_log is not None:
            yscale = 'log'
        flipX = kwargs.pop('flipX', False)
        flipY = kwargs.pop('flipY', False)

        equalize = kwargs.pop('equalize', False)
        clim_scaling = kwargs.pop('clim_scaling', None)
        colorbar = kwargs.pop('colorbar', True)
        cl = kwargs.pop('clabel', None)
        cax = kwargs.pop('cax', None)
        cmap = kwargs.pop('cmap', 'viridis')
        cmapIntervals = kwargs.pop('cmapIntervals', None)
        cmap = copy(mplcm.get_cmap(cmap, cmapIntervals))
        cmap.set_bad(color='white')
        orientation = kwargs.pop('orientation', 'vertical')
        log = kwargs.pop('log', False)

        grid = kwargs.pop('grid', False)
        if 'edgecolor' in kwargs:
            grid = True
        if grid:
            kwargs['edgecolor'] = kwargs.pop('edgecolor', 'k')
            kwargs['linewidth'] = kwargs.pop('linewidth', 2)

        if (log):
            values, logLabel = utilities._log(values, log)

        if equalize:
            nBins = kwargs.pop('nbins', 256)
            assert nBins > 0, ValueError('nBins must be greater than zero')
            values, dummy = utilities.histogramEqualize(values, nBins=nBins)

        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        v = rescale(values)

        y_edges = utilities._power(self.y_edges, self.y_log)

        # First
        bottom = y_edges[:, 0].copy()
        if self.relativeTo is not None:
            bottom += self.relativeTo

        width = y_edges[:, 1] - y_edges[:, 0]

        plt.bar(self.x.centres, width, self.x.widths, bottom=bottom, color=cmap(v[:, 0]), **kwargs)

        i = 1
        while np.any(i < self.nCells):
            active = np.where(i < self.nCells)
            bottom[active] = y_edges[active, i]
            if self.relativeTo is not None:
                bottom[active] += self.relativeTo[active]
            width = np.zeros(self.x.nCells)
            width[active] = y_edges[active, i+1] - y_edges[active, i]
            pm = plt.bar(self.x.centres, width, self.x.widths, bottom=bottom, color=cmap(v[:, i]), **kwargs)
            i += 1
        
        plt.xscale(xscale)
        plt.yscale(yscale)

        if flipX:
            ax.invert_xaxis()

        if flipY:
            ax.invert_yaxis()

        cbar = None
        if (colorbar):

            sm = mplcm.ScalarMappable(cmap=cmap, norm=plt.Normalize(np.min(values), np.max(values)))
            sm.set_array([])

            # cbar = plt.colorbar(sm)

            if (equalize):
                cbar = plt.colorbar(sm, extend='both', cax=cax, orientation=orientation)
            else:
                cbar = plt.colorbar(sm, cax=cax, orientation=orientation)

            if cl is None:
                if (log):
                    cP.clabel(cbar, logLabel + utilities.getNameUnits(values))
                else:
                    cP.clabel(cbar, utilities.getNameUnits(values))
            else:
                cP.clabel(cbar, cl)

    def createHdf(self, parent, name, withPosterior=True, add_axis=None, fillvalue=None, upcast=False):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = self.create_hdf_group(parent, name)
        self.x.createHdf(grp, 'x', withPosterior=withPosterior, fillvalue=fillvalue, upcast=upcast)
        self.y_edges.createHdf(grp, 'y/edges', withPosterior=withPosterior, fillvalue=fillvalue, upcast=upcast)
        
        if not self.relativeTo is None:
            self.relativeTo.createHdf(grp, 'y/relativeTo', withPosterior=withPosterior, add_axis=add_axis, fillvalue=fillvalue, upcast=upcast)

        if self._nCells is not None:
            self.nCells.createHdf(grp, 'nCells', withPosterior=withPosterior, fillvalue=fillvalue, upcast=upcast)

        return grp

    @classmethod
    def fromHdf(cls, grp, index=None, skip_posterior=False):
        """ Reads in the object from a HDF file """
        from .RectilinearMesh1D import RectilinearMesh1D
        if index is None:

            x = RectilinearMesh1D.fromHdf(grp['x'], skip_posterior=skip_posterior)
            nCells = StatArray.StatArray.fromHdf(grp['nCells'], skip_posterior=skip_posterior)
            edges = StatArray.StatArray.fromHdf(grp['y/edges'], skip_posterior=skip_posterior)
            
            relativeTo = None
            if 'y/relativeTo' in grp:
                relativeTo = StatArray.StatArray.fromHdf(grp['y/relativeTo'], skip_posterior=skip_posterior)
            
            out = cls(max_cells=edges.shape[1],  x=x, relativeTo=relativeTo)

            out.nCells = nCells
            out.y_edges = edges
        
        else:
            return RectilinearMesh1D.fromHdf(grp, index=index, skip_posterior=skip_posterior)
        
        return out

    