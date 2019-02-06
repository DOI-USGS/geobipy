import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pylab as py
from matplotlib.collections import LineCollection as lc
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np
import numpy.ma as ma
from .fileIO import deleteFile
from ..base import Error as Err
from ..base.customFunctions import (getName, getUnits, getNameUnits, histogramEqualize, _logSomething, findFirstLastNonZeros)
from cycler import cycler
import scipy as sp

def make_colourmap(seq, cname):
    """Generate a Linear Segmented colourmap

    Generates a colourmap from the sequence given and registers the colourmap with matplotlib.

    Parameters
    ----------
    seq : array of hex colours. 
        e.g. ['#000000','#00fcfd',...]
    cname : str
        Name of the colourmap.    

    Returns
    -------
    out 
        matplotlib.colors.LinearSegmentedColormap.

    """
    nl = len(seq)
    dl = 1.0 / (len(seq))
    l = []
    for i, item in enumerate(seq):
        l.append(mcolors.hex2color(item))
        if (i < nl - 1):
            l.append((i + 1) * dl)
    l = [(0.0,) * 3, 0.0] + list(l) + [1.0, (1.0,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(l):
        if isinstance(item, float):
            r1, g1, b1 = l[i - 1]
            r2, g2, b2 = l[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    myMap = mcolors.LinearSegmentedColormap(cname, cdict, 256)
    plt.register_cmap(name=cname, cmap=myMap)
    return myMap

# Define our own colour maps in hex. Gets better range and nicer visuals.
wellSeparated = [
"#3F5D7D",'#881d67','#2e8bac','#ffcf4d','#1d3915',
'#1a8bff','#00fcfd','#0f061f','#fa249d','#00198f','#c7fe1c']

make_colourmap(wellSeparated, 'wellseparated')

tatarize = [
"#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
"#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
"#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
"#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
"#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
"#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
"#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
"#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
"#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
"#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
"#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
"#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
"#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C"]
make_colourmap(tatarize, 'tatarize')

armytage = [
"#02a53f","#b97d87","#c1d4be","#c0bcd6","#bb8477","#8e3b06","#4ae36f","#e19585",
"#e3bbb5","#b9e6af","#e0917b","#6ad33f","#3811c6","#93d58d","#c6dec7","#ead3c6",
"#f0b98d","#08ef97","#c00fcf","#9cded6","#ead5e7","#e1ebf3","#e1c4f6","#9cd4f7"]


#==============================================
# Generate default properties for GeoBIPy
#==============================================

label_size = 12

try:
    mpl.axes.rcParams['ytick.labelsize'] = label_size
except:
    assert not Err.isIpython(), 'Please use %matplotlib inline for ipython notebook on the very first line'

myFonts = {'fontsize': 12}
# mpl.rcParams['title.labelsize'] = 14
#mpl.rcParams['xtick.labelsize'] = label_size
#mpl.rcParams['ytick.labelsize'] = label_size
mpl.rc('axes', labelsize=label_size)
mpl.rc('xtick', labelsize=label_size)
mpl.rc('ytick', labelsize=label_size)
# mpl.rc('axes', labelsize='small')

mpl.rc('lines', linewidth=2, markersize=5, markeredgewidth=2, color=wellSeparated[0])
mpl.rcParams['boxplot.flierprops.markerfacecolor'] = wellSeparated[0]
mpl.rcParams['grid.alpha'] = 0.1
mpl.rcParams['axes.prop_cycle'] = cycler('color', wellSeparated)
mpl.rcParams['image.cmap'] = 'viridis'
plt.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.titlesize'] = 'small'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
#plt.rcParams.update({'figure.autolayout': True})

def pretty(ax):
    """Make a plot with nice axes. 
    
    Removes fluff from the axes.

    Parameters
    ----------
    ax : matplotlib .Axes
        A .Axes class from for example ax = plt.subplot(111), or ax = plt.gca()
    
    """
    # Remove the plot frame lines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def xlabel(label, **kwargs):
    """Create an x label with default fontsizes

    Parameters
    ----------
    label : str
        The x label.    

    """
    mpl.pyplot.xlabel(label, **myFonts, **kwargs)


def ylabel(label, **kwargs):
    """Create a y label with default fontsizes

    Parameters
    ----------
    label : str
        The y label.

    """
    mpl.pyplot.ylabel(label, **myFonts, **kwargs)


def clabel(cb, label, **kwargs):
    """Create a colourbar label with default fontsizes

    Parameters
    ----------
    cb : matplotlib.colorbar.Colorbar
        A colourbar to label
    label : str
        The colourbar label

    """
    cb.ax.set_ylabel(label, **myFonts, **kwargs)


def title(label, **kwargs):
    """Create a title with default fontsizes

    Parameters
    ----------
    label : str
        The title.

    """
    mpl.pyplot.title(label, **myFonts, **kwargs)


def suptitle(label, **kwargs):
    """Create a super title above all subplots with default font sizes

    Parameters
    ----------
    label : str
        The suptitle.

    """
    mpl.pyplot.suptitle(label, **myFonts, **kwargs)


def bar(values, x=None, i=None, **kwargs):
    """Plot a bar chart.

    Plots a bar chart and auto labels it if x and values have type geobipy.StatArray

    Parameters
    ----------
    values : array_like or StatArray
        The height of each bar.
    x : array_like or StatArray, optional
        The horizontal locations of the bars
    i : sequence of ints, optional
        Plot the ith indices of values, against the ith indices of x.

    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
    matplotlib.pyplot.bar : For additional keyword arguments you may use.

    """

    if (i is None):
        i = np.size(values)
    if (x is None):
        x = np.arange(i)
    ax = plt.gca()
    pretty(ax)
    plt.bar(x[:i], values[:i], color=wellSeparated[0], **kwargs)
    title(getName(values))
    xlabel(getName(x))
    ylabel(getNameUnits(values))
    plt.margins(0.1, 0.1)

    return ax


def hist(counts, bins, rotate=False, flipX=False, flipY=False, trim=True, normalize=False, **kwargs):
    """Plot a histogram.

    Plot of histogram of values, if density is 1, a normal is fit to the values.
    If values is an StatArray, the axes are automatically labelled.

    Parameters
    ----------
    counts : array_like or StatArray
        Compute the histogram of these values
    bins : array_like or StatArray
        Bins of the histogram

    Other Parameters
    ----------------
    log : 'e' or float, optional
        Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
        Values in c that are <= 0 are masked.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'.
    yscale : str, optional
        Scale the y axis? e.g. yscale = 'linear' or 'log'.

    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
        matplotlib.pyplot.hist : For additional keyword arguments you may use.

    """

    c = kwargs.pop('color', wellSeparated[0])
    lw = kwargs.pop('linewidth', 0.5)
    ec = kwargs.pop('edgecolor', 'k')

    ax = plt.gca()
    pretty(ax)

    if normalize:
        cnts = counts / np.trapz(counts, x = bins)
    else:
        cnts = counts

    width = np.abs(np.diff(bins))
    centres = bins[:-1] + 0.5 * (np.diff(bins))

    if (rotate):
        plt.barh(centres, cnts, height=width, align='center', color = c, linewidth = lw, edgecolor = ec, **kwargs)
        ylabel(centres.getNameUnits())
        xlabel('Frequency')
    else:
        plt.bar(centres, cnts, width=width, align='center', color = c, linewidth = lw, edgecolor = ec, **kwargs)
        xlabel(centres.getNameUnits())
        ylabel('Frequency')

    if all(counts == 0):
        trim = False
    i0 = 0
    i1 = np.size(centres) - 1
    if (trim):
        while counts[i0] == 0:
            i0 += 1
        while counts[i1] == 0:
            i1 -= 1
    if (i1 > i0):
        if (rotate):
            plt.ylim(bins[i0], bins[i1+1])
        else:
            plt.xlim(bins[i0], bins[i1+1])

    if flipX:
        ax.invert_xaxis()
        # ax.set_xlim(ax.get_xlim()[::-1])

    if flipY:
        ax.invert_yaxis()
        # ax.set_ylim(ax.get_ylim()[::-1])

    return ax


    def plotGrid(self, **kwargs):
        """Plot a set of lines contained in a line collection. """

        xscale = kwargs.pop('xscale','linear')
        yscale = kwargs.pop('yscale','linear')
        flipX = kwargs.pop('flipX',False)
        flipY = kwargs.pop('flipY',False)
        c = kwargs.pop('color', 'k')

        ax = plt.gca()
        cP.pretty(ax)
        ax.vlines(x = self.x.cellEdges, ymin=self._zMesh[0, :], ymax=self._zMesh[-1, :], **kwargs)
        segs = np.zeros([self.y.nEdges, self.x.nEdges, 2])
        segs[:, :, 0] = np.repeat(self.x.cellEdges[np.newaxis, :], self.y.nEdges, 0)
        segs[:, :, 1] = np.repeat(self.y.cellEdges[:, np.newaxis], self.x.nEdges, 1) + self.z.cellEdges

        ls = LineCollection(segs, color='k', linestyle='solid', **kwargs)
        ax.add_collection(ls)

        ax.set_xlim(self.x.displayLimits)
        dz = 0.02 * np.abs(self._zMesh.max() - self._zMesh.min())
        ax.set_ylim(self._zMesh.min() - dz, self._zMesh.max() + dz)


        plt.xscale(xscale)
        plt.yscale(yscale)
        cP.xlabel(self.x._cellCentres.getNameUnits())
        cP.ylabel(self.y._cellCentres.getNameUnits())
        
        if flipX:
            ax.invert_xaxis()
            # ax.set_xlim(ax.get_xlim()[::-1])

        if flipY:
            ax.invert_yaxis()
            # ax.set_ylim(ax.get_ylim()[::-1])


def pcolor(values, x=None, y=None, **kwargs):
    """Create a pseudocolour plot of a 2D array, Actually uses pcolormesh for speed.

    Create a colour plot of a 2D array.
    If the arrays x, y, and values are geobipy.StatArray classes, the axes can be automatically labelled.
    Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

    Parameters
    ----------
    values : array_like or StatArray
        A 2D array of colour values.
    x : 1D array_like or StatArray, optional
        Horizontal coordinates of the values edges.
    y : 1D array_like or StatArray, optional
        Vertical coordinates of the values edges.

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

    # Set the grid colour if specified.


    if (x is None):
        mx = np.arange(np.size(values,1)+1)
    else:
        mx = np.asarray(x)
        if (x.size == values.shape[1]):
            mx = x.edges()
        else:
            assert x.size == values.shape[1]+1, ValueError('x must be size {}. Not {}'.format(values.shape[1]+1, x.size))
            
    if (y is None):
        my = np.arange(np.size(values,0)+1)
    else:
        my = np.asarray(y)
        if (y.size == values.shape[0]):
            my = y.edges()
        else:
            assert y.size == values.shape[0]+1, ValueError('y must be size {}. Not {}'.format(values.shape[0]+1, y.size))

    X, Y = np.meshgrid(mx, my)

    ax = pcolormesh(X, Y, values, **kwargs)

    xlabel(getNameUnits(x))
    ylabel(getNameUnits(y))

    return ax


def pcolormesh(X, Y, values, **kwargs):
    """Create a pseudocolour plot of a 2D array, Actually uses pcolormesh for speed.

    Create a colour plot of a 2D array.
    If the arrays x, y, and values are geobipy.StatArray classes, the axes can be automatically labelled.
    Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

    Parameters
    ----------
    values : array_like or StatArray
        A 2D array of colour values.
    X : 1D array_like or StatArray, optional
        Horizontal coordinates of the values edges.
    Y : 1D array_like or StatArray, optional
        Vertical coordinates of the values edges.

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

    assert np.ndim(values) == 2, ValueError('Number of dimensions must be 2')

    ax = plt.gca()
    pretty(ax)

    equalize = kwargs.pop('equalize', False)

    log = kwargs.pop('log', False)

    xscale = kwargs.pop('xscale', 'linear')
    yscale = kwargs.pop('yscale', 'linear')
    flipX = kwargs.pop('flipX', False)
    flipY = kwargs.pop('flipY', False)
    
    cl = kwargs.pop('clabel', None)
    grid = kwargs.pop('grid', False)

    noColorBar = kwargs.pop('noColorbar', False)
    trim = kwargs.pop('trim', False)

    alpha = kwargs.pop('alpha', 1.0)

    if 'edgecolor' in kwargs:
        grid = True
    if grid:
        kwargs['edgecolor'] = kwargs.pop('edgecolor', 'k')
        kwargs['linewidth'] = kwargs.pop('linewidth', 2)

    values = ma.masked_invalid(values)

    if (log):
        values, logLabel = _logSomething(values, log)

    if equalize:
        nBins = kwargs.pop('nbins', 256)
        assert nBins > 0, ValueError('nBins must be greater than zero')
        values, dummy = histogramEqualize(values, nBins=nBins)

    Zm = ma.masked_invalid(values, copy=False)

    if np.size(alpha) == 1:
        pm = ax.pcolormesh(X, Y, Zm, alpha = alpha, **kwargs)
    else:
        pm = ax.pcolormesh(X, Y, Zm, **kwargs)

    plt.xscale(xscale)
    plt.yscale(yscale)

    if trim:
        bounds = findFirstLastNonZeros(Zm)
        ax.set_xlim(X[0, bounds[1, 0]], X[0, bounds[1, 1]])
        ax.set_ylim(Y[bounds[0, 0], 0], Y[bounds[0, 1], 0])

    if (not noColorBar):
        if (equalize):
            cbar = plt.colorbar(pm,extend='both')
        else:
            cbar = plt.colorbar(pm)

        if cl is None:
            if (log):
                clabel(cbar,logLabel+getNameUnits(values))
            else:
                clabel(cbar,getNameUnits(values))
        else:
            clabel(cbar, cl)

    if grid:
        xlim = ax.get_xlim()
        dz = 0.02 * np.abs(xlim[1] - xlim[0])
        ax.set_xlim(xlim[0] - dz, xlim[1] + dz)
        ylim = ax.get_ylim()
        dz = 0.02 * np.abs(ylim[1] - ylim[0])
        ax.set_ylim(ylim[0] - dz, ylim[1] + dz)

    if flipX:
        ax.invert_xaxis()
        # ax.set_xlim(ax.get_xlim()[::-1])

    if flipY:
        ax.invert_yaxis()
        # ax.set_ylim(ax.get_ylim()[::-1])

    if np.size(alpha) > 1:
        setAlphaPerPcolormeshPixel(pm, alpha)

    return ax


def pcolor_1D(values, y=None, **kwargs):
    """Create a pseudocolour plot of an array, Actually uses pcolormesh for speed.

    Create a colour plot of an array.
    If the arrays x, y, and values are geobipy.StatArray classes, the axes can be automatically labelled.
    Can take any other matplotlib arguments and keyword arguments e.g. cmap etc.

    Parameters
    ----------
    values : array_like or StatArray
        An array of colour values.
    x : 1D array_like or StatArray
        Horizontal coordinates of the values edges.
    y : 1D array_like or StatArray, optional
        Vertical coordinates of the values edges.

    Other Parameters
    ----------------
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
    flipY : bool, optional
        Flip the Y axis
    clabel : str, optional
        colourbar label
    grid : bool, optional
        Show the grid lines
    noColorBar : bool, optional
        Turn off the colour bar, useful if multiple customPlots plotting routines are used on the same figure.        

    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
    matplotlib.pyplot.pcolormesh : For additional keyword arguments you may use.

    """

#    assert np.ndim(values) == 2, ValueError('Number of dimensions must be 2')

    equalize = kwargs.pop('equalize', False)

    log = kwargs.pop('log', False)
    xscale = kwargs.pop('xscale', 'linear')
    yscale = kwargs.pop('yscale', 'linear')
    
    cl = kwargs.pop('clabel', None)
    grid = kwargs.pop('grid', False)

    flipY = kwargs.pop('flipY', False)

    noColorBar = kwargs.pop('noColorbar', False)

    alpha = kwargs.pop('alpha', 1.0)

    width = kwargs.pop('width', None)


    # Set the grid colour if specified
    c = None
    if grid:
        c = kwargs.pop('color', 'k')

    # Get the figure axis
    ax = plt.gca()
    pretty(ax)

    # Set the x and y axes before meshgridding them
    if (y is None):
        my = np.arange(np.size(values)+1)
        mx = np.asarray([0.0, 0.1*(np.nanmax(my)-np.nanmin(my))])
    else:
        assert y.size == values.size+1, ValueError('y must be size '+str(values.size+1))
        #my = np.hstack([np.asarray(y), y[-1]])
        my = y
        mx = np.asarray([0.0, 0.1*(np.nanmax(y)-np.nanmin(y))])

    if not width is None:
        assert width > 0.0, ValueError("width must be positive")
        mx[1] = width

    v = ma.masked_invalid(values)
    if (log):
        v,logLabel=_logSomething(v,log)

    # Append with null values to correctly use pcolormesh
    v = np.concatenate([np.atleast_2d(np.hstack([np.asarray(v),0])), np.atleast_2d(np.zeros(v.size+1))], axis=0)

    if equalize:
        nBins = kwargs.pop('nbins',256)
        assert nBins > 0, ValueError('nBins must be greater than zero')
        v,dummy = histogramEqualize(v, nBins=nBins)

    # Zm = ma.masked_invalid(v, copy=False)

    Y, X = np.meshgrid(my, mx)

    pm = ax.pcolormesh(X, Y, v, color=c, **kwargs)

    ax.set_aspect('equal')

    if flipY:
        ax.invert_yaxis()

    plt.yscale(yscale)
    ylabel(getNameUnits(y))
    ax.get_xaxis().set_ticks([])
    
    if (not noColorBar):
        if (equalize):
            cbar = plt.colorbar(pm,extend='both')
        else:
            cbar = plt.colorbar(pm)

        if cl is None:
            if (log):
                clabel(cbar,logLabel+getNameUnits(values))
            else:
                clabel(cbar,getNameUnits(values))
        else:
            clabel(cbar, cl)
    
    if np.size(alpha) > 1:
        setAlphaPerPcolormeshPixel(pm, alpha)

    return ax


def plot(x, y, **kwargs):
    """Plot y against x

    If x and y are StatArrays, the axes are automatically labelled.

    Parameters
    ----------
    x : array_like or StatArray
        The abcissa
    y : array_like or StatArray
        The ordinate, can be upto 2 dimensions.

    Other Parameters
    ----------------
    log : 'e' or float, optional
        Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
        Values in c that are <= 0 are masked.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'.
    yscale : str, optional
        Scale the y axis? e.g. yscale = 'linear' or 'log'.
    flipX : bool, optional
        Flip the X axis
    flipY : bool, optional
        Flip the Y axis
    labels : bool, optional
        Plot the labels? Default is True.

    Returns
    -------
    ax
        matplotlib.Axes

    See Also
    --------
        matplotlib.pyplot.plot : For additional keyword arguments you may use.

    """

    ax = kwargs.pop('ax', None)
    xscale = kwargs.pop('xscale','linear')
    yscale = kwargs.pop('yscale','linear')
    flipX = kwargs.pop('flipX',False)
    flipY = kwargs.pop('flipY',False)
    labels = kwargs.pop('labels', True)
    log = kwargs.pop('log', None)
    
    if (labels):
        xl = getNameUnits(x)
        yl = getNameUnits(y)

    tmp = y
    if (log):
        tmp, logLabel = _logSomething(y, log)
        yl = logLabel + yl

    if (ax is None):
        ax = plt.gca()
        pretty(ax)
    
    try:
        plt.plot(x, tmp, **kwargs)
    except:
        plt.plot(x, tmp.T, **kwargs)

    plt.xscale(xscale)
    plt.yscale(yscale)
    
    if (labels):
        xlabel(xl)
        ylabel(yl)
    plt.margins(0.1, 0.1)

    if flipX:
        ax.invert_xaxis()

    if flipY:
        ax.invert_yaxis()

    return ax


def scatter2D(x, c, y=None, i=None, *args, **kwargs):
    """Create a 2D scatter plot.

    Create a 2D scatter plot, if the y values are not given, the colours are used instead.
    If the arrays x, y, and c are geobipy.StatArray classes, the axes can be automatically labelled.
    Can take any other matplotlib arguments and keyword arguments e.g. markersize etc.

    Parameters
    ----------
    x : 1D array_like or StatArray
        Horizontal locations of the points to plot
    c : 1D array_like or StatArray
        Colour values of the points
    y : 1D array_like or StatArray, optional
        Vertical locations of the points to plot, if y = None, then y = c.
    i : sequence of ints, optional
        Plot a subset of x, y, c, using the indices in i.
    
    Other Parameters
    ----------------
    log : 'e' or float, optional
        Take the log of the colour to base 'e' if log = 'e', and a number e.g. log = 10.
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
    noColorBar : bool, optional
        Turn off the colour bar, useful if multiple customPlots plotting routines are used on the same figure.
    
    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
    matplotlib.pyplot.scatter : For additional keyword arguments you may use.

    """
    
    if (i is None):
        i = np.arange(np.size(x))
        
    # Pull options from kwargs to prevent silly crashes
    log = kwargs.pop('log',False)
    equalize = kwargs.pop('equalize',False)
    xscale = kwargs.pop('xscale','linear')
    yscale = kwargs.pop('yscale','linear')
    sl = kwargs.pop('sizeLegend', None)
    noColorBar = kwargs.pop('noColorBar', False)
    
    kwargs.pop('color',None) # Remove color which could conflict with c

    #c = kwargs.pop('c',None)
    assert (not c is None), ValueError('Must specify colour with argument "c"')

    _cLabel = kwargs.pop('clabel',getNameUnits(c))
    
    c = c[i]
    xt = x[i]
    
    iNaN = np.where(~np.isnan(c))[0]
    c = c[iNaN]
    xt = xt[iNaN]

    # Did the user ask for a log colour plot?
    if (log):
        c,logLabel = _logSomething(c,log)
    # Equalize the colours?
    if equalize:
        nBins = kwargs.pop('nbins',256)
        assert nBins > 0, ValueError('nBins must be greater than zero')
        c,dummy = histogramEqualize(c, nBins=nBins)

    # Get the yAxis values
    if (y is None):
        yt = c
    else:
        yt = y[i]
        yt = yt[iNaN]

    ax = plt.gca()
    pretty(ax)
    f = plt.scatter(xt,yt, c = c, **kwargs)

    yl = getNameUnits(yt)
    if (not noColorBar):
        if (equalize):
            cbar = plt.colorbar(f,extend='both')
        else:
            cbar = plt.colorbar(f)
        if (log):
            _cLabel = logLabel + _cLabel
            if y is None:
                yl = logLabel + yl

        clabel(cbar,_cLabel)

    if ('s' in kwargs and not sl is None):
        assert (not isinstance(sl, bool)), TypeError('sizeLegend must have type int, or array_like')
        if (isinstance(sl, int)):
            tmp0 = np.nanmin(kwargs['s'])
            tmp1 = np.nanmax(kwargs['s'])
            a = np.linspace(tmp0, tmp1, sl)
        else:
            a = sl
        sizeLegend(kwargs['s'], a)
        
    plt.xscale(xscale)
    plt.yscale(yscale)
    xlabel(getNameUnits(x,'x'))
    ylabel(yl)
    plt.margins(0.1,0.1)
    plt.grid(True)
    return ax


def setAlphaPerPcolormeshPixel(pcmesh, alphaArray):
    """Set the opacity of each pixel in a pcolormesh

    Parameters
    ----------
    pcmesh : matplotlib.collections.QuadMesh
        pcolormesh object
    alphaArray : array_like
        Values per pixel each between 0 and 1.

    """    
    plt.savefig('tmp.png')
    for i, j in zip(pcmesh.get_facecolors(), alphaArray.flatten()):
        if i[3] > 0.0:
            i[3] = j  # Set the alpha value of the RGBA tuple using a

    return pcmesh


def sizeLegend(values, intervals=None, **kwargs):
    """Add a legend to a plot if the point sizes have been specified.

    If values is an StatArray, labels are generated automatically.

    Parameters
    ----------
    values : array_like or StatArray 
        The array that was used as the size (s=) in a scatter function.
    intervals : array_like, optional
        The legend will have items at each value in intervals.
    **kwargs : dict
        kwargs are applied to plt.legend.

    """

    # Scatter parameters
    a = kwargs.pop('alpha', 0.3)
    kwargs.pop('color', None)
    c = kwargs.pop('c', 'k')
    # Legend paramters
    sp = kwargs.pop('scatterpoints', 1)
    f = kwargs.pop('frameon', False)
    ls = kwargs.pop('labelspacing', 1)

    for x in intervals:
        plt.scatter([], [], c=c, alpha=a, s=x, label=str(x) + ' ' + getUnits(values))
    plt.legend(scatterpoints=sp, frameon=f, labelspacing=ls, title=getName(values), **kwargs)


def stackplot2D(x, y, labels=[], colors=tatarize, **kwargs):
    """Plot a 2D array with column elements stacked on top of each other.

    Parameters
    ----------
    x : array_like or StatArray
        The abcissa.
    y : array_like or StatArray, 2D
        The cumulative sum along the columns is taken and stacked on top of each other.

    Other Parameters
    ----------------
    labels : list of str, optional
        The labels to assign to each column.
    colors : matplotlib.colors.LinearSegmentedColormap or list of colours
        The colour used for each column.
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'.
    yscale : str, optional
        Scale the y axis? e.g. yscale = 'linear' or 'log'.     

    Returns
    -------
    ax
        matplotlib .Axes

    See Also
    --------
        matplotlib.pyplot.scatterplot : For additional keyword arguments you may use.

    """

    xscale = kwargs.pop('xscale','linear')
    yscale = kwargs.pop('yscale','linear')
    ax = plt.gca()
    pretty(ax)

    plt.stackplot(x, y, labels=labels, colors=colors, **kwargs)

    if (not len(labels)==0): 
        plt.legend()

    plt.xscale(xscale)
    plt.yscale(yscale)
    xlabel(getNameUnits(x))
    ylabel(getNameUnits(y))
    plt.margins(0.1, 0.1)

    return ax


def step(x, y, **kwargs):
    """Plots y against x as a piecewise constant (step like) function.

    Parameters
    ----------
    x : array_like

    y : array_like

    flipY : bool, optional
        Flip the y axis
    xscale : str, optional
        Scale the x axis? e.g. xscale = 'linear' or 'log'
    yscale : str, optional
        Scale the y axis? e.g. yscale = 'linear' or 'log'
    flipX : bool, optional
        Flip the X axis
    flipY : bool, optional
        Flip the Y axis
    noLabels : bool, optional
        Do not plot the labels

    """
    # Must create a new parameter, so that the last layer is plotted
    ax = plt.gca()
    pretty(ax)

    
    flipX = kwargs.pop('flipX', False)
    flipY = kwargs.pop('flipY', False)
    noLabels = kwargs.pop('noLabels', False)
    xscale = kwargs.pop('xscale', 'linear')
    yscale = kwargs.pop('yscale', 'linear')

    kwargs['color'] = kwargs.pop('color', wellSeparated[3])

    plt.step(x=x, y=y, **kwargs)

    if (flipX):
        ax.invert_xaxis()
        # ax.set_xlim(ax.get_xlim()[::-1])

    if (flipY):
        ax.invert_yaxis()
        # ax.set_ylim(ax.get_ylim()[::-1])

    if (not noLabels):
        xlabel(getNameUnits(x))
        ylabel(getNameUnits(y))

    plt.xscale(xscale)
    plt.yscale(yscale)


def pause(interval):
    """Custom pause command to override matplotlib.pyplot.pause 
    which keeps the figure on top of all others when using interactve mode.

    Parameters
    ----------
    interval : float
        Pause for *interval* seconds.

    """
    
    backend = plt.rcParams['backend']
    if backend in mpl.rcsetup.interactive_bk:
        figManager = mpl._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return
